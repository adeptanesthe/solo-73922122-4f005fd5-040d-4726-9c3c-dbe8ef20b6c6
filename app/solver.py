"""z3 求解：在“合格概率 >= 0.7”约束下，按字典序优化
  1) 改变原料种数最少；2) 总克数改变量最小。

约束按原分类器的真实阈值语义编码：sklearn 树内部以 float32 特征与
float64 阈值比较，这里把每个分裂条件精确翻译成整数克数比较（g <= K），
K 由 model.cutoff_left 预计算。每份建议解出后都会用原 sklearn 分类器复算。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction

import z3

from .model import (
    FEATURE_NAMES,
    ForestSpec,
    Leaf,
    N_FEATURES,
    N_TREES,
    PROBA_THRESHOLD,
    TOTAL_GRAMS,
    cutoff_left,
    is_qualified,
    predict_proba_qualified,
)

SOLVER_TIMEOUT_MS = 60_000

# 状态取值
OK = "ok"                      # 已找到并证明最优
INFEASIBLE = "infeasible"       # 已证明无解
INVALID = "invalid"             # 输入本身不合法
SOLVER_FAILURE = "solver_failure"  # 求解器未知/超时/异常，或复核不一致


@dataclass
class SolveInput:
    original: list[int]
    lower: list[int]
    upper: list[int]
    locked: list[bool]


@dataclass
class SolveResult:
    status: str
    reason: str = ""
    suggestion: list[int] | None = None
    probability: float | None = None       # 原分类器复算概率
    changed_count: int | None = None
    total_change: int | None = None
    per_ingredient: list[dict] | None = None
    solve_ms: float = 0.0
    optimality: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "suggestion": self.suggestion,
            "probability": self.probability,
            "changed_count": self.changed_count,
            "total_change": self.total_change,
            "per_ingredient": self.per_ingredient,
            "solve_ms": round(self.solve_ms, 1),
            "optimality": self.optimality,
        }


def validate(inp: SolveInput) -> list[str]:
    errs = []
    for name, arr in (("original", inp.original), ("lower", inp.lower), ("upper", inp.upper)):
        if len(arr) != N_FEATURES or any((not isinstance(v, int)) or isinstance(v, bool) for v in arr):
            errs.append(f"{name} 必须是 {N_FEATURES} 个整数")
    if len(inp.locked) != N_FEATURES or any(not isinstance(v, bool) for v in inp.locked):
        errs.append(f"locked 必须是 {N_FEATURES} 个布尔值")
    if errs:
        return errs
    o, lo, hi, lk = inp.original, inp.lower, inp.upper, inp.locked
    if any(g < 0 for g in o):
        errs.append("原配方克数不能为负")
    if sum(o) != TOTAL_GRAMS:
        errs.append(f"原配方总克数必须等于 {TOTAL_GRAMS}（当前 {sum(o)}）")
    for i in range(N_FEATURES):
        if not (0 <= lo[i] <= hi[i] <= TOTAL_GRAMS):
            errs.append(f"原料{i+1} 的上下限不合法（需 0 <= 下限 <= 上限 <= {TOTAL_GRAMS}）")
        elif lk[i] and not (lo[i] <= o[i] <= hi[i]):
            errs.append(f"原料{i+1} 已锁定，但原用量 {o[i]} 不在限制区间 [{lo[i]}, {hi[i]}] 内")
    if sum(lo) > TOTAL_GRAMS:
        errs.append(f"各原料下限之和 {sum(lo)} 超过总重 {TOTAL_GRAMS}，不可能配平")
    if sum(hi) < TOTAL_GRAMS:
        errs.append(f"各原料上限之和 {sum(hi)} 低于总重 {TOTAL_GRAMS}，不可能配平")
    return errs


def _build_optimize(forest: ForestSpec, inp: SolveInput):
    """构造 Optimize 对象与全部中间变量（供求解与自测复用）。"""
    g = z3.IntVector("g", N_FEATURES)
    opt = z3.Optimize()
    opt.set("timeout", SOLVER_TIMEOUT_MS)
    for i in range(N_FEATURES):
        opt.add(g[i] >= inp.lower[i], g[i] <= inp.upper[i])
        if inp.locked[i]:
            opt.add(g[i] == inp.original[i])
    opt.add(z3.Sum(g) == TOTAL_GRAMS)

    # ---- 森林约束：每棵树恰好落在一个叶子，概率和达到阈值 ----
    tree_probs = []
    for ti, tree in enumerate(forest.trees):
        # 该树的父指针表：子节点 id -> (父节点 id, 是否走左分支)
        parent = {}
        for pid, pn in enumerate(tree.nodes):
            if not isinstance(pn, Leaf):
                parent[pn.left] = (pid, True)
                parent[pn.right] = (pid, False)
        leaf_selects = []
        for nid, node in enumerate(tree.nodes):
            if not isinstance(node, Leaf):
                continue
            # 回溯根到该叶子的路径条件
            conds = []
            cur = nid
            while cur in parent:
                pid, went_left = parent[cur]
                pn = tree.nodes[pid]
                k = cutoff_left(pn.threshold)
                conds.append(g[pn.feature] <= k if went_left else g[pn.feature] >= k + 1)
                cur = pid
            b = z3.Bool(f"t{ti}_leaf{nid}")
            opt.add(b == z3.And(*conds))
            leaf_selects.append((b, node.prob1))
        opt.add(z3.PbEq([(b, 1) for b, _ in leaf_selects], 1))  # 恰好一个叶子生效
        tree_probs.append(
            z3.Sum([z3.If(b, z3.Q(p.numerator, p.denominator), 0) for b, p in leaf_selects])
        )
    # mean(p_t) >= 0.7  <=>  sum(p_t) >= 3.5；留 1e-9 量级容差与复核口径一致
    threshold_sum = Fraction(N_TREES) * Fraction(PROBA_THRESHOLD) - Fraction(1, 10**9)
    opt.add(z3.Sum(tree_probs) >= z3.Q(threshold_sum.numerator, threshold_sum.denominator))

    # ---- 改动量变量 ----
    changed, abs_delta = [], []
    for i in range(N_FEATURES):
        if inp.locked[i]:
            continue
        c = z3.Bool(f"chg{i}")
        opt.add(c == (g[i] != inp.original[i]))
        changed.append(c)
        abs_delta.append(
            z3.If(g[i] >= inp.original[i], g[i] - inp.original[i], inp.original[i] - g[i])
        )
    num_changed = z3.Sum([z3.If(c, 1, 0) for c in changed]) if changed else z3.IntVal(0)
    total_change = z3.Sum(abs_delta) if abs_delta else z3.IntVal(0)
    return opt, g, num_changed, total_change


def solve(forest: ForestSpec, clf, inp: SolveInput) -> SolveResult:
    """两阶段字典序求解；只有 z3 证明 unsat 才报无解。"""
    t0 = time.perf_counter()
    errs = validate(inp)
    if errs:
        return SolveResult(status=INVALID, reason="；".join(errs))
    try:
        opt, g, num_changed, total_change = _build_optimize(forest, inp)
        # 阶段 1：最少改变原料种数
        h1 = opt.minimize(num_changed)
        r1 = opt.check()
        if r1 == z3.unsat:
            return SolveResult(
                status=INFEASIBLE,
                reason="求解器已证明：在当前上下限与锁定条件下，不存在合格概率 ≥ 0.7 的配方。",
                solve_ms=(time.perf_counter() - t0) * 1000,
            )
        if r1 != z3.sat:
            return SolveResult(
                status=SOLVER_FAILURE,
                reason=f"求解器未能判定（{r1}，可能超时），既未找到方案也未证明无解。",
                solve_ms=(time.perf_counter() - t0) * 1000,
            )
        best_k = opt.lower(h1).as_long()
        # 阶段 2：在种数最优的前提下最小化总克数改变量
        opt.add(num_changed == best_k)
        h2 = opt.minimize(total_change)
        r2 = opt.check()
        if r2 != z3.sat:
            return SolveResult(
                status=SOLVER_FAILURE,
                reason=f"第二阶段求解未能判定（{r2}）。",
                solve_ms=(time.perf_counter() - t0) * 1000,
            )
        best_delta = opt.lower(h2).as_long()
        m = opt.model()
        sug = [m.eval(g[i]).as_long() for i in range(N_FEATURES)]
    except z3.Z3Exception as exc:  # 求解器异常：与“无解”严格区分
        return SolveResult(
            status=SOLVER_FAILURE,
            reason=f"求解器异常：{exc}",
            solve_ms=(time.perf_counter() - t0) * 1000,
        )

    # ---- 用原分类器复算每份建议 ----
    proba = predict_proba_qualified(clf, sug)
    elapsed = (time.perf_counter() - t0) * 1000
    if not is_qualified(proba):
        return SolveResult(
            status=SOLVER_FAILURE,
            reason=f"内部不一致：求解器给出的配方经原分类器复核概率为 {proba:.6f}，未达阈值。",
            solve_ms=elapsed,
        )
    per = [
        {
            "name": FEATURE_NAMES[i],
            "original": inp.original[i],
            "new": sug[i],
            "delta": sug[i] - inp.original[i],
            "locked": inp.locked[i],
        }
        for i in range(N_FEATURES)
    ]
    optimality = (
        f"已证明最优：不存在改变少于 {best_k} 种原料的合格方案；"
        f"在所有只改变 {best_k} 种原料的合格方案中，不存在总改变量小于 {best_delta} 克的方案。"
    )
    if best_k == 0:
        optimality = "原配方本身已合格（概率 ≥ 0.7），无需任何改动，改动量为 0 已是最优。"
    return SolveResult(
        status=OK,
        suggestion=sug,
        probability=proba,
        changed_count=best_k,
        total_change=best_delta,
        per_ingredient=per,
        solve_ms=elapsed,
        optimality=optimality,
    )
