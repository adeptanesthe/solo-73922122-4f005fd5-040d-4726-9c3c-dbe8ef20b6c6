"""关键功能自测（开发环境运行，不进镜像）：

  python -m tests.selftest

覆盖：
 1. 模型规格：5 棵树、深度 ≤ 3、同种子重训结果一致；
 2. 阈值语义：cutoff_left 对全部树阈值 × 全部克数逐一与 float32 比较核对；
    随机配方上纯 Python 走树概率与 sklearn predict_proba 完全一致；
 3. 可行样例：求解 ok、概率复算达标、改动量一致，并用 z3 反证 + sklearn 暴力
    枚举双重核对最优性结论；
 4. 无解样例：z3 判 unsat，并对整个可行域用 sklearn 暴力枚举复核确无合格配方；
 5. 字典序：改 2 种（量大）与改 3 种（量小）并存时，必须返回改 2 种；
 6. 边界：原配方已合格（0 改动）、全锁定合格/不合格、各类非法输入；
 7. 求解失败路径：异常注入返回 solver_failure 而非无解；
 8. API：/、/api/meta、/api/samples、/api/solve 各状态分支；页面含迟到结果丢弃逻辑。
 9. 批次核查：参考集（1000 条训练外、整数克数、固定种子、float32 路由）、
    文本解析七类非法输入、统计量手工核对、置换 p 值公式与确定性、同分布/强偏移
    两分支、20/200 行边界、/api/drift 与 /api/drift/example 各分支。
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import z3

from app.model import (
    MAX_DEPTH,
    N_TREES,
    TOTAL_GRAMS,
    cutoff_left,
    extract_forest,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
    train_model,
)
from app.solver import (
    INFEASIBLE,
    INVALID,
    OK,
    SOLVER_FAILURE,
    SolveInput,
    _build_optimize,
    solve,
)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    print("== 1. 模型规格 ==")
    clf = train_model()
    forest = extract_forest(clf)
    check("树数量为 5", len(clf.estimators_) == N_TREES == 5)
    depths = [e.tree_.max_depth for e in clf.estimators_]
    check("树深均 ≤ 3", all(d <= MAX_DEPTH for d in depths), f"depths={depths}")
    clf2 = train_model()
    same = np.allclose(
        clf.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
        clf2.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
    )
    check("同种子重训结果一致", same)

    print("== 2. 阈值语义 ==")
    # cutoff_left：对森林中全部阈值 × 全部整数克数，逐一与 float32 比较核对
    thr_ok = True
    for tree in forest.trees:
        for node in tree.nodes:
            if node.__class__.__name__ == "Leaf":
                continue
            k = cutoff_left(node.threshold)
            for g in range(TOTAL_GRAMS + 1):
                expect = bool(np.float32(g / TOTAL_GRAMS) <= node.threshold)
                if (g <= k) != expect:
                    thr_ok = False
                    break
            if not thr_ok:
                break
    check("cutoff_left 与 float32 比较逐点一致", thr_ok)

    rng = np.random.default_rng(99)
    proba_ok = True
    for _ in range(500):
        g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        p_sk = predict_proba_qualified(clf, g)
        p_py = float(forest_proba_rational(forest, g))
        if abs(p_sk - p_py) > 1e-12:
            proba_ok = False
            break
    check("500 个随机配方概率与 sklearn 完全一致", proba_ok)

    print("== 3. 可行样例 ==")
    from app.samples import build_samples

    samples = build_samples(forest, clf)
    fs = samples["feasible"]
    inp = SolveInput(fs["original"], fs["lower"], fs["upper"], fs["locked"])
    res = solve(forest, clf, inp)
    check("可行样例求解为 ok", res.status == OK, res.reason)
    check("建议概率 ≥ 0.7（原分类器复算）", res.probability is not None and is_qualified(res.probability),
          f"p={res.probability:.6f}")
    sug = res.suggestion
    check("建议满足总重/上下限/锁定",
          sum(sug) == TOTAL_GRAMS
          and all(fs["lower"][i] <= sug[i] <= fs["upper"][i] for i in range(4))
          and all(sug[i] == fs["original"][i] for i in range(4) if fs["locked"][i]))
    k, m = res.changed_count, res.total_change
    real_k = sum(1 for i in range(4) if sug[i] != fs["original"][i])
    real_m = sum(abs(sug[i] - fs["original"][i]) for i in range(4))
    check("报告的改动种数/克数与建议一致", (k, m) == (real_k, real_m), f"k={k}, m={m}")

    # 3a. z3 反证最优性
    opt, g, num_changed, total_change = _build_optimize(forest, inp)
    opt.add(num_changed <= k - 1)
    check("z3 反证：更少种数无解", opt.check() == z3.unsat)
    opt2, g2, num_changed2, total_change2 = _build_optimize(forest, inp)
    opt2.add(num_changed2 == k, total_change2 <= m - 1)
    check("z3 反证：同种数更小改动量无解", opt2.check() == z3.unsat)

    # 3b. sklearn 暴力枚举复核最优性（改动 ≤2 种且更优者不存在）
    def brute_best(inp_d, max_kinds):
        orig, lo, hi, lk = inp_d["original"], inp_d["lower"], inp_d["upper"], inp_d["locked"]
        free = [i for i in range(4) if not lk[i]]
        best = None  # (kinds, delta)
        for n_k in range(0, max_kinds + 1):
            for S in itertools.combinations(free, n_k):
                fixed_idx = [i for i in range(4) if i not in S]
                fixed_sum = sum(orig[i] for i in fixed_idx)
                if len(S) == 0:
                    cand = [tuple(orig)]
                elif len(S) == 1:
                    i = S[0]
                    gi = TOTAL_GRAMS - fixed_sum
                    cand = [] if not (lo[i] <= gi <= hi[i]) else [tuple(orig[:i]) + (gi,) + tuple(orig[i+1:])]
                else:  # len(S) == 2：和约束下第二变量被唯一确定
                    i, j = S
                    cand = []
                    for gi in range(lo[i], hi[i] + 1):
                        gj = TOTAL_GRAMS - fixed_sum - gi
                        if lo[j] <= gj <= hi[j]:
                            row = list(orig); row[i], row[j] = gi, gj
                            cand.append(tuple(row))
                if not cand:
                    continue
                arr = np.array(cand, dtype=np.float32) / TOTAL_GRAMS
                probs = clf.predict_proba(arr)[:, 1]
                for row, p in zip(cand, probs):
                    if p >= 0.7 - 1e-9:
                        d = sum(abs(row[t] - orig[t]) for t in range(4))
                        cur = (n_k, d)
                        if best is None or cur < best:
                            best = cur
        return best

    bb = brute_best(fs, 2)
    check("sklearn 暴力枚举复核最优性", bb == (k, m), f"brute={bb}, solver=({k},{m})")

    print("== 4. 无解样例 ==")
    isf = samples["infeasible"]
    res_i = solve(forest, clf, SolveInput(isf["original"], isf["lower"], isf["upper"], isf["locked"]))
    check("无解样例求解为 infeasible", res_i.status == INFEASIBLE, res_i.reason)
    # 对整个可行域用 sklearn 暴力枚举复核“确实无解”
    lo, hi = isf["lower"], isf["upper"]
    combos = []
    for g0 in range(lo[0], hi[0] + 1):
        for g1 in range(lo[1], hi[1] + 1):
            for g2 in range(lo[2], hi[2] + 1):
                g3 = TOTAL_GRAMS - g0 - g1 - g2
                if lo[3] <= g3 <= hi[3]:
                    combos.append((g0, g1, g2, g3))
    found_q = False
    for s in range(0, len(combos), 2_000_000):
        arr = np.array(combos[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
        if (clf.predict_proba(arr)[:, 1] >= 0.7 - 1e-9).any():
            found_q = True
            break
    check(f"sklearn 全可行域枚举（{len(combos)} 点）确无合格配方", not found_q)

    print("== 5. 字典序优先级 ==")
    lex_orig = [244, 294, 218, 244]  # 改 2 种最少 218 克；改 3 种可 214 克
    res_lex = solve(forest, clf, SolveInput(lex_orig, [0]*4, [1000]*4, [False]*4))
    check("先最少种数再最小克数", res_lex.status == OK and res_lex.changed_count == 2
          and res_lex.total_change == 218,
          f"got kinds={res_lex.changed_count}, delta={res_lex.total_change}")

    print("== 6. 边界与非法输入 ==")
    r_ok0 = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [False]*4))
    check("原配方已合格 → 0 改动", r_ok0.status == OK and r_ok0.changed_count == 0
          and r_ok0.total_change == 0 and r_ok0.suggestion == [700, 100, 100, 100])
    r_lock_bad = solve(forest, clf, SolveInput([150, 130, 620, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且不合格 → infeasible", r_lock_bad.status == INFEASIBLE)
    r_lock_ok = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且合格 → ok 0 改动", r_lock_ok.status == OK and r_lock_ok.changed_count == 0)
    bad_cases = [
        SolveInput([250, 250, 350, 149], [0]*4, [1000]*4, [False]*4),   # 总和 999
        SolveInput([250, 250, 350, 150], [0, 0, 500, 0], [1000, 1000, 400, 1000], [False]*4),  # lo>hi
        SolveInput([250, 250, 350, 150], [300, 0, 0, 0], [1000]*4, [True, False, False, False]),  # 锁定值越界
        SolveInput([250, 250, 350, 150], [600, 600, 0, 0], [1000]*4, [False]*4),  # 下限和超 1000
        SolveInput([250, 250, 350, 150], [0]*4, [200, 200, 200, 200], [False]*4),  # 上限和不足 1000
    ]
    check("非法输入均判 invalid", all(solve(forest, clf, c).status == INVALID for c in bad_cases))

    print("== 7. 求解失败路径 ==")
    import app.solver as sv
    orig_build = sv._build_optimize
    def boom(forest_, inp_):
        raise z3.Z3Exception("injected")
    sv._build_optimize = boom
    try:
        r_fail = solve(forest, clf, SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False]*4))
    finally:
        sv._build_optimize = orig_build
    check("求解器异常 → solver_failure（非无解）", r_fail.status == SOLVER_FAILURE)

    print("== 8. API ==")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/")
    check("GET / 返回网页", r.status_code == 200 and "开始求解" in r.text)
    r = client.get("/api/meta")
    mj = r.json()
    check("GET /api/meta", r.status_code == 200 and mj["n_trees"] == 5 and mj["max_depth"] == 3
          and abs(mj["proba_threshold"] - 0.7) < 1e-12)
    r = client.get("/api/samples")
    sj = r.json()
    check("GET /api/samples 两类样例齐全", r.status_code == 200
          and sj["feasible"]["check"]["status"] == "ok"
          and sj["infeasible"]["check"]["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": fs["original"], "lower": fs["lower"],
                                        "upper": fs["upper"], "locked": fs["locked"]})
    rj = r.json()
    check("POST /api/solve 可行分支", r.status_code == 200 and rj["status"] == "ok"
          and rj["probability"] >= 0.7 and rj["suggestion"] == sug)
    r = client.post("/api/solve", json={"original": isf["original"], "lower": isf["lower"],
                                        "upper": isf["upper"], "locked": isf["locked"]})
    check("POST /api/solve 无解分支", r.json()["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": [1, 2, 3, 4], "lower": [0]*4,
                                        "upper": [1000]*4, "locked": [False]*4})
    check("POST /api/solve 非法输入分支", r.json()["status"] == "invalid")
    html = client.get("/").text
    check("页面含迟到结果丢弃逻辑", "my !== epoch" in html and "abort" in html)

    print("== 9. 新增批次适用性核查 ==")
    from app import drift as drift_mod
    from app.main import (
        LEAF_IDS_BY_TREE,
        NODE_COUNTS,
        REF_GRAMS,
        REF_LEAVES,
    )

    # 9.1 参考配方：1000 条、四料整数、总重 1000、取值合法、固定种子可复现
    check("参考集 1000 条且全部总重 1000",
          REF_GRAMS.shape == (1000, 4) and bool((REF_GRAMS.sum(axis=1) == TOTAL_GRAMS).all()))
    check("参考集均为 [0,1000] 整数",
          bool(((REF_GRAMS >= 0) & (REF_GRAMS <= TOTAL_GRAMS)).all())
          and np.issubdtype(REF_GRAMS.dtype, np.integer))
    ref_again = drift_mod.generate_reference_recipes()
    check("参考集固定种子可复现", np.array_equal(ref_again, REF_GRAMS))
    check("参考种子与训练种子不同（训练外）",
          drift_mod.REFERENCE_SEED != 20241117)
    check("参考集预路由为五棵树叶 id", REF_LEAVES.shape == (1000, 5))
    # float32 语义：参考集路由与 clf.apply(float32(g/1000)) 完全一致
    check("叶节点路由符合 float32 输入语义",
          np.array_equal(drift_mod.route_leaves(clf, REF_GRAMS), REF_LEAVES))

    # 9.2 文本解析：合法、各类非法、行数边界
    good_text = drift_mod.example_batch_text(30)
    parsed = drift_mod.parse_batch_text(good_text)
    check("合法批次文本解析通过", parsed.rows is not None and len(parsed.rows) == 30
          and all(sum(r) == TOTAL_GRAMS for r in parsed.rows))
    bad_cases_drift = [
        "",                                                    # 空
        "1 2 3\n" + good_text,                                 # 行字段数错
        "250.5 250 250 249.5\n" + good_text,                   # 小数
        "250 250 250 249\n" + good_text,                       # 总重 999
        "1001 -1 0 0\n" + good_text,                           # 越界/负数
        "\n".join("250 250 250 250" for _ in range(19)),       # 行数不足
        "\n".join("250 250 250 250" for _ in range(201)),      # 行数超限
    ]
    invalid_flags = []
    for tc in bad_cases_drift:
        p = drift_mod.parse_batch_text(tc)
        invalid_flags.append(p.rows is None and len(p.errors) > 0)
    check("七类非法输入均被拒绝且带行号错误", all(invalid_flags), f"{invalid_flags}")
    comma_text = "250,250,250,250\n" + "\n".join("250 250 250 250" for _ in range(29))
    p_comma = drift_mod.parse_batch_text(comma_text)
    check("逗号分隔的合法文本可解析", p_comma.rows is not None and len(p_comma.rows) == 30)

    # 9.3 统计量：手工核对一组占比差
    rng_d = np.random.default_rng(321)
    in_dist = drift_mod._integer_compose(
        rng_d.dirichlet(drift_mod.DIRICHLET_ALPHA, size=40)
    ).tolist()
    res_ok = drift_mod.run_drift_check(clf, REF_LEAVES, LEAF_IDS_BY_TREE, NODE_COUNTS, in_dist)
    # 手工重算统计量与最差叶节点
    b_leaves = drift_mod.route_leaves(clf, np.asarray(in_dist))
    manual_stat, manual_worst = drift_mod.max_leaf_diff(
        REF_LEAVES, b_leaves, LEAF_IDS_BY_TREE, NODE_COUNTS
    )
    check("返回统计量与手工重算一致",
          abs(res_ok["statistic"] - manual_stat) < 1e-15
          and res_ok["worst_leaf"]["leaf_id"] == manual_worst["leaf_id"]
          and res_ok["worst_leaf"]["tree"] == manual_worst["tree"])
    # 占比口径核对：最差叶节点上的两组计数/组大小
    w = manual_worst
    t = w["tree_index"]
    bp = float((b_leaves[:, t] == w["leaf_id"]).sum()) / len(in_dist)
    rp = float((REF_LEAVES[:, t] == w["leaf_id"]).sum()) / 1000
    check("最差叶节点两组占比口径正确",
          abs(bp - w["batch_proportion"]) < 1e-15
          and abs(rp - w["reference_proportion"]) < 1e-15
          and abs(abs(bp - rp) - w["abs_diff"]) < 1e-15)

    # 9.4 置换 p 值：999 次、公式 (1+exceed)/1000、范围 [1/1000,1]、确定性
    check("p 值为含观测值的无偏估计",
          res_ok["p_value"] == (1 + res_ok["permutation_exceed"]) / 1000
          and 1 / 1000 - 1e-12 <= res_ok["p_value"] <= 1.0)
    check("置换次数固定 999", res_ok["n_permutations"] == 999)
    res_ok2 = drift_mod.run_drift_check(clf, REF_LEAVES, LEAF_IDS_BY_TREE, NODE_COUNTS, in_dist)
    check("同输入重复核查结果完全一致",
          res_ok["p_value"] == res_ok2["p_value"]
          and res_ok["statistic"] == res_ok2["statistic"])

    # 9.5 同分布批次通常不触发复核；强偏移批次必须 p<=0.05 并给人工复核
    check("同分布批次未检出偏移（p>0.05）", res_ok["p_value"] > 0.05, f"p={res_ok['p_value']}")
    rng_s = np.random.default_rng(555)
    shifted = drift_mod._integer_compose(
        rng_s.dirichlet([0.5, 0.5, 7.0, 0.6], size=40)
    ).tolist()
    res_shift = drift_mod.run_drift_check(
        clf, REF_LEAVES, LEAF_IDS_BY_TREE, NODE_COUNTS, shifted
    )
    check("强偏移批次触发人工复核（p<=0.05）",
          res_shift["p_value"] <= 0.05, f"p={res_shift['p_value']}")
    check("偏移批次统计量显著大于同分布批次",
          res_shift["statistic"] > res_ok["statistic"])

    # 9.6 边界组大小：20 行与 200 行均可完成核查
    rng_b = np.random.default_rng(77)
    for n_b in (20, 200):
        rows_b = drift_mod._integer_compose(
            rng_b.dirichlet(drift_mod.DIRICHLET_ALPHA, size=n_b)
        ).tolist()
        rb = drift_mod.run_drift_check(clf, REF_LEAVES, LEAF_IDS_BY_TREE, NODE_COUNTS, rows_b)
        check(f"{n_b} 行批次可完成核查", rb["n_rows"] == n_b and 0 < rb["p_value"] <= 1)

    # 9.7 API：/api/meta 新参数、/api/drift 三分支、/api/drift/example、页面区块
    mj2 = client.get("/api/meta").json()
    check("meta 含漂移核查参数",
          mj2["n_reference"] == 1000 and mj2["n_permutations"] == 999
          and abs(mj2["drift_alpha"] - 0.05) < 1e-12 and "reference_seed" in mj2)
    r = client.post("/api/drift", json={"text": good_text})
    rj = r.json()
    check("POST /api/drift 合法批次返回结论", r.status_code == 200
          and rj["status"] in ("review", "no_shift")
          and "worst_leaf" in rj and {"tree", "leaf_id", "batch_proportion",
                                      "reference_proportion", "abs_diff"} <= set(rj["worst_leaf"]))
    r = client.post("/api/drift", json={"text": bad_cases_drift[0]})
    rj = r.json()
    check("POST /api/drift 非法输入不出结论", r.status_code == 200
          and rj["status"] == "invalid" and "p_value" not in rj and len(rj["errors"]) >= 1)
    r = client.post("/api/drift", json={"text": "x"})
    check("POST /api/drift 脏数据判 invalid", r.json()["status"] == "invalid")
    r = client.get("/api/drift/example")
    ex = r.json()
    check("GET /api/drift/example 返回可解析合法批次",
          drift_mod.parse_batch_text(ex["text"]).rows is not None)
    check("页面含批次核查区块与独立撤回逻辑",
          "runDrift" in html and "driftEpoch" in html and "onBatchEdit" in html
          and "my !== driftEpoch" in html and "未检出偏移" in html and "人工复核" in html)

    print()
    if failures:
        print(f"共 {len(failures)} 项失败：{failures}")
        return 1
    print("全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
