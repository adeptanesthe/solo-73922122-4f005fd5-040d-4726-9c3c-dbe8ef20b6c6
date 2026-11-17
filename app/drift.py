"""新增批次适用性核查：五棵树叶节点占比差 + 联合置换检验。

流程（全部由固定种子决定，可复现）：
  1. 以与训练数据不同的固定种子另行生成 1000 条**训练外**参考配方——与训练
     数据同属 Dirichlet 生成族（即“熟悉的配方范围”），但落为总重 1000 克的
     四料整数克数；
  2. 按模型的 float32 输入语义（克数 / 1000 后转 float32 再走 sklearn 分裂
     比较）把参考组与待检批次路由到五棵树各自的叶节点；
  3. 逐树逐叶节点比较两组样本占比，取所有叶节点占比差的最大绝对值为统计量 T；
  4. 合并两组样本，保持批次组大小不变做 999 次联合置换，按
     p = (1 + #{T_perm >= T_obs}) / (1 + 999) 估计整体 p 值；
  5. p <= 0.05 才提示人工复核；否则只报“未检出偏移”，不做放行判断。

输入解析失败时只返回错误，不产生任何统计结论。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from .model import N_FEATURES, N_TREES, TOTAL_GRAMS

# ---- 固定超参 ----
REFERENCE_SEED = 20241118        # 与训练种子 TRAIN_SEED(20241117) 不同，保证“另行生成”
PERMUTATION_SEED = 20241119      # 置换流固定种子：同输入永远得到同一 p 值
EXAMPLE_SEED = 20241120          # 网页“填入示例批次”用
N_REFERENCE = 1000               # 参考配方条数（训练外）
N_PERMUTATIONS = 999             # 联合置换次数
MIN_ROWS = 20
MAX_ROWS = 200
ALPHA = 0.05                     # 仅当 p <= ALPHA 才提示人工复核
# 与训练数据同一生成族（app.model.make_synthetic_data 的 Dirichlet 参数）
DIRICHLET_ALPHA = (2.5, 2.0, 1.6, 1.2)

_INT_RE = re.compile(r"^-?\d+$")
_MAX_ERROR_LINES = 20


# ---------------------------------------------------------------------------
# 参考配方生成
# ---------------------------------------------------------------------------

def _integer_compose(props: np.ndarray, total: int = TOTAL_GRAMS) -> np.ndarray:
    """把每行和为 1 的比例矩阵确定性地凑成总和为 total 的整数克数矩阵。

    采用最大余数法：先向下取整，再把每行缺的克数依次补到小数部分最大的
    原料上。argsort 使用稳定排序，并列时按原料下标决定，结果完全确定。
    """
    raw = props * total
    floors = np.floor(raw).astype(np.int64)
    remainder = raw - floors
    deficit = total - floors.sum(axis=1, keepdims=True)
    # order[i] 给出第 i 行按余数从大到小排列的原料下标；rank 为其逆置换
    order = np.argsort(-remainder, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(order.shape[1])[None, :]
    add = (rank < deficit).astype(np.int64)
    return floors + add


def generate_reference_recipes(n: int = N_REFERENCE, seed: int = REFERENCE_SEED) -> np.ndarray:
    """生成 n 条总重 1000 克的四料整数克参考配方（训练外，固定种子可复现）。"""
    rng = np.random.default_rng(seed)
    props = rng.dirichlet(DIRICHLET_ALPHA, size=n)
    grams = _integer_compose(props)
    return grams.astype(np.int64)


def example_batch_text(n: int = 30, seed: int = EXAMPLE_SEED) -> str:
    """生成一段合法的示例批次文本（与参考同分布，供网页手动操作演示）。"""
    rng = np.random.default_rng(seed)
    grams = _integer_compose(rng.dirichlet(DIRICHLET_ALPHA, size=n))
    return "\n".join(" ".join(str(v) for v in row) for row in grams)


# ---------------------------------------------------------------------------
# float32 输入语义的叶节点路由
# ---------------------------------------------------------------------------

def route_leaves(clf, grams: np.ndarray) -> np.ndarray:
    """按模型 float32 输入语义返回每个样本在五棵树上的叶节点 id。

    与训练 / 预测路径完全一致：先把克数除以 1000 转成 float32，再交给
    sklearn 的 apply 走真实分裂比较（float32 特征对 float64 阈值）。
    返回形状 (n_samples, N_TREES) 的 int64 节点 id 矩阵。
    """
    x = np.asarray(grams, dtype=np.float32) / TOTAL_GRAMS
    return clf.apply(x).astype(np.int64)


# ---------------------------------------------------------------------------
# 批次文本解析（出错不出结论）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedBatch:
    rows: list[list[int]] | None
    errors: list[str]


def parse_batch_text(text: object) -> ParsedBatch:
    """解析粘贴文本：20～200 行、每行 4 个 [0,1000] 整数且总重恰为 1000。

    允许空白或逗号分隔；任何一行不合规都收集带行号的错误并整体拒绝，
    绝不带着脏数据进入统计。
    """
    if not isinstance(text, str) or not text.strip():
        return ParsedBatch(None, ["内容为空：请粘贴 20～200 行四料整数克数记录。"])
    errors: list[str] = []
    rows: list[list[int]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line == "":
            errors.append(f"第 {line_no} 行为空行。")
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) != N_FEATURES:
            errors.append(f"第 {line_no} 行应有 {N_FEATURES} 个整数克数，实际 {len(tokens)} 个。")
            continue
        vals: list[int] = []
        line_bad = False
        for tok in tokens:
            if not _INT_RE.match(tok):
                errors.append(f"第 {line_no} 行存在非整数克数：{tok!r}。")
                line_bad = True
                continue
            v = int(tok)
            if not 0 <= v <= TOTAL_GRAMS:
                errors.append(f"第 {line_no} 行克数 {v} 超出允许范围 [0, {TOTAL_GRAMS}]。")
                line_bad = True
            vals.append(v)
        if line_bad or len(vals) != N_FEATURES:
            continue
        line_sum = sum(vals)
        if line_sum != TOTAL_GRAMS:
            errors.append(f"第 {line_no} 行总重为 {line_sum} 克，须等于 {TOTAL_GRAMS} 克。")
            continue
        rows.append(vals)
    if errors:
        shown = errors[:_MAX_ERROR_LINES]
        if len(errors) > _MAX_ERROR_LINES:
            shown.append(f"……另有 {len(errors) - _MAX_ERROR_LINES} 处错误未展开。")
        return ParsedBatch(None, shown)
    if not MIN_ROWS <= len(rows) <= MAX_ROWS:
        return ParsedBatch(
            None,
            [f"有效记录 {len(rows)} 行，不在要求的 {MIN_ROWS}～{MAX_ROWS} 行范围内。"],
        )
    return ParsedBatch(rows, [])


# ---------------------------------------------------------------------------
# 统计量与置换检验
# ---------------------------------------------------------------------------

def _tree_proportions(leaf_col: np.ndarray, group_idx: np.ndarray, node_count: int) -> np.ndarray:
    counts = np.bincount(leaf_col[group_idx], minlength=node_count)
    return counts.astype(np.float64) / len(group_idx)


def max_leaf_diff(
    ref_leaves: np.ndarray,
    batch_leaves: np.ndarray,
    leaf_ids_by_tree: list[list[int]],
    node_counts: list[int],
) -> tuple[float, dict]:
    """统计量 T：五棵树所有叶节点上两组占比差的最大绝对值。

    并列时按“树下标升序、叶节点 id 升序”取第一个，保证结果确定可复现。
    """
    n_ref = ref_leaves.shape[0]
    n_batch = batch_leaves.shape[0]
    stat = 0.0
    worst: dict | None = None
    for t in range(N_TREES):
        ref_p = _tree_proportions(ref_leaves[:, t], np.arange(n_ref), node_counts[t])
        batch_p = _tree_proportions(batch_leaves[:, t], np.arange(n_batch), node_counts[t])
        for leaf_id in leaf_ids_by_tree[t]:
            p_r, p_b = float(ref_p[leaf_id]), float(batch_p[leaf_id])
            d = abs(p_b - p_r)
            if d > stat:  # 严格大于：并列保留先遇到的（树/叶 id 均按升序遍历）
                stat = d
                worst = {
                    "tree_index": t,
                    "tree": t + 1,
                    "leaf_id": int(leaf_id),
                    "batch_proportion": p_b,
                    "reference_proportion": p_r,
                    "abs_diff": d,
                }
    assert worst is not None  # 五棵树均有叶子，不可能为 None
    return stat, worst


def permutation_p_value(
    pooled_leaves: np.ndarray,
    n_batch: int,
    leaf_ids_by_tree: list[list[int]],
    node_counts: list[int],
    observed: float,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, int]:
    """保持组大小的联合置换检验。

    把两组合并后每次随机抽 n_batch 个样本作“批次组”、其余作“参考组”，
    重算同一统计量。p 值采用含观测值的无偏估计
    (1 + #{T_perm >= T_obs}) / (1 + n_permutations)，最小可取 1/1000。
    """
    n_total = pooled_leaves.shape[0]
    n_ref = n_total - n_batch
    rng = np.random.default_rng(seed)
    n_exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        batch_idx, ref_idx = perm[:n_batch], perm[n_batch:]
        t_stat = 0.0
        for t in range(N_TREES):
            batch_p = _tree_proportions(pooled_leaves[:, t], batch_idx, node_counts[t])
            ref_p = _tree_proportions(pooled_leaves[:, t], ref_idx, node_counts[t])
            for leaf_id in leaf_ids_by_tree[t]:
                d = abs(float(batch_p[leaf_id]) - float(ref_p[leaf_id]))
                if d > t_stat:
                    t_stat = d
        if t_stat >= observed:
            n_exceed += 1
    return (1 + n_exceed) / (1 + n_permutations), n_exceed


def run_drift_check(
    clf,
    ref_leaves: np.ndarray,
    leaf_ids_by_tree: list[list[int]],
    node_counts: list[int],
    rows: list[list[int]],
) -> dict:
    """对已通过校验的批次执行完整核查，返回可直接序列化的结论字典。"""
    t0 = time.perf_counter()
    batch_grams = np.asarray(rows, dtype=np.int64)
    batch_leaves = route_leaves(clf, batch_grams)
    stat, worst = max_leaf_diff(ref_leaves, batch_leaves, leaf_ids_by_tree, node_counts)
    pooled = np.concatenate([ref_leaves, batch_leaves], axis=0)
    p_value, n_exceed = permutation_p_value(
        pooled, batch_leaves.shape[0], leaf_ids_by_tree, node_counts, stat
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "n_rows": int(batch_leaves.shape[0]),
        "n_reference": int(ref_leaves.shape[0]),
        "n_permutations": N_PERMUTATIONS,
        "alpha": ALPHA,
        "statistic": float(stat),
        "p_value": float(p_value),
        "permutation_exceed": int(n_exceed),
        "worst_leaf": worst,
        "elapsed_ms": round(elapsed_ms, 1),
    }
