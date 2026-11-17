"""FastAPI 入口：质检模型 + 配方求解 + 新增批次适用性核查服务。

启动时以固定种子训练随机森林（5 棵树、深度 ≤ 3），预生成经求解器实测的
可行 / 无解样例，并生成 1000 条训练外参考配方、按 float32 输入语义预路由到
五棵树叶节点，供批次核查的占比差统计与联合置换检验复用。
求解结果一律用原分类器复算概率后返回。
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .drift import (
    ALPHA,
    N_PERMUTATIONS,
    N_REFERENCE,
    REFERENCE_SEED,
    example_batch_text,
    generate_reference_recipes,
    parse_batch_text,
    route_leaves,
    run_drift_check,
)
from .model import (
    FEATURE_NAMES,
    MAX_DEPTH,
    N_TREES,
    PROBA_THRESHOLD,
    TOTAL_GRAMS,
    TRAIN_SEED,
    Leaf,
    extract_forest,
    train_model,
)
from .samples import build_samples
from .solver import SolveInput, solve

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="再生塑料配方试配筛选", version="1.0.0")

# ---- 启动即训练（固定种子，秒级完成），并构建求解用森林规范 ----
_t0 = time.perf_counter()
CLF = train_model()
FOREST = extract_forest(CLF)
SAMPLES = build_samples(FOREST, CLF)
TRAIN_SECONDS = time.perf_counter() - _t0

# ---- 新增批次适用性核查的固定参考集（启动即生成并按 float32 语义预路由） ----
REF_GRAMS = generate_reference_recipes()
REF_LEAVES = route_leaves(CLF, REF_GRAMS)
# 每棵树的叶节点 id 与总节点数（与 extract_forest 导出的同一森林结构）
LEAF_IDS_BY_TREE = [
    [nid for nid, node in enumerate(tree.nodes) if isinstance(node, Leaf)]
    for tree in FOREST.trees
]
NODE_COUNTS = [len(tree.nodes) for tree in FOREST.trees]
# 参考配方校验：条数、总重、每行四料整数
assert REF_GRAMS.shape == (N_REFERENCE, 4)
assert int((REF_GRAMS.sum(axis=1) == TOTAL_GRAMS).all()) == 1
assert int(((REF_GRAMS >= 0) & (REF_GRAMS <= TOTAL_GRAMS)).all()) == 1
# 参考集不得与训练数据重合：训练样本是连续 Dirichlet 比例，参考集为整数克数配方
assert REF_LEAVES.shape == (N_REFERENCE, N_TREES)


def _depth(tree, nid: int) -> int:
    node = tree.nodes[nid]
    if isinstance(node, Leaf):
        return 0
    return 1 + max(_depth(tree, node.left), _depth(tree, node.right))


@app.get("/api/meta")
def meta() -> dict:
    return {
        "feature_names": FEATURE_NAMES,
        "total_grams": TOTAL_GRAMS,
        "proba_threshold": PROBA_THRESHOLD,
        "n_trees": N_TREES,
        "max_depth": MAX_DEPTH,
        "tree_depths": [_depth(t, 0) for t in FOREST.trees],
        "train_seed": TRAIN_SEED,
        "train_seconds": round(TRAIN_SECONDS, 3),
        # 新增批次核查参数
        "reference_seed": REFERENCE_SEED,
        "n_reference": N_REFERENCE,
        "n_permutations": N_PERMUTATIONS,
        "drift_alpha": ALPHA,
        "drift_min_rows": 20,
        "drift_max_rows": 200,
    }


@app.post("/api/solve")
def api_solve(payload: dict) -> dict:
    def _int_list(key):
        v = payload.get(key)
        return list(v) if isinstance(v, (list, tuple)) else []

    def _bool_list(key):
        v = payload.get(key)
        return [bool(x) for x in v] if isinstance(v, (list, tuple)) else []

    def _as_ints(key):
        out = []
        for x in _int_list(key):
            if isinstance(x, bool):
                out.append(x)  # 校验阶段会拒绝
            elif isinstance(x, int):
                out.append(x)
            elif isinstance(x, float) and x.is_integer():
                out.append(int(x))
            else:
                out.append(x)  # 原样传入，由 validate 报错
        return out

    inp = SolveInput(
        original=_as_ints("original"),
        lower=_as_ints("lower"),
        upper=_as_ints("upper"),
        locked=_bool_list("locked"),
    )
    result = solve(FOREST, CLF, inp)
    return result.to_dict()


@app.get("/api/samples")
def api_samples() -> dict:
    return SAMPLES


@app.get("/api/drift/example")
def api_drift_example() -> dict:
    """一段合法的示例批次文本（与参考同分布），供网页手动操作演示。"""
    return {"text": example_batch_text()}


@app.post("/api/drift")
def api_drift(payload: dict) -> dict:
    """新增批次适用性核查。

    输入解析失败时返回 status=invalid 并附逐行错误，**不产生任何统计结论**；
    校验通过才执行五棵树叶节点占比差统计量与 999 次联合置换检验。
    p <= 0.05 时 status=review（提示人工复核），否则 status=no_shift
    （仅“未检出偏移”，不做放行判断）。
    """
    text = payload.get("text") if isinstance(payload, dict) else None
    parsed = parse_batch_text(text)
    if parsed.rows is None:
        return {"status": "invalid", "errors": parsed.errors}
    result = run_drift_check(
        CLF, REF_LEAVES, LEAF_IDS_BY_TREE, NODE_COUNTS, parsed.rows
    )
    result["status"] = "review" if result["p_value"] <= ALPHA else "no_shift"
    return result


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
