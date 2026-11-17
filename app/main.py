"""FastAPI 入口：质检模型 + 配方求解服务。

启动时以固定种子训练随机森林（5 棵树、深度 ≤ 3），并预生成经求解器
实测的可行 / 无解样例。求解结果一律用原分类器复算概率后返回。
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
