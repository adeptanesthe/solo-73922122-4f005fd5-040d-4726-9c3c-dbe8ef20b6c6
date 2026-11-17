"""可行 / 无解样例：启动时用真实模型与求解器验证后生成，保证与当前模型一致。"""
from __future__ import annotations

from .model import FEATURE_NAMES, predict_proba_qualified
from .solver import INFEASIBLE, OK, SolveInput, solve

# 可行样例候选：原配方不合格，但不难通过少量改动达标
_FEASIBLE_CANDIDATES = [
    # (原配方, 下限, 上限, 锁定)
    ([250, 250, 350, 150], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [False, True, False, False]),
    ([300, 200, 350, 150], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [False, False, False, False]),
    ([200, 300, 300, 200], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [True, False, False, False]),
    ([350, 150, 350, 150], [100, 0, 0, 0], [1000, 1000, 600, 1000], [False, False, False, False]),
]

# 无解样例候选：限制条件把配方逼进不可能合格的区域（非简单全锁）
_INFEASIBLE_CANDIDATES = [
    # 填料(原料3)被迫 >= 620g，且再生PP(原料1)被压到 <= 150g
    ([150, 130, 620, 100], [0, 0, 620, 0], [150, 1000, 1000, 1000], [False, False, False, False]),
    # 全部锁定在一个不合格配方上
    ([150, 130, 620, 100], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [True, True, True, True]),
    ([200, 150, 550, 100], [0, 0, 550, 0], [200, 1000, 1000, 1000], [False, False, False, False]),
]


def _try_candidates(forest, clf, candidates, want_status):
    for original, lower, upper, locked in candidates:
        inp = SolveInput(original=list(original), lower=list(lower), upper=list(upper), locked=list(locked))
        res = solve(forest, clf, inp)
        if res.status == want_status:
            return {
                "original": list(original),
                "lower": list(lower),
                "upper": list(upper),
                "locked": list(locked),
                "original_probability": predict_proba_qualified(clf, list(original)),
                "check": res.to_dict(),
            }
    return None


def build_samples(forest, clf) -> dict:
    """返回 {"feasible": ..., "infeasible": ...}，均经求解器实测。"""
    feasible = _try_candidates(forest, clf, _FEASIBLE_CANDIDATES, OK)
    infeasible = _try_candidates(forest, clf, _INFEASIBLE_CANDIDATES, INFEASIBLE)
    if feasible is None or infeasible is None:
        raise RuntimeError("样例生成失败：候选样例未通过求解器验证")
    return {"feasible": feasible, "infeasible": infeasible, "feature_names": FEATURE_NAMES}
