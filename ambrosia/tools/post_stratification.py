from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as sps


def _target_weights(
    df_list: Iterable[pd.DataFrame],
    strat_columns: Iterable[Any],
    target: Optional[Union[Dict[Tuple, float], str]],
) -> Dict[Tuple, float]:
    """
    Build target weights for strata. If target is:
      - None or "pooled": use pooled sample shares
      - "uniform": equal weights across observed strata
      - dict: explicit mapping from stratum key -> weight
    """
    if target is None or target == "pooled":
        pooled = pd.concat(df_list, axis=0)
        w = (pooled.groupby(list(strat_columns)).size() / len(pooled)).astype(float)
        return w.to_dict()  # type: ignore[return-value]
    if target == "uniform":
        pooled = pd.concat(df_list, axis=0)
        keys = pooled.groupby(list(strat_columns)).size().index
        v = 1.0 / len(keys)
        return {k: v for k in keys}  # type: ignore[dict-item]
    if isinstance(target, dict):
        return target
    raise ValueError("Unsupported post_strat_target specification")


def _strata_stats(
    df: pd.DataFrame, column: Any, strat_columns: Iterable[Any]
) -> Dict[Tuple, Tuple[float, float, int]]:
    """
    Per-stratum mean, unbiased variance (ddof=1) and counts.
    """
    g = df.groupby(list(strat_columns))[column]
    means = g.mean()
    vars_ = g.var(ddof=1).fillna(0.0)
    ns = g.size()
    out: Dict[Tuple, Tuple[float, float, int]] = {}
    for k in means.index:
        out[k] = (float(means.loc[k]), float(vars_.loc[k]), int(ns.loc[k]))
    return out


def post_strat_ttest(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    column: Any,
    strat_columns: Iterable[Any],
    alpha: Union[float, np.ndarray] = 0.05,
    alternative: str = "two-sided",
    post_strat_target: Optional[Union[Dict[Tuple, float], str]] = None,
) -> Tuple[float, Iterable[Tuple[float, float]], float]:
    """
    Post-stratified Welch t-test for difference in means.

    Returns (effect, confidence_intervals, pvalue).
    """
    W = _target_weights([df_a, df_b], strat_columns, post_strat_target)
    stats_a = _strata_stats(df_a, column, strat_columns)
    stats_b = _strata_stats(df_b, column, strat_columns)

    # Validate strata presence
    for k, w in W.items():
        if w > 0 and (k not in stats_a or k not in stats_b):
            raise ValueError("Stratum missing in one of groups for positive target weight")

    diff = 0.0
    var_sum = 0.0
    df_den = 0.0
    for k, w in W.items():
        if w == 0 or k not in stats_a or k not in stats_b:
            continue
        ma, va, na = stats_a[k]
        mb, vb, nb = stats_b[k]
        diff += w * (mb - ma)
        term = (w * w) * (vb / max(nb, 1) + va / max(na, 1))
        var_sum += term
        if nb > 1:
            df_den += (w**4) * ((vb / nb) ** 2) / (nb - 1)
        if na > 1:
            df_den += (w**4) * ((va / na) ** 2) / (na - 1)

    se = float(np.sqrt(max(var_sum, 0.0)))
    if se == 0.0:
        t_stat = np.inf if diff != 0 else 0.0
        pvalue = 0.0 if diff != 0 else 1.0
        alphas = np.array([alpha]) if isinstance(alpha, float) else alpha
        ci = [(diff, diff)] if len(alphas) == 1 else [(diff, diff)] * len(alphas)
        return diff, ci, pvalue

    df_satt = float((var_sum ** 2) / max(df_den, 1e-16))
    t_stat = diff / se
    if alternative == "two-sided":
        pvalue = 2 * sps.t.sf(np.abs(t_stat), df=df_satt)
    elif alternative == "greater":
        pvalue = sps.t.sf(t_stat, df=df_satt)
    elif alternative == "less":
        pvalue = sps.t.cdf(t_stat, df=df_satt)
    else:
        raise ValueError("alternative must be one of: two-sided, greater, less")

    alphas = np.array([alpha]) if isinstance(alpha, float) else alpha
    t_quant = sps.t.ppf(1 - alphas / 2.0, df=df_satt)
    ci = list(zip(diff - t_quant * se, diff + t_quant * se))
    return diff, ci, float(pvalue)
