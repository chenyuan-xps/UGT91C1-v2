"""平台的展示层辅助：配色、指标卡行与图表构建。

配色沿用项目一贯的「黑—蓝—金」科技风，不使用红色系。
"""

from __future__ import annotations

import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PALETTE = {
    "ink": "#0f1720",
    "blue": "#1f4e9c",
    "blue_light": "#6b8cbf",
    "blue_pale": "#dbe4ef",
    "gold": "#c8a657",
}


def metric_row(items, *, columns: int | None = None) -> None:
    """一行指标卡。

    items 为 (label, value, help?) 三元组序列；label 放中文说明，value 放数值。
    """
    cols = st.columns(columns or len(items))
    for col, item in zip(cols, items):
        help_text = item[2] if len(item) > 2 else None
        col.metric(item[0], item[1], help=help_text, border=True)


def score_histogram(values, *, bins: int = 12, x_title: str = "分数区间", y_title: str = "候选数量"):
    """分数分布直方图（离散分箱，纵轴为候选数量）。"""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None

    counts, edges = np.histogram(series.to_numpy(dtype=float), bins=bins)
    frame = pd.DataFrame(
        {
            "分数区间": [f"{edges[i]:.1f} ~ {edges[i + 1]:.1f}" for i in range(len(counts))],
            "候选数量": counts,
        }
    )
    return (
        alt.Chart(frame)
        .mark_bar(color=PALETTE["blue"], size=28)
        .encode(
            x=alt.X("分数区间:N", title=x_title, sort=None),
            y=alt.Y("候选数量:Q", title=y_title),
            tooltip=["分数区间", "候选数量"],
        )
    )


def position_bar(variants, *, top: int = 16, title: str = "突变位点"):
    """突变位点出现频次（识别热区）。"""
    counts: dict[str, int] = {}
    for variant in variants:
        for pos in re.findall(r"[A-Z](\d+)[A-Z]", str(variant)):
            counts[pos] = counts.get(pos, 0) + 1
    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    frame = pd.DataFrame(
        {"位点": [pos for pos, _ in ranked], "出现次数": [count for _, count in ranked]}
    )
    return (
        alt.Chart(frame)
        .mark_bar(color=PALETTE["blue_light"], size=28)
        .encode(
            x=alt.X("位点:N", title=title, sort="-y"),
            y=alt.Y("出现次数:Q", title="出现次数"),
            tooltip=["位点", "出现次数"],
        )
    )


def support_scatter(frame: pd.DataFrame, score_col: str, support_col: str, *, limit: int = 200):
    """分数与家族支持度的关系。"""
    if support_col not in frame.columns:
        return None
    data = frame[["variant", score_col, support_col]].copy()
    data[support_col] = pd.to_numeric(data[support_col], errors="coerce")
    data = data[data[support_col].notna()].head(limit)
    if data.empty:
        return None
    return (
        alt.Chart(data)
        .mark_circle(size=70, color=PALETTE["gold"], opacity=0.75)
        .encode(
            x=alt.X(f"{score_col}:Q", title="排序分数"),
            y=alt.Y(f"{support_col}:Q", title="家族支持度"),
            tooltip=["variant", score_col, support_col],
        )
    )


def score_bar(frame: pd.DataFrame, variant_col: str, score_col: str, *, limit: int = 16):
    """Top N 候选的分数条形图。"""
    data = frame.head(limit)[[variant_col, score_col]].copy()
    if data.empty:
        return None
    return (
        alt.Chart(data)
        .mark_bar(color=PALETTE["blue"], size=24)
        .encode(
            x=alt.X(f"{variant_col}:N", title="突变体", sort="-y"),
            y=alt.Y(f"{score_col}:Q", title="零样本分数"),
            tooltip=[variant_col, score_col],
        )
    )


# --------------------------------------------------------------------------- #
# 候选推荐页专用图表
# --------------------------------------------------------------------------- #
COVERAGE_COLORS = {
    "all_seen": PALETTE["blue"],
    "mixed": PALETTE["blue_light"],
    "none_seen": PALETTE["gold"],
    "background_only": "#8a95a5",
}


def fused_bar(frame: pd.DataFrame, *, value_col: str, label_col: str, limit: int = 15):
    """Top N 候选的融合分数条形图。"""
    data = frame.head(limit)[[label_col, value_col, "coverage"]].copy()
    if data.empty:
        return None
    return (
        alt.Chart(data)
        .mark_bar(size=24)
        .encode(
            x=alt.X(f"{label_col}:N", title="候选突变体", sort="-y"),
            y=alt.Y(f"{value_col}:Q", title="融合名次分（0~1，越高越优先）"),
            color=alt.Color(
                "coverage:N",
                scale=alt.Scale(domain=list(COVERAGE_COLORS), range=list(COVERAGE_COLORS.values())),
                legend=alt.Legend(title="位点覆盖"),
            ),
            tooltip=[label_col, value_col, "coverage"],
        )
    )


def coverage_bar(frame: pd.DataFrame, *, label_col: str = "coverage_label"):
    """位点覆盖分布的条形图。"""
    if label_col not in frame.columns:
        return None
    counts = frame[label_col].value_counts().rename_axis("覆盖类型").reset_index(name="候选数量")
    if counts.empty:
        return None
    return (
        alt.Chart(counts)
        .mark_bar(size=28, color=PALETTE["blue_light"])
        .encode(
            x=alt.X("覆盖类型:N", title="位点覆盖", sort="-y"),
            y=alt.Y("候选数量:Q", title="候选数量"),
            tooltip=["覆盖类型", "候选数量"],
        )
    )


def model_vs_zero_scatter(frame: pd.DataFrame, *, limit: int = 500):
    """模型预测分与零样本分数的关系，按位点覆盖着色。"""
    data = frame[frame["model_pred"].notna()][
        ["variant", "model_pred", "zero_score", "coverage", "n_mut"]
    ].copy()
    if data.empty:
        return None
    data = data.head(limit)
    return (
        alt.Chart(data)
        .mark_circle(size=70, opacity=0.7)
        .encode(
            x=alt.X("model_pred:Q", title="模型预测活性（相对 WT）"),
            y=alt.Y("zero_score:Q", title="零样本分数"),
            color=alt.Color(
                "coverage:N",
                scale=alt.Scale(domain=list(COVERAGE_COLORS), range=list(COVERAGE_COLORS.values())),
                legend=alt.Legend(title="位点覆盖"),
            ),
            tooltip=["variant", "coverage", "n_mut", "model_pred", "zero_score"],
        )
    )


# --------------------------------------------------------------------------- #
# 对比分析页专用图表
# --------------------------------------------------------------------------- #
def metric_compare_row(left_title, left_items, right_title, right_items) -> None:
    """左右并排两组指标卡：左边现役模型，右边候选模型。

    items 为 (中文说明, 数值) 二元组序列；st.metric 会把数值放大在第一排、
    中文说明放在第二排。
    """
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"**{left_title}**")
        metric_row(list(left_items))
    with right_col:
        st.markdown(f"**{right_title}**")
        metric_row(list(right_items))


def pred_vs_actual_scatter(
    frame: pd.DataFrame,
    *,
    pred_col: str = "oof_pred",
    actual_col: str = "activity_rel",
    color_col: str | None = None,
    title_pred: str = "预测 activity_rel",
    title_actual: str = "实测 activity_rel",
    limit: int = 400,
):
    """预测 vs 实测散点图，叠加 y = x 参考线（越贴合参考线说明预测越准）。"""
    if pred_col not in frame.columns or actual_col not in frame.columns:
        return None
    cols = [pred_col, actual_col] + ([color_col] if color_col else [])
    data = frame[cols].copy()
    for col in (pred_col, actual_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[pred_col, actual_col])
    if data.empty:
        return None
    data = data.head(limit)

    lo = float(min(data[pred_col].min(), data[actual_col].min()))
    hi = float(max(data[pred_col].max(), data[actual_col].max()))
    span = [lo, hi] if hi > lo else [lo - 0.5, lo + 0.5]

    encoding = dict(
        x=alt.X(f"{actual_col}:Q", title=title_actual, scale=alt.Scale(domain=span)),
        y=alt.Y(f"{pred_col}:Q", title=title_pred, scale=alt.Scale(domain=span)),
        tooltip=list(dict.fromkeys(cols)),
    )
    if color_col and color_col in data.columns:
        encoding["color"] = alt.Color(f"{color_col}:N", legend=alt.Legend(title=color_col))
    else:
        encoding["color"] = alt.value(PALETTE["blue"])

    points = alt.Chart(data).mark_circle(size=70, opacity=0.75).encode(**encoding)
    diag = (
        alt.Chart(pd.DataFrame({"v": span}))
        .mark_line(strokeDash=[5, 4], color=PALETTE["gold"], size=2)
        .encode(x="v:Q", y="v:Q")
    )
    return alt.layer(points, diag).resolve_scale(color="independent")


def residual_histogram(values, *, bins: int = 12, x_title: str = "残差（预测 − 实测）"):
    """残差分布：以 0 为中心越集中说明系统性偏差越小。"""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    counts, edges = np.histogram(series.to_numpy(dtype=float), bins=bins)
    frame = pd.DataFrame(
        {
            "残差区间": [f"{edges[i]:.2f} ~ {edges[i + 1]:.2f}" for i in range(len(counts))],
            "样本数量": counts,
        }
    )
    return (
        alt.Chart(frame)
        .mark_bar(color=PALETTE["blue_light"], size=28)
        .encode(
            x=alt.X("残差区间:N", title=x_title, sort=None),
            y=alt.Y("样本数量:Q", title="样本数量"),
            tooltip=["残差区间", "样本数量"],
        )
    )


