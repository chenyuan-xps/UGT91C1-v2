"""结果浏览页：查看 results/ 下已有预测结果的分布与明细。"""

import streamlit as st

from prediction_core import RESULT_SOURCES, load_result_frame, score_column_name, source_summary
from ui_helpers import metric_row, position_bar, score_bar, score_histogram, support_scatter

labels = {key: value["label"] for key, value in RESULT_SOURCES.items()}
selected = st.segmented_control(
    "结果来源",
    list(RESULT_SOURCES),
    format_func=lambda key: labels[key],
    default="zeroshot",
    key="result_source",
)
source = selected or "zeroshot"

try:
    frame = load_result_frame(source)
    summary = source_summary(source)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"来源文件 `{summary['file']}` · {summary['note']}")

metric_row(
    [
        ("候选总数", str(summary["total"])),
        ("最高分", f"{summary['top_score']:.3f}"),
        ("中位分", f"{summary['median_score']:.3f}"),
        ("平均分", f"{summary['mean_score']:.3f}"),
    ]
)
st.caption(f"排序依据 `{summary['score_col']}`（降序）；该文件不含真实标签，仅反映候选优先级。")

score_col = score_column_name(frame)
support_col = next(
    (col for col in ("family_support", "family_support_sum") if col in frame.columns), None
)

left, right = st.columns(2)
with left:
    st.markdown("**Top 16 候选**")
    chart = score_bar(frame, "variant", score_col, limit=16)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
with right:
    st.markdown("**分数分布**")
    chart = score_histogram(frame[score_col])
    if chart is not None:
        st.altair_chart(chart, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("**突变位点热区**")
    chart = position_bar(frame["variant"].head(300))
    if chart is not None:
        st.altair_chart(chart, width="stretch")
with right:
    st.markdown("**分数与家族支持度**")
    chart = support_scatter(frame, score_col, support_col) if support_col else None
    if chart is None:
        st.info("该结果文件不含 family_support 列。")
    else:
        st.altair_chart(chart, width="stretch")

st.subheader(f"候选明细（共 {summary['total']} 条，显示前 200 条）")
preferred = [
    "rank",
    "variant",
    "mutations",
    "mutation_list",
    score_col,
    "n_mut",
    "positions",
    "source",
    "family_support",
    "homolog_pool_size",
    "zs_sum",
    "zs_min",
    "zs_mean",
]
display_columns = list(dict.fromkeys(col for col in preferred if col in frame.columns))
st.dataframe(
    frame.head(200)[display_columns],
    hide_index=True,
    width="stretch",
    height=460,
    column_config={
        "rank": st.column_config.NumberColumn("名次", width="small"),
        "variant": st.column_config.TextColumn("突变体", width="medium"),
        "mutations": st.column_config.TextColumn("突变列表", width="medium"),
        "mutation_list": st.column_config.TextColumn("突变列表", width="medium"),
        score_col: st.column_config.NumberColumn("排序分数", format="%.3f"),
        "n_mut": st.column_config.NumberColumn("突变数", width="small"),
        "positions": st.column_config.TextColumn("位点", width="small"),
        "source": st.column_config.TextColumn("来源", width="small"),
        "family_support": st.column_config.NumberColumn("家族支持度", format="%.2f"),
        "homolog_pool_size": st.column_config.NumberColumn("同源池", width="small"),
        "zs_sum": st.column_config.NumberColumn("零样本总分", format="%.3f"),
        "zs_min": st.column_config.NumberColumn("零样本最低", format="%.3f"),
        "zs_mean": st.column_config.NumberColumn("零样本均值", format="%.3f"),
    },
)
st.caption("已隐藏 462 位的 sequence 列，避免表格被长文本撑开；完整字段请下载 CSV。")
st.download_button(
    "下载完整结果 CSV",
    frame.to_csv(index=False).encode("utf-8-sig"),
    f"{source}_ranking.csv",
    "text/csv",
    icon=":material/download:",
    key="download_source",
)
