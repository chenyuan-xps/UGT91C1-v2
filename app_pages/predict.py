"""候选预测页：上传 CSV → 零样本排序 + 模型预测 → 导出结果。"""

from datetime import datetime

import streamlit as st

from prediction_core import prediction_payload, read_upload_bytes, sample_template
from ui_helpers import metric_row, score_bar, score_histogram

st.caption("上传候选突变体 CSV，平台按「零样本分数 → 模型预测」的顺序输出候选名次。")
st.markdown(
    """
- 必需列：`variant`（如 `F208M`、`F208M;V129A`），多个突变用 `;` 或 `-` 分隔。
- 可选列：`sequence`（留空会按 WT 序列自动补齐）、`activity_rel` / `activity`（≥5 条时额外给出数值精度）。
"""
)

uploaded = st.file_uploader("上传候选 CSV", type=["csv"], help="UTF-8 或 GBK 编码的 CSV")

left, right = st.columns([3, 1], vertical_alignment="bottom")
with left:
    run = st.button("开始预测", type="primary", icon=":material/play_arrow:")
with right:
    st.download_button(
        "下载模板",
        sample_template(),
        "ugt_prediction_template.csv",
        "text/csv",
        icon=":material/download:",
        key="download_template",
    )

if run:
    if uploaded is None:
        st.error("请先选择一个 CSV 文件。")
    else:
        try:
            raw_df = read_upload_bytes(uploaded.getvalue())
        except Exception as exc:
            st.error(f"读取 CSV 失败：{exc}")
        else:
            with st.status("正在预测…", expanded=True) as status:

                def progress(step, message):
                    status.update(label=f"[{step}/6] {message}")
                    st.write(f"{step}／6　{message}")

                try:
                    st.session_state.predict_result = prediction_payload(raw_df, progress=progress)
                    status.update(label="预测完成", state="complete", expanded=False)
                except Exception as exc:
                    status.update(label="预测失败", state="error")
                    st.error(str(exc))

payload = st.session_state.get("predict_result")
if not payload:
    st.info("还没有预测结果：上传 CSV 后点击「开始预测」。")
    st.stop()

frame = payload["frame"]
score_col = payload["score_col"]

st.divider()
st.subheader(f"候选排序（{payload['counts']['input']} 条候选）")
metric_row(
    [
        ("候选数量", str(payload["counts"]["input"])),
        ("最高零样本分数", f"{frame[score_col].max():.3f}"),
        ("中位零样本分数", f"{frame[score_col].median():.3f}"),
        ("特征数量", str(payload["counts"]["features"])),
    ]
)

left_chart, right_chart = st.columns(2)
with left_chart:
    st.markdown("**Top 16 候选**")
    chart = score_bar(frame, "variant", score_col, limit=16)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
with right_chart:
    st.markdown("**零样本分数分布**")
    chart = score_histogram(frame[score_col])
    if chart is not None:
        st.altair_chart(chart, width="stretch")

st.subheader("本次上传数据的数值精度（参考）")
evaluation = payload["evaluation"]
if evaluation.get("available"):
    metric_row(
        [
            ("五折 RMSE", f"{evaluation['rmse_mean']:.3f}"),
            ("五折 MAE", f"{evaluation['mae_mean']:.3f}"),
            ("R²", "—" if evaluation["r2"] is None else f"{evaluation['r2']:.2f}"),
            ("参与样本", str(evaluation["n_samples"])),
        ]
    )
    st.caption("随机五折只衡量数值拟合，不代表候选排序能力；挑候选请以名次为准。")
else:
    st.info(evaluation.get("message", "未提供标签，跳过数值精度评估。"))

st.subheader("候选明细")
display_columns = [
    col
    for col in (
        "rank",
        "variant",
        "mutation_list",
        score_col,
        "rank_model",
        "pred_activity",
        "activity_rel",
        "activity",
        "cv_error",
    )
    if col in frame.columns
]
st.dataframe(
    frame[display_columns],
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "rank": st.column_config.NumberColumn("零样本名次", width="small"),
        "variant": st.column_config.TextColumn("突变体", width="medium"),
        "mutation_list": st.column_config.TextColumn("突变列表", width="medium"),
        score_col: st.column_config.NumberColumn("零样本分数", format="%.3f"),
        "rank_model": st.column_config.NumberColumn("模型名次", width="small"),
        "pred_activity": st.column_config.NumberColumn("预测活性", format="%.3f"),
        "activity_rel": st.column_config.NumberColumn("真实相对活性", format="%.3f"),
        "activity": st.column_config.NumberColumn("真实活性", format="%.3f"),
        "cv_error": st.column_config.NumberColumn("预测误差", format="%.3f"),
    },
)
st.caption(
    "预测活性为 activity_rel 口径（WT = 1）。"
    + (
        f"本次上传带标签，该列由上传数据的五折外推得到（{evaluation['n_samples']} 条），"
        "样本较少时波动大，仅作参考。"
        if evaluation.get("available")
        else "本次上传未带标签，该列来自平台预训练模型。"
    )
)

st.download_button(
    "下载结果 CSV",
    frame.to_csv(index=False).encode("utf-8-sig"),
    f"ugt91c1_predictions_{datetime.now():%Y%m%d_%H%M}.csv",
    "text/csv",
    type="primary",
    icon=":material/download:",
    key="download_predictions",
)
