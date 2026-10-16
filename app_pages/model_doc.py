"""模型与数据说明页：数据来源、评测协议、方案对比与后续优先级。"""

import streamlit as st

from prediction_core import RESULT_SOURCES, model_evaluation, tier_cv_metrics
from ui_helpers import metric_row

MODEL_LABELS = {
    "ZS_zeroshot_only": "零样本分数（当前排序器）",
    "M0_meanpool_prior_XGB": "线上配方（mean-pool+先验）",
    "M1_zeroshot_prior_XGB": "零样本+先验+XGBoost",
    "M2_sitelocal_zs_prior_PCA_Ridge": "位点窗口+零样本+PCA",
    "M3_sitelocal_zs_prior_XGB": "位点窗口+零样本+XGBoost",
    "M4_fusion_PCA_Ridge": "融合特征+PCA-Ridge",
}

evaluation = model_evaluation()

st.subheader("数据")
metric_row(
    [
        ("训练数据", f"{evaluation['n_train']} 条"),
        ("实验批次", "2 个", "批次 A：WT = 0.021；批次 B：WT = 3.7"),
        ("批间差异", "1.76 倍", "按各自批内 WT 归一化后再合并，避免引入批间偏差"),
        ("背景位点", "F208M", "出现在几乎所有批次 B 变体中，评测与排序时剔除"),
    ]
)

st.subheader("评测协议")
st.markdown(
    """
采用**按位点留出交叉验证**（leave-one-site-out）：每次把某个突变位点的全部变体留作测试集，
用其余位点训练并预测，从而外推到从未见过的位点。

- 这是「挑候选突变体」的真实场景，因此作为主指标；随机折的结果会虚高泛化能力，不作参考。
- 主指标为排序能力（Spearman / AUC / Recall@10）；R² / RMSE 受标签尺度影响，参考价值有限。
- 评测集为剔除背景位点 208 后的 27 个突变位点 / 64 个变体。
"""
)

st.subheader("方案对比")
metrics = tier_cv_metrics()
if metrics.empty:
    st.info("未找到 `results/tier_cv_metrics.csv`，跳过方案对比。")
else:
    show = metrics.assign(方案=metrics["model"].map(lambda key: MODEL_LABELS.get(key, key)))
    show = show[["方案", "n_features", "macro_spearman", "macro_auc", "macro_recall10"]]
    show.columns = ["方案", "特征数", "Macro Spearman", "Macro AUC", "Recall@10"]
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "方案": st.column_config.TextColumn(width="medium"),
            "特征数": st.column_config.NumberColumn(width="small"),
            "Macro Spearman": st.column_config.NumberColumn(width="small", format="%.3f"),
            "Macro AUC": st.column_config.NumberColumn(width="small", format="%.3f"),
            "Recall@10": st.column_config.NumberColumn(width="small", format="%.2f"),
        },
    )
    st.caption(
        "结论：参数无关的零样本分数排序能力最强；位点窗口表征是负贡献；"
        "在 67 条数据规模下加训练头反而拖累排序（0.48 → 0.17）。"
    )

st.subheader("结果来源")
st.dataframe(
    [
        {
            "文件名": info["path"].name,
            "说明": info["short"],
            "状态": "已就绪" if info["path"].exists() else "文件缺失",
        }
        for info in RESULT_SOURCES.values()
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "文件名": st.column_config.TextColumn(width="medium"),
        "说明": st.column_config.TextColumn(width="medium"),
        "状态": st.column_config.TextColumn(width="small"),
    },
)

st.subheader("后续优先级")
st.markdown(
    f"""
- **P0 已完成** 用本地同配方重训的模型替换线上工件（旧工件在 XGBoost 3.0.5 下丢失截距，预测整体偏移约 1.4），并修复平台首页的评估展示。
- **P1 待新数据** 主动学习回灌，专门补组合突变样本；每轮实验后按批内 WT 归一化并入训练集。
- **P2 已完成** 零样本分数作为候选筛选的第一道排序器已上线；位点窗口特征方向已放弃。
- **P3 待数据量** 数据积累到 200 条以上后，再考虑轻量微调（LoRA r=4 仅挂最后两层）或适配器方案。

当前泛化能力：Macro Spearman {evaluation['text']['spearman_macro']} ·
Macro AUC {evaluation['text']['auc_macro']} ·
Recall@10 {evaluation['text']['recall10_macro']}。
{evaluation['rank_only_note']}。
"""
)
