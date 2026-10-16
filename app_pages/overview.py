"""概览页：模型配置、泛化能力评测与使用须知。"""

import streamlit as st

from prediction_core import get_resources, model_evaluation, model_params_text
from ui_helpers import metric_row

st.caption(
    "面向 UGT91C1 糖基转移酶的突变体活性预测与候选排序工具："
    "用现有实验数据，对候选库里的突变体统一打分，直接给出下一批最值得实验的序列。"
)

evaluation = model_evaluation()
resources = get_resources()

st.subheader("当前模型")
metric_row(
    [
        ("预测目标", evaluation["target"], evaluation["target_note"] or None),
        (
            "特征数量",
            str(len(resources["feature_names"])),
            "ESM-2 序列表征 1280 维 + 进化 / 结构 / 稳定性先验 23 维",
        ),
        ("训练数据", f"{evaluation['n_train']} 条", "两个实验批次按各自批内 WT 归一化后合并"),
        ("回归器", model_params_text()),
    ]
)

st.subheader("泛化能力（按位点留出）")
metric_row(
    [
        (
            "Macro Spearman",
            evaluation["text"]["spearman_macro"],
            "留出整个突变位点后外推，最贴近「挑候选突变体」的真实场景",
        ),
        ("Macro AUC", evaluation["text"]["auc_macro"], "区分「高于 WT」与「低于 WT」的能力"),
        ("Recall@10", evaluation["text"]["recall10_macro"], "前 10 名候选覆盖真实高活性突变的比例"),
        (
            "RMSE",
            evaluation["text"]["rmse"],
            f"{evaluation['rmse_protocol']}，只反映数值精度，挑候选时参考价值有限",
        ),
    ]
)
st.caption(
    f"评测协议 {evaluation['protocol']} · "
    f"{evaluation['n_variants']} 个变体 / {evaluation['n_sites']} 个突变位点"
)

st.markdown(
    f"""
<div style="border-left:4px solid #1f4e9c; background:#eef2f8; padding:14px 18px;
            border-radius:0 8px 8px 0; margin:8px 0 4px; line-height:1.75;">
<strong style="color:#1f4e9c;">使用须知</strong>：{evaluation['rank_only_note']}。
平台的作用是把实验范围从几十个候选收敛到前 N 名，而不是给出可信的绝对活性。
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("预测流程")
st.markdown(
    """
1. **读取候选库** 合并单点突变池与 F208M 打底的组合池（共两千余条），按突变体去重。
2. **位点分流** 候选的突变位点若曾在训练数据中出现，交给模型；未出现则交给零样本分数。
3. **模型预测** 已见位点候选：ESM-2 序列表征 1280 维 + 进化 / 结构 / 稳定性先验 23 维 → XGBoost。
4. **零样本打分** 未见位点候选：对 WT 逐位点 mask，以 Σ[log p(突变) − log p(WT)] 打分。
5. **名次融合** 两分数尺度不同，按名次百分位加权合并成「融合名次分」，越接近 1 越优先。
6. **导出序列** 在「候选推荐」页选 Top-N 并下载下一批建议实验的突变体 CSV。
"""
)

st.subheader("页面导航")
st.markdown(
    """
- **候选推荐**：不用上传，直接给出下一批建议实验的序列与导出表。
- **候选预测**：自己准备一批候选 CSV 上传，得到零样本名次 + 模型名次。
- **数据入库**：把新一批实验数据并入训练集并自动重训，人工确认后采纳新模型。
- **对比分析**：把预测值与实测值放在一起看，评估模型泛化能力与本轮推荐命中率。
- **结果浏览**：查看 results/ 下已有的各套预测结果分布与明细。
- **模型与数据**：模型配方、评测协议与使用须知。
"""
)

st.subheader("已知短板")
st.markdown(
    """
- **组合突变排序偏弱**：批次 A 单点突变零样本 Spearman 0.72，批次 B 组合突变仅 0.23，这是目前唯一被验证的短板。
- **位点窗口表征已停用**：在 67 条数据规模下为负贡献，不再投入。
- **暂不做微调**：数据量积累到 200 条以上再考虑 LoRA / 适配器方案。
"""
)
