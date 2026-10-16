"""对比分析页：把模型给出的预测值和实测值放在一起看，评估模型好坏。

两个板块（口径不同，不要混读）：
  板块一 · 全量按位点留出的折外预测 vs 实测
     衡量「模型外推到没见过的新位点」的泛化能力，是模型好坏的主口径。
     数据来自 models/training_report.csv（重训或补算时产出的 OOF 预测）。
  板块二 · 本轮新增数据：入库前预测 vs 实测
     衡量「这一轮推荐的候选实际做得怎么样」，即主动学习的命中率。
     数据来自 results/active_learning_log.csv（入库时用当时线上模型打的快照）。
"""

import numpy as np
import pandas as pd
import streamlit as st

from prediction_core import (
    RANK_ONLY_NOTE,
    deploy_paths,
    load_active_learning_log,
    load_training_report,
    report_metric_row,
)
from ui_helpers import metric_row, pred_vs_actual_scatter, residual_histogram

ACTIVE_THRESHOLD = 1.0  # activity_rel > 1 视为高于 WT


def spearman(a, b) -> float | None:
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def topk_hits(y_true, y_pred, k):
    """按预测取前 k 个，其中实测高于 WT 的个数。"""
    k = min(k, len(y_pred))
    order = np.argsort(-np.asarray(y_pred, dtype=float))[:k]
    return int((np.asarray(y_true, dtype=float)[order] > ACTIVE_THRESHOLD).sum()), k


def grouped_table(grouped: dict | None, key_title: str) -> pd.DataFrame:
    """把 report.json 里的分组指标转成展示表。"""
    if not grouped:
        return pd.DataFrame()
    frame = pd.DataFrame(grouped).T.reset_index()
    frame.columns = [key_title, "样本数", "Spearman", "AUC", "Recall@5", "Recall@10"]
    return frame


st.caption(
    "把模型给出的预测值和实测活性放在一起对比。两个板块口径不同："
    "上板块看**外推到新位点的泛化能力**，下板块看**本轮推荐的命中情况**。"
)

report, oof = load_training_report()
metrics_view = report_metric_row(report)

st.markdown(
    f"""
<div style="border-left:4px solid #1f4e9c; background:#eef2f8; padding:14px 18px;
            border-radius:0 8px 8px 0; margin:8px 0 4px; line-height:1.75;">
<strong style="color:#1f4e9c;">读图须知</strong>：散点越贴近金色虚线（y = x）说明预测越准；
点云整体在虚线上方表示模型系统性高估、下方表示系统性低估。
排序能力（Spearman / AUC / Recall@K）才是挑候选的核心，数值精度只是参考。<strong>{RANK_ONLY_NOTE}</strong>。
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# 板块一：全量按位点留出
# --------------------------------------------------------------------------- #
st.subheader("板块一 · 全量按位点留出的折外预测 vs 实测")

if oof.empty or report is None:
    paths = deploy_paths()
    st.info(
        "还没有折外预测报告，本板块暂时为空。\n\n"
        "这份报告记录的是「每次留出一个突变位点、用其余数据训练后对该位点样本做出的预测」，"
        "是模型泛化能力的估计，需要在重训或补算时才会生成。"
    )
    st.markdown(
        f"首次生成（对当前线上模型补算一份，不改变模型本身，报告落在 `{paths['active_report_dir'].name}/`）："
    )
    st.code(
        f"python scripts/retrain_deploy_model.py --eval-only "
        f"--model models/xgb_esm.pkl --report-dir {paths['active_report_dir'].as_posix()}",
        language="powershell",
    )
    st.caption("也可以在「数据入库」页重训一次并采纳新模型，报告会随采纳自动落盘。")
    if not load_active_learning_log().empty:
        st.caption("提示：下方板块二已有入库前的预测记录，可直接查看。")
else:
    eval_rows = oof[oof["oof_pred"].notna()].copy()
    eval_rows["residual"] = eval_rows["oof_pred"] - eval_rows["activity_rel"]

    metric_row(
        [
            ("Macro Spearman", metrics_view["spearman_macro"], "按批次分组后的平均排序相关性"),
            ("Macro AUC", metrics_view["auc_macro"], "把 activity_rel > 1 视为高活性后的区分能力"),
            ("Recall@10", metrics_view["recall10_macro"], "取前 10 个候选能覆盖多少真实高活性突变"),
            ("评估突变体数", str(metrics_view["n_variants"]), f"覆盖 {metrics_view['n_sites']} 个位点"),
        ]
    )
    st.caption(
        f"报告生成于 {metrics_view['generated_at']}，训练集 {metrics_view['dataset_rows']} 行；"
        "折外预测来自「同评测协议重新训练各折」，是外部泛化的估计，不是现役模型对训练样本的拟合值。"
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("**预测 vs 实测（按批次着色）**")
        chart = pred_vs_actual_scatter(
            eval_rows,
            color_col="batch",
            title_pred="预测 activity_rel",
            title_actual="实测 activity_rel",
            limit=400,
        )
        if chart is None:
            st.info("没有可用的折外预测点。")
        else:
            st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("**残差分布**")
        chart = residual_histogram(eval_rows["residual"])
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.caption(
            f"残差均值 {eval_rows['residual'].mean():+.3f}，"
            f"中位 {eval_rows['residual'].median():+.3f}"
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**按批次分层**")
        frame = grouped_table((report or {}).get("by_batch"), "批次")
        if frame.empty:
            st.info("报告里没有按批次的分层指标。")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
    with right:
        st.markdown("**按突变数分层**")
        frame = grouped_table((report or {}).get("by_n_mut"), "突变数")
        if frame.empty:
            st.info("报告里没有按突变数的分层指标。")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        "样本不足 3 条或该组内真值无方差时，分层指标留空（显示为 None），不做无意义的计算。"
    )

    with st.expander(f"折外预测明细（共 {len(eval_rows)} 条）", expanded=False):
        st.dataframe(
            eval_rows[
                ["variant", "batch", "n_mut", "activity_rel", "oof_pred", "residual"]
            ].head(400),
            hide_index=True,
            width="stretch",
            height=340,
            column_config={
                "variant": st.column_config.TextColumn("突变体", width="medium"),
                "batch": st.column_config.TextColumn("批次", width="small"),
                "n_mut": st.column_config.NumberColumn("突变数", width="small"),
                "activity_rel": st.column_config.NumberColumn("实测", format="%.3f"),
                "oof_pred": st.column_config.NumberColumn("折外预测", format="%.3f"),
                "residual": st.column_config.NumberColumn("残差", format="%.3f"),
            },
        )
        st.download_button(
            "下载折外预测明细 CSV",
            eval_rows.to_csv(index=False).encode("utf-8-sig"),
            "training_report_oof.csv",
            "text/csv",
            icon=":material/download:",
            key="download_oof",
        )

# --------------------------------------------------------------------------- #
# 板块二：本轮新增数据
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("板块二 · 本轮新增数据：入库前预测 vs 实测")

log = load_active_learning_log()
if log.empty:
    st.info(
        "还没有入库前的预测记录。到「数据入库」页上传新批次数据并执行入库，"
        "平台会先用当时的线上模型给这批突变体打一份预测快照，再把它与实测值一起记在这里。"
    )
else:
    log = log.copy()
    log["actual_activity_rel"] = pd.to_numeric(log["actual_activity_rel"], errors="coerce")
    log["pred_before"] = pd.to_numeric(log["pred_before"], errors="coerce")

    rounds = sorted(log["round_id"].astype(str).unique())
    options = ["全部轮次"] + list(reversed(rounds))
    choice = st.selectbox("查看范围", options, index=0, key="compare_round")

    view = log if choice == "全部轮次" else log[log["round_id"].astype(str) == choice]
    view = view.reset_index(drop=True)
    valid = view.dropna(subset=["pred_before", "actual_activity_rel"])

    hit_count, hit_k = (0, 0)
    if not valid.empty:
        hit_count, hit_k = topk_hits(
            valid["actual_activity_rel"], valid["pred_before"], min(5, len(valid))
        )

    metric_row(
        [
            ("轮次", "全部" if choice == "全部轮次" else str(choice), "每次入库对应一轮"),
            ("突变体数", str(len(view)), f"含 {int(view['n_mut'].gt(1).sum())} 条组合突变"),
            (
                "Top-5 命中",
                f"{hit_count}/{hit_k}" if hit_k else "-",
                "按预测取前 5 个，其中实测高于 WT 的个数",
            ),
            (
                "Spearman",
                "-" if spearman(valid["actual_activity_rel"], valid["pred_before"]) is None
                else f"{spearman(valid['actual_activity_rel'], valid['pred_before']):.3f}",
                "单轮样本少，仅作趋势参考",
            ),
        ]
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("**入库前预测 vs 实测（按轮次着色）**")
        color = "round_id" if choice == "全部轮次" else None
        chart = pred_vs_actual_scatter(
            valid,
            pred_col="pred_before",
            actual_col="actual_activity_rel",
            color_col=color,
            title_pred="入库前预测 activity_rel",
            title_actual="实测 activity_rel",
            limit=400,
        )
        if chart is None:
            st.info("当前范围内没有同时含预测与实测的记录。")
        else:
            st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("**残差分布**")
        chart = residual_histogram(valid["pred_before"] - valid["actual_activity_rel"])
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        else:
            st.info("数据不足，暂不绘制残差分布。")

    st.markdown("**明细**")
    detail = view.assign(
        实测减预测=view["actual_activity_rel"] - view["pred_before"]
    )[
        [
            "round_id",
            "batch",
            "variant",
            "n_mut",
            "pred_before",
            "zero_score",
            "rank_before",
            "actual_activity_rel",
            "实测减预测",
        ]
    ]
    st.dataframe(
        detail,
        hide_index=True,
        width="stretch",
        height=340,
        column_config={
            "round_id": st.column_config.TextColumn("轮次", width="small"),
            "batch": st.column_config.TextColumn("批次", width="small"),
            "variant": st.column_config.TextColumn("突变体", width="medium"),
            "n_mut": st.column_config.NumberColumn("突变数", width="small"),
            "pred_before": st.column_config.NumberColumn("入库前预测", format="%.3f"),
            "zero_score": st.column_config.NumberColumn("零样本分数", format="%.3f"),
            "rank_before": st.column_config.NumberColumn("原推荐名次", width="small"),
            "actual_activity_rel": st.column_config.NumberColumn("实测", format="%.3f"),
            "实测减预测": st.column_config.NumberColumn("实测 − 预测", format="%.3f"),
        },
    )
    st.caption(
        "如实说明：单轮新增样本通常只有几条，命中率与相关系数噪声很大，只适合看趋势；"
        "组合突变叠加是已验证的短板（多突变候选的把握度低于单点），不要据此对单个候选下强结论。"
    )
    st.download_button(
        "下载本轮对比明细 CSV",
        detail.to_csv(index=False).encode("utf-8-sig"),
        "active_learning_compare.csv",
        "text/csv",
        icon=":material/download:",
        key="download_active_log",
    )
