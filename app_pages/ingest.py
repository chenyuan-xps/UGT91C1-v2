"""数据入库页：把新一批实验数据并入训练集，并用统一训练入口重训模型。

流程（页面自上而下即操作顺序）：
  上传新批次 CSV → 校验与批内 WT 归一化 → 预览确认 → 入库前打预测快照
  → 追加进 data/train_dataset.csv → 重建 ESM 特征缓存 → 重训候选模型
  → 并排对照现役/候选指标 → 人工决定是否采纳

安全约定：
  * 追加训练集前会先写带时间戳的备份；
  * 重训只产出候选模型（models/candidates/run_*），绝不直接覆盖线上的 xgb_esm.pkl；
  * 只有人工点「采纳新模型」才会替换线上模型，旧模型归档进 models/history/；
  * 丢弃候选不会删除已入库的实验数据。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_batch as ib  # noqa: E402

from prediction_core import (  # noqa: E402
    RANK_ONLY_NOTE,
    adopt_candidate,
    deploy_paths,
    discard_candidate,
    latest_candidate_dir,
    load_candidate_report,
    load_training_history,
    model_evaluation,
    new_candidate_dir,
    report_metric_row,
    run_stage_subprocess,
    score_with_current_model,
)

st.session_state.setdefault("ingest_candidate_dir", None)

paths = deploy_paths()

st.caption(
    "拿到新一批实验数据后，在本页把数据并入统一训练集，并用统一入口重训模型。"
    "训练数据与模型采纳是两个独立决定：数据入库后即使不采纳新模型，实验测量值也会保留在训练集里。"
)

st.markdown(
    """
<div style="border-left:4px solid #1f4e9c; background:#eef2f8; padding:14px 18px;
            border-radius:0 8px 8px 0; margin:8px 0 4px; line-height:1.75;">
<strong style="color:#1f4e9c;">为什么必须按批内 WT 归一化</strong>：不同批次的比酶活存在批间漂移，
直接混用绝对值会让模型学到「批次」而不是「突变」。本页会用该批次的 WT 活性做除数，
算出 activity_rel（相对 WT 倍数，WT = 1）后再入库。
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# 1. 格式说明与模板
# --------------------------------------------------------------------------- #
with st.expander("新批次 CSV 格式（点开查看 / 下载模板）", expanded=False):
    st.markdown(
        """
最小格式只有两列：

| 列名 | 必填 | 说明 |
| --- | --- | --- |
| `variant` | 是 | 突变体写法，`F208M` 或 `F208M;V129A`（分隔符 `-` 与 `;` 均可）；WT 行写 `WT` |
| `activity` | 是 | 该批次的**原始**比酶活实测值；WT 必须为正数 |

平台会自动补齐 `mutation_list / sequence / sites / sites_str / n_mut / is_background_only`，
并按 `activity_rel = activity / 批内 WT` 归一化。突变写法会与 WT 序列逐位点核对，
WT 残基对不上会直接报错（挡住手误）。
"""
    )
    template = pd.DataFrame(
        {"variant": ["WT", "F208M-V129W", "F208M-H93C"], "activity": [3.5, 7.0, 9.0]}
    )
    st.download_button(
        "下载新批次模板 CSV",
        template.to_csv(index=False).encode("utf-8-sig"),
        "ugt91c1_new_batch_template.csv",
        "text/csv",
        icon=":material/download:",
        key="download_batch_template",
    )

# --------------------------------------------------------------------------- #
# 2. 上传与预览
# --------------------------------------------------------------------------- #
st.subheader("① 上传新批次数据并预览")

upload = st.file_uploader(
    "新批次 CSV（需含 variant 与 activity 列）", type=["csv"], key="ingest_upload"
)

batch_default = ib.next_batch_label(paths["dataset"])
col_batch, col_wt = st.columns(2)
with col_batch:
    batch_label = st.text_input("批次标签", value=batch_default, key="ingest_batch_label")
with col_wt:
    wt_input = st.number_input(
        "批内 WT 比酶活（CSV 已含 WT 行时留 0 即可）",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="ingest_wt",
    )
wt_activity = None if wt_input == 0.0 else float(wt_input)

rows = pd.DataFrame(columns=ib.COLUMNS)
warnings: list[str] = []
errors: list[str] = []

if upload is None:
    st.info("请上传新批次 CSV 后继续。", icon=":material/upload_file:")
    st.stop()

try:
    raw = pd.read_csv(upload, encoding="utf-8-sig")
except UnicodeDecodeError:
    upload.seek(0)
    raw = pd.read_csv(upload, encoding="gbk")

existing = ib.load_dataset(paths["dataset"])
rows, warnings, errors = ib.parse_new_batch(
    raw, batch_label, wt_activity=wt_activity, existing=existing
)

if errors:
    st.error("解析失败，未做任何写入。请修正后再上传：\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()

for msg in warnings:
    st.warning(msg)

st.caption(
    f"当前训练集 {len(existing)} 行，来自批次 {sorted(existing['batch'].astype(str).unique())}。"
)

if rows.empty:
    st.info("没有可入库的新记录，已在上方说明原因。")
    st.stop()

st.markdown(f"**待入库 {len(rows)} 条（批内 WT = {rows['wt_batch'].iloc[0]:g}）**")
st.dataframe(
    rows[
        [
            "variant",
            "activity",
            "activity_rel",
            "sites_str",
            "n_mut",
            "is_background_only",
        ]
    ],
    hide_index=True,
    width="stretch",
    height=260,
    column_config={
        "variant": st.column_config.TextColumn("突变体", width="medium"),
        "activity": st.column_config.NumberColumn("原始活性", format="%.3f"),
        "activity_rel": st.column_config.NumberColumn("相对 WT 倍数", format="%.3f"),
        "sites_str": st.column_config.TextColumn("突变位点", width="small"),
        "n_mut": st.column_config.NumberColumn("突变数", width="small"),
        "is_background_only": st.column_config.CheckboxColumn("仅背景位点", width="small"),
    },
)
st.caption("`相对 WT 倍数` 才是进训练集的标签；WT 行自身恒为 1，作为本批次的归一化基准。")

# --------------------------------------------------------------------------- #
# 3. 确认并执行
# --------------------------------------------------------------------------- #
st.subheader("② 确认入库并重训")

col_confirm, col_grid = st.columns([1.2, 1])
with col_confirm:
    confirmed = st.checkbox(
        f"我已核对上述 {len(rows)} 条数据无误，同意追加进训练集（会先自动备份原文件）",
        key="ingest_confirm",
    )
with col_grid:
    use_grid = st.toggle(
        "自动调优（小网格搜索）",
        value=False,
        key="ingest_grid",
        help="关闭＝沿用线上配方快速重训；打开＝另外试 4 组超参，按按位点留出的 Macro Spearman 择优，耗时明显更长。",
    )

candidate_dir = st.session_state.get("ingest_candidate_dir")
if candidate_dir:
    candidate_dir = Path(candidate_dir)

run = st.button(
    "入库并重训",
    type="primary",
    disabled=not confirmed,
    icon=":material/model_training:",
    key="ingest_run",
)

if run:
    snapshot_rows = rows[rows["variant"].astype(str).str.upper().ne("WT")].reset_index(drop=True)
    target_dir = new_candidate_dir()

    # ---- ① 入库前快照：必须早于任何模型改动，记录「当时模型怎么预测这批」----
    with st.status("① 用当前线上模型给新批次打「入库前预测」快照…", expanded=True) as status:
        try:
            if snapshot_rows.empty:
                st.write("本批次只有 WT 行，无需快照。")
            else:
                scored = score_with_current_model(snapshot_rows[["variant", "sequence"]])
                log_rows = snapshot_rows[["variant", "activity_rel", "n_mut"]].merge(
                    scored[["variant", "pred_before", "zero_score", "rank_before"]],
                    on="variant",
                    how="left",
                )
                round_id = None
                from prediction_core import append_active_learning_log  # 延迟导入，避免页面顶部堆叠

                round_id = append_active_learning_log(log_rows, batch_label=batch_label)
                st.write(f"已写入预测快照，轮次 {round_id}（共 {len(log_rows)} 条突变体）")
            status.update(label="① 入库前预测快照已记录", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="① 快照失败，流程已中断（未改动任何文件）", state="error")
            st.exception(exc)
            st.stop()

    # ---- ② 追加进训练集 ----
    with st.status("② 追加进训练集（先备份原文件）…", expanded=True) as status:
        try:
            info = ib.append_rows(rows, paths["dataset"])
            st.write(f"备份：{Path(info['backup']).name}")
            st.write(f"行数：{info['n_before']} → {info['n_after']}")
            status.update(label=f"② 已入库，训练集 {info['n_after']} 行", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="② 入库失败，流程已中断", state="error")
            st.exception(exc)
            st.stop()

    # ---- ③ 重建 ESM 特征缓存（行序必须与新的 CSV 对齐，因此整体重算）----
    st.write("**③ 重建 ESM 特征缓存**（行序必须与新的训练集逐行对齐，因此整体重算）")
    feature_holder = st.empty()
    with st.status("正在重建特征缓存（需要 GPU，可能较久）…", expanded=True) as status:
        try:
            lines: list[str] = []
            for line in run_stage_subprocess([paths["features_script"]]):
                lines.append(line)
                feature_holder.code("\n".join(lines[-20:]), language="text")
            status.update(label="③ 特征缓存已重建", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="③ 特征重建失败：数据已入库，但模型未更新", state="error")
            st.error(str(exc))
            st.warning("数据仍在训练集里。修好问题后重跑本页即可继续，无需重复入库。")
            st.stop()

    # ---- ④ 重训候选模型（不覆盖线上 pkl）----
    retrain_args = [
        paths["retrain_script"],
        "--out",
        target_dir / "model.pkl",
        "--report-dir",
        target_dir,
    ]
    if use_grid:
        retrain_args.append("--grid")
    st.write(
        f"**④ 重训候选模型**（输出到 `models/candidates/{target_dir.name}/`，不覆盖线上模型）"
    )
    retrain_holder = st.empty()
    with st.status("正在重训并做按位点留出评估…", expanded=True) as status:
        try:
            lines = []
            for line in run_stage_subprocess(retrain_args):
                lines.append(line)
                retrain_holder.code("\n".join(lines[-20:]), language="text")
            status.update(label="④ 候选模型已就绪", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="④ 重训失败：数据已入库，但模型未更新", state="error")
            st.error(str(exc))
            st.warning("数据仍在训练集里，修好问题后重跑本页即可。")
            st.stop()

    st.session_state["ingest_candidate_dir"] = str(target_dir)
    st.rerun()

# --------------------------------------------------------------------------- #
# 4. 现役 / 候选 指标对照
# --------------------------------------------------------------------------- #
candidate_dir = st.session_state.get("ingest_candidate_dir")
if not candidate_dir:
    fallback = latest_candidate_dir()
    if fallback is not None:
        candidate_dir = fallback
if candidate_dir:
    candidate_dir = Path(candidate_dir)

if candidate_dir and (candidate_dir / "model.pkl").exists():
    st.divider()
    st.subheader("③ 现役模型 vs 候选模型")

    live = model_evaluation()
    cand_report, cand_oof = load_candidate_report(candidate_dir)
    cand = report_metric_row(cand_report)

    from ui_helpers import metric_compare_row  # 延迟导入，保持顶部 import 精简

    metric_compare_row(
        f"现役模型（{live.get('n_train')} 条训练样本）",
        [
            ("按位点留出 Macro Spearman", live["text"]["spearman_macro"]),
            ("按位点留出 Macro AUC", live["text"]["auc_macro"]),
            ("Recall@10", live["text"]["recall10_macro"]),
            ("RMSE（仅参考）", live["text"]["rmse"]),
        ],
        f"候选模型（{cand['dataset_rows']} 条训练样本）",
        [
            ("按位点留出 Macro Spearman", cand["spearman_macro"]),
            ("按位点留出 Macro AUC", cand["auc_macro"]),
            ("Recall@10", cand["recall10_macro"]),
            ("RMSE（仅参考）", cand["rmse"]),
        ],
    )

    if cand["available"]:
        st.caption(
            f"候选报告生成于 {cand['generated_at']}；留出评估覆盖 {cand['n_sites']} 个位点、"
            f"{cand['n_variants']} 条突变体。"
        )
        if cand["selected_label"]:
            st.caption(f"网格择优选中的配方：{cand['selected_label']}")
        grid = (cand_report or {}).get("grid")
        if grid:
            st.markdown("**自动调优：各候选配方的按位点留出表现**")
            grid_frame = pd.DataFrame(grid)
            grid_frame["params"] = grid_frame["params"].apply(
                lambda p: ", ".join(f"{k}={v}" for k, v in dict(p).items() if k != "verbosity")
            )
            st.dataframe(
                grid_frame.rename(
                    columns={
                        "label": "配方",
                        "params": "超参数",
                        "spearman_macro": "Macro Spearman",
                        "auc_macro": "Macro AUC",
                        "recall10_macro": "Recall@10",
                        "rmse": "RMSE",
                        "n_variants": "留出突变体数",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.caption(
            "口径须知：主指标来自**按位点留出**（外推到训练中从未出现的突变位点），与「挑候选突变体」"
            f"的目标一致；{RANK_ONLY_NOTE}。"
        )

    with st.expander("候选模型的折外预测明细（前 200 行）"):
        if cand_oof.empty:
            st.info("候选报告里没有折外预测明细。")
        else:
            st.dataframe(cand_oof.head(200), hide_index=True, width="stretch", height=320)

    col_adopt, col_discard = st.columns(2)
    with col_adopt:
        if st.button(
            "采纳新模型（替换线上）",
            type="primary",
            icon=":material/check_circle:",
            key="ingest_adopt",
            help="旧模型会先归档进 models/history/，可随时回退。",
        ):
            entry = adopt_candidate(candidate_dir)
            st.session_state["ingest_candidate_dir"] = None
            st.success(
                f"已采纳新模型：Macro Spearman {entry['spearman_macro']}、"
                f"Macro AUC {entry['auc_macro']}、Recall@10 {entry['recall10_macro']}。"
                f"旧模型已归档为 {Path(entry['archived_model']).name if entry['archived_model'] else '（无）'}。"
            )
            st.info(
                "注意：候选推荐表仍是按重训前的模型算的。如需同步刷新，请到「候选推荐」页重建推荐表。"
            )
    with col_discard:
        if st.button(
            "丢弃候选模型",
            icon=":material/delete:",
            key="ingest_discard",
            help="只删除候选模型与报告，已入库的实验数据不会受影响。",
        ):
            discard_candidate(candidate_dir)
            st.session_state["ingest_candidate_dir"] = None
            st.warning("已丢弃候选模型。已入库的实验数据保留在训练集里。")

# --------------------------------------------------------------------------- #
# 5. 训练历史
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("模型采纳历史")
history = load_training_history()
if history.empty:
    st.caption("还没有采纳过任何新模型。入库重训后点「采纳新模型」即会在此留下记录。")
else:
    show = history.rename(
        columns={
            "adopted_at": "采纳时间",
            "dataset_rows": "训练样本数",
            "spearman_macro": "Macro Spearman",
            "auc_macro": "Macro AUC",
            "recall10_macro": "Recall@10",
            "rmse": "RMSE",
            "archived_model": "归档旧模型",
            "candidate_dir": "候选目录",
            "note": "备注",
        }
    )
    st.dataframe(
        show[
            [
                "采纳时间",
                "训练样本数",
                "Macro Spearman",
                "Macro AUC",
                "Recall@10",
                "RMSE",
                "归档旧模型",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
