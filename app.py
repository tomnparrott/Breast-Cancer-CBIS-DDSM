from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from SRC.inference import (
    load_config,
    processed_dir_from_cfg,
    resolve_latest_run_dir,
    list_run_dirs,
    resolve_checkpoint_path,
    get_device,
    load_model_from_checkpoint,
    preprocess_series_dir,
    predict_proba,
    gradcam_resnet18,
    extract_zip_to_temp,
    ExtractedZip,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Breast Cancer Decision Support Dashboard", layout="wide")
st.title("Breast Cancer Detection (CBIS-DDSM) — Decision Support Dashboard")
st.caption("Research/demo tool. Not for clinical use.")


# -----------------------------
# Utilities
# -----------------------------
def safe_read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_latest_eval_metrics_dir(run_dir: Path) -> Path | None:
    """
    Expected structure:
      run_dir/eval_outputs/<YYYY-MM-DD>/<run_name>/metrics
    """
    root = run_dir / "eval_outputs"
    if not root.exists():
        return None
    date_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not date_dirs:
        return None
    date_dirs.sort(key=lambda p: p.name)
    latest_date_dir = date_dirs[-1]

    candidate = latest_date_dir / run_dir.name / "metrics"
    if candidate.exists():
        return candidate

    run_folders = [p for p in latest_date_dir.iterdir() if p.is_dir()]
    if not run_folders:
        return None
    run_folders.sort(key=lambda p: p.name)
    cand = run_folders[-1] / "metrics"
    return cand if cand.exists() else None


@st.cache_data
def load_manifest(processed_dir: Path) -> pd.DataFrame | None:
    p = processed_dir / "manifest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def add_density_group(df: pd.DataFrame) -> None:
    def to_group(v: object) -> str:
        s = str(v).strip()
        if s in {"1", "2"}:
            return "1-2"
        if s in {"3", "4"}:
            return "3-4"
        return "Unknown"

    if "breast_density" not in df.columns:
        df["density_group"] = "Unknown"
        return
    df["density_group"] = df["breast_density"].map(to_group).astype(str)


def add_outcome_column(df: pd.DataFrame) -> None:
    if "outcome" in df.columns:
        return
    df["outcome"] = "TN"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "outcome"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "outcome"] = "FN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "outcome"] = "TP"


def enrich_case_index(case_index: pd.DataFrame, manifest: pd.DataFrame | None) -> pd.DataFrame:
    if case_index is None or case_index.empty:
        return case_index

    if manifest is None or manifest.empty:
        add_density_group(case_index)
        return case_index

    if "image_dir" not in case_index.columns or "image_dir" not in manifest.columns:
        add_density_group(case_index)
        return case_index

    meta_cols = [
        "breast_density",
        "abnormality_type",
        "laterality",
        "view",
        "abnormality_id",
        "source_csv",
        "patient_id",
        "label",
    ]
    meta_cols = [c for c in meta_cols if c in manifest.columns]
    m = manifest[["image_dir"] + meta_cols].copy()

    df = case_index.copy()
    df = df.merge(m, on="image_dir", how="left", suffixes=("", "_m"))

    for c in ["breast_density", "abnormality_type", "laterality", "view", "abnormality_id", "source_csv"]:
        if c not in df.columns:
            df[c] = "Unknown"
        mc = f"{c}_m"
        if mc in df.columns:
            df[c] = df[c].astype(str)
            df[mc] = df[mc].astype(str)
            bad = df[c].isna() | (df[c].str.lower().isin({"unknown", "nan", ""}))
            df.loc[bad, c] = df.loc[bad, mc]

    for c in [col for col in df.columns if col.endswith("_m")]:
        df.drop(columns=[c], inplace=True)

    add_density_group(df)
    return df


def compute_subgroup_from_case_index(case_index: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if case_index is None or case_index.empty or group_col not in case_index.columns:
        return pd.DataFrame()

    thr = float(case_index["threshold_used"].iloc[0]) if "threshold_used" in case_index.columns else 0.5

    rows = []
    for gval, g in case_index.groupby(group_col):
        y_true = g["y_true"].astype(int).to_numpy()
        y_pred = g["y_pred"].astype(int).to_numpy()

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        rows.append(
            {
                "group_by": group_col,
                "group": str(gval),
                "n": int(len(g)),
                "pos": int((y_true == 1).sum()),
                "neg": int((y_true == 0).sum()),
                "threshold": float(thr),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def overlay_fig(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, alpha=alpha)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def heatmap_fig(cam: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.imshow(cam)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def triage_label(pred_label: int) -> str:
    return "Flag for radiologist review" if pred_label == 1 else "No flag"


def resolve_eval_threshold(case_index: pd.DataFrame | None, test_metrics: dict) -> float | None:
    if case_index is not None and not case_index.empty and "threshold_used" in case_index.columns:
        try:
            return float(case_index["threshold_used"].iloc[0])
        except Exception:
            pass

    for k in ["threshold", "threshold_at_spec_0.90"]:
        if k in test_metrics:
            try:
                return float(test_metrics[k])
            except Exception:
                continue
    return None


def cleanup_zip_session() -> None:
    old: ExtractedZip | None = st.session_state.get("uploaded_zip_obj", None)
    if old is not None:
        try:
            old.cleanup()
        except Exception:
            pass
    st.session_state["uploaded_zip_obj"] = None
    st.session_state["uploaded_zip_name"] = None


def decision_panel(prob: float, pred: int, thr: float, mode_text: str, y_true: int | None) -> None:
    rec = triage_label(pred)
    if pred == 1:
        st.warning(f"### Recommendation: {rec}")
    else:
        st.success(f"### Recommendation: {rec}")

    left, mid, right = st.columns(3)
    left.metric("Cancer risk score (0–1)", f"{prob:.3f}")
    mid.metric("Decision strictness (threshold)", f"{thr:.3f}")
    right.metric("Decision mode", mode_text)

    if y_true is None:
        st.info("Ground truth is not available for uploaded/unseen cases. This is a model prediction only.")
    else:
        st.write(f"**Ground truth label:** `{y_true}` (0 = benign, 1 = malignant)")

    with st.expander("What do these numbers mean? (plain English)", expanded=False):
        st.write(
            "- **Risk score**: the model’s estimate of how likely the case is malignant.\n"
            "- **Threshold**: how strict the system is. A higher threshold means fewer false alarms but more missed cancers.\n"
            "- **Recommendation**: if risk score ≥ threshold, the system flags the case for a radiologist to review.\n"
            "- This is **decision support**, not a diagnosis."
        )


def why_panel(viz_mode: str, show_gradcam: bool) -> None:
    st.write("#### Why this decision?")
    st.write(
        "The model learned patterns from training mammograms. The heatmap (Grad-CAM) highlights regions that most influenced the risk score."
    )
    if not show_gradcam or viz_mode == "Input only":
        st.caption("Grad-CAM is not currently displayed (switch Explainability view to Heatmap/Overlay and toggle Grad-CAM on).")
    else:
        st.caption("Grad-CAM shows **influence**, not a confirmed tumour location. It can highlight broader tissue patterns or artefacts.")


# -----------------------------
# Audit / policy merge helpers
# -----------------------------
def _norm_path_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\\\\", "/", regex=True)
        .str.replace("\\", "/", regex=False)
        .str.strip()
        .str.rstrip("/")
    )


def merge_audit_fields(case_index: pd.DataFrame | None, preds_meta: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Merge audit + density-policy fields from test_predictions_with_meta.csv into case_index.
    """
    if case_index is None or case_index.empty:
        return case_index
    if preds_meta is None or preds_meta.empty:
        return case_index
    if "image_dir" not in case_index.columns or "image_dir" not in preds_meta.columns:
        return case_index

    audit_cols = [c for c in preds_meta.columns if c.startswith("audit_")]

    preferred = [
        # audit
        "audit_flag_any",
        "audit_cam_outside_ratio",
        "audit_cam_edge_ratio",
        "audit_cam_ring_ratio",
        "audit_p_masked",
        "audit_delta_masked",
        # policy (Patch C)
        "threshold_density_policy",
        "y_pred_density_policy",
    ]

    cols = [c for c in preferred if c in preds_meta.columns]
    cols = list(dict.fromkeys(cols + [c for c in audit_cols if c not in cols]))

    left = case_index.copy()
    right = preds_meta.copy()

    # prevent _x/_y duplication
    drop_cols = [c for c in left.columns if c.startswith("audit_")]
    for c in ["threshold_density_policy", "y_pred_density_policy"]:
        if c in left.columns:
            drop_cols.append(c)
    left.drop(columns=drop_cols, errors="ignore", inplace=True)

    left["_img_norm"] = _norm_path_series(left["image_dir"])
    right["_img_norm"] = _norm_path_series(right["image_dir"])
    right = right.drop_duplicates(subset=["_img_norm"], keep="first")

    m = right[["_img_norm"] + cols].copy()
    out = left.merge(m, on="_img_norm", how="left").drop(columns=["_img_norm"])

    # ensure main fields exist even if missing in preds_meta
    for c in preferred:
        if c not in out.columns:
            out[c] = np.nan

    return out


def apply_shortcut_filters(
    df: pd.DataFrame,
    only_flagged: bool,
    min_outside: float | None,
    use_delta_filter: bool,
    max_delta: float | None,
) -> pd.DataFrame:
    out = df

    if only_flagged and "audit_flag_any" in out.columns:
        out = out[out["audit_flag_any"].fillna(0).astype(int) == 1]

    if (
        min_outside is not None
        and float(min_outside) > 0.0
        and "audit_cam_outside_ratio" in out.columns
    ):
        out = out[out["audit_cam_outside_ratio"].fillna(0.0).astype(float) >= float(min_outside)]

    if use_delta_filter and max_delta is not None and "audit_delta_masked" in out.columns:
        out = out[out["audit_delta_masked"].astype(float) <= float(max_delta)]

    return out


def sort_triage(df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if sort_mode == "Risk score (p)":
        return df.sort_values("y_prob", ascending=False)

    if sort_mode == "Largest drop when masked":
        if "audit_delta_masked" in df.columns:
            return df.sort_values("audit_delta_masked", ascending=True)
        return df.sort_values("y_prob", ascending=False)

    if sort_mode == "Highest CAM outside breast":
        if "audit_cam_outside_ratio" in df.columns:
            return df.sort_values("audit_cam_outside_ratio", ascending=False)
        return df.sort_values("y_prob", ascending=False)

    return df.sort_values("y_prob", ascending=False)


def reset_case_review_filters() -> None:
    for k in [
        "cr_outcome",
        "cr_density",
        "cr_abn",
        "cr_audit_only",
        "cr_min_outside",
        "cr_use_delta",
        "cr_max_delta",
        "cr_sort_mode",
        "cr_case_select",
    ]:
        st.session_state.pop(k, None)
    st.rerun()


# -----------------------------
# Sidebar: Run + model settings
# -----------------------------
cfg = load_config()
proc_dir = processed_dir_from_cfg(cfg)
latest_run_dir = resolve_latest_run_dir(proc_dir)
all_runs = list_run_dirs(proc_dir)

with st.sidebar:
    st.header("Model & Run")

    prefer_cuda = st.toggle("Prefer CUDA", value=True, key="sb_prefer_cuda")
    device = get_device(prefer_cuda=prefer_cuda)
    st.write(f"Device: `{device}`")

    if all_runs:
        run_names = [p.name for p in all_runs]
        default_idx = run_names.index(latest_run_dir.name) if latest_run_dir.name in run_names else 0
        chosen_run = st.selectbox("Run", run_names, index=default_idx, key="sb_run")
        run_dir = proc_dir / "runs" / chosen_run
    else:
        run_dir = latest_run_dir
        st.warning("No run folders found under Data/Processed/runs. Using Data/Processed directly.")

    ckpt_choice = st.selectbox("Checkpoint", ["best", "final"], index=0, key="sb_ckpt")
    ckpt_path = resolve_checkpoint_path(run_dir, ckpt_choice)

    st.divider()
    st.header("Decision Settings")

    threshold_slider = st.slider("Manual threshold", 0.0, 1.0, 0.5, 0.01, key="sb_threshold")
    tta = st.toggle("TTA (flip average)", value=False, key="sb_tta")
    show_gradcam = st.toggle("Grad-CAM", value=True, key="sb_cam")
    cam_alpha = st.slider("Heatmap opacity", 0.05, 0.80, 0.35, 0.05, key="sb_cam_alpha")

    viz_mode = st.selectbox(
        "Explainability view",
        ["Input only", "Heatmap only", "Overlay"],
        index=2,
        key="sb_viz_mode",
    )

    st.divider()
    st.header("Quick links")
    st.write(f"Run dir:\n`{run_dir}`")
    st.write(f"Checkpoint:\n`{ckpt_path}`")


@st.cache_resource
def get_model_cached(ckpt_path_str: str, device: str):
    ckpt = Path(ckpt_path_str)
    return load_model_from_checkpoint(ckpt, device=device)


model = get_model_cached(str(ckpt_path), device)
img_size = int(cfg["data"]["img_size"])


# -----------------------------
# Load artefacts
# -----------------------------
def load_case_index(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_metrics_dir(run_dir)
    if mdir is not None:
        p2 = mdir / "case_index.csv"
        if p2.exists():
            return pd.read_csv(p2)

    p1 = run_dir / "case_index.csv"
    if p1.exists():
        return pd.read_csv(p1)

    return None


def load_test_predictions_with_meta(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_metrics_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "test_predictions_with_meta.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "density_group" not in df.columns:
            add_density_group(df)
        if "y_true" in df.columns and "y_pred" in df.columns and "outcome" not in df.columns:
            add_outcome_column(df)
        return df
    return None


def load_subgroup_metrics(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_metrics_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "subgroup_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def load_test_metrics(run_dir: Path) -> dict:
    mdir = find_latest_eval_metrics_dir(run_dir)
    if mdir is None:
        return {}
    p = mdir / "test_metrics.json"
    return safe_read_json(p)


def load_run_info(run_dir: Path) -> dict:
    p = run_dir / "run_info.json"
    return safe_read_json(p)


manifest_df = load_manifest(proc_dir)

case_index_raw = load_case_index(run_dir)
case_index = enrich_case_index(case_index_raw, manifest_df) if case_index_raw is not None else None

preds_meta = load_test_predictions_with_meta(run_dir)
if preds_meta is not None and not preds_meta.empty:
    case_index = merge_audit_fields(case_index, preds_meta)

subgroup_file = load_subgroup_metrics(run_dir)
test_metrics = load_test_metrics(run_dir)
run_info = load_run_info(run_dir)

eval_threshold = resolve_eval_threshold(case_index, test_metrics)

with st.sidebar:
    st.divider()
    st.header("Operating point")
    use_eval_threshold = st.toggle(
        "Use eval operating threshold (recommended)",
        value=True,
        disabled=(eval_threshold is None),
        key="sb_use_eval_thr",
        help="Uses the threshold used during evaluation (e.g., chosen to hit a target specificity).",
    )
    if eval_threshold is None:
        st.warning("Eval threshold not found. Using manual threshold slider.")
    else:
        st.write(f"Eval threshold: **{eval_threshold:.3f}**")

    use_density_policy = st.toggle(
        "Use density-aware policy (val-calibrated)",
        value=False,
        key="sb_use_density_policy",
        help="Uses per-density thresholds learned on VAL to target the same specificity per density group.",
    )
    if use_density_policy:
        st.caption("When enabled, policy overrides the threshold for TEST-set cases shown in the dashboard.")


# -----------------------------
# Tabs
# -----------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Plain English Summary", "Case Review", "Batch Triage", "Subgroup Performance", "Model Info"]
)

# -----------------------------
# Tab 0: Plain English Summary
# -----------------------------
with tab0:
    st.subheader("Plain English Summary")

    if case_index is None or case_index.empty or not test_metrics:
        st.warning("Evaluation artefacts not found for this run. Run: `py -m SRC.eval` and `py -m SRC.case_index`.")
    else:
        tp = int((case_index["outcome"] == "TP").sum())
        fp = int((case_index["outcome"] == "FP").sum())
        tn = int((case_index["outcome"] == "TN").sum())
        fn = int((case_index["outcome"] == "FN").sum())

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)

        auc = float(test_metrics.get("auc", np.nan))
        thr = float(test_metrics.get("threshold", eval_threshold if eval_threshold is not None else 0.5))

        st.info(
            f"At the chosen operating point (threshold ≈ {thr:.3f}) the system is tuned to reduce false alarms "
            "and will flag fewer cases for review."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test cases", f"{len(case_index)}")
        c2.metric("Sensitivity", f"{sens:.3f}")
        c3.metric("Specificity", f"{spec:.3f}")
        c4.metric("Precision", f"{prec:.3f}")
        c5.metric("AUC", f"{auc:.3f}")

        st.write(
            f"- Malignant: **{tp + fn}** (flagged **{tp}**, missed **{fn}**)\n"
            f"- Benign: **{tn + fp}** (not flagged **{tn}**, false alarms **{fp}**)\n"
        )

# -----------------------------
# Tab 1: Case Review
# -----------------------------
with tab1:
    st.subheader("Case Review (per image)")
    st.caption("Select a case to see the input image, Grad-CAM, and shortcut-risk indicators (when available).")

    colA, colB = st.columns([1, 1])

    with colA:
        source = st.radio(
            "Input source",
            ["From case_index (test set)", "Manual folder path", "Upload ZIP (unseen)"],
            horizontal=True,
            key="cr_source",
        )

        selected_series_dir: Path | None = None
        meta: dict = {}
        y_true: int | None = None
        per_case_eval_thr: float | None = None

        if source == "From case_index (test set)":
            if case_index is None or case_index.empty:
                st.warning("case_index.csv not found for this run. Run: `py -m SRC.case_index`")
            else:
                if st.button("Reset Case Review filters", key="cr_reset_btn"):
                    reset_case_review_filters()

                f1, f2, f3 = st.columns(3)
                with f1:
                    outcome_filter = st.multiselect(
                        "Outcome (TP/FP/TN/FN)",
                        options=sorted(case_index["outcome"].unique().tolist()),
                        default=sorted(case_index["outcome"].unique().tolist()),
                        key="cr_outcome",
                    )
                with f2:
                    density_filter = st.multiselect(
                        "Breast density group",
                        options=sorted(case_index["density_group"].unique().tolist()),
                        default=sorted(case_index["density_group"].unique().tolist()),
                        key="cr_density",
                    )
                with f3:
                    abn_filter = st.multiselect(
                        "Abnormality type",
                        options=sorted(case_index["abnormality_type"].unique().tolist()),
                        default=sorted(case_index["abnormality_type"].unique().tolist()),
                        key="cr_abn",
                    )

                filtered = case_index[
                    case_index["outcome"].isin(outcome_filter)
                    & case_index["density_group"].isin(density_filter)
                    & case_index["abnormality_type"].isin(abn_filter)
                ].copy()

                audit_available = ("audit_flag_any" in filtered.columns) and filtered["audit_flag_any"].notna().any()

                if audit_available:
                    st.divider()
                    st.write("#### Shortcut-risk filters (audit)")
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        only_flagged = st.checkbox("Flagged only", value=False, key="cr_audit_only")
                    with a2:
                        min_outside = st.slider(
                            "Min CAM outside-breast",
                            0.0,
                            1.0,
                            0.0,
                            0.01,
                            key="cr_min_outside",
                        )
                    with a3:
                        use_delta = st.checkbox("Use mask ablation filter", value=False, key="cr_use_delta")

                    max_delta = None
                    if use_delta:
                        max_delta = st.slider(
                            "Max Δ prob (masked - original)",
                            -1.0,
                            1.0,
                            -0.05,
                            0.01,
                            key="cr_max_delta",
                        )

                    filtered = apply_shortcut_filters(
                        filtered,
                        only_flagged=only_flagged,
                        min_outside=min_outside,
                        use_delta_filter=use_delta,
                        max_delta=max_delta,
                    )

                st.write(f"Filtered cases: **{len(filtered)}**")
                if filtered.empty:
                    st.warning("No cases match the current filters. Relax filters to continue.")
                else:
                    sort_options = ["Risk score (p)"]
                    if "audit_delta_masked" in filtered.columns and filtered["audit_delta_masked"].notna().any():
                        sort_options.append("Largest drop when masked")
                    if "audit_cam_outside_ratio" in filtered.columns and filtered["audit_cam_outside_ratio"].notna().any():
                        sort_options.append("Highest CAM outside breast")

                    sort_mode = st.selectbox("Sort cases by", sort_options, index=0, key="cr_sort_mode")
                    filtered = sort_triage(filtered, sort_mode)

                    filtered_view = filtered.head(min(500, len(filtered))).copy()

                    extra = ""
                    if "audit_flag_any" in filtered_view.columns and filtered_view["audit_flag_any"].notna().any():
                        outside = filtered_view.get("audit_cam_outside_ratio", pd.Series([np.nan] * len(filtered_view)))
                        delta = filtered_view.get("audit_delta_masked", pd.Series([np.nan] * len(filtered_view)))
                        extra = (
                            " | audit="
                            + filtered_view["audit_flag_any"].fillna(0).astype(int).astype(str)
                            + " | out="
                            + outside.fillna(np.nan).astype(float).round(2).astype(str)
                            + " | d="
                            + delta.fillna(np.nan).astype(float).round(2).astype(str)
                        )

                    filtered_view["display"] = (
                        filtered_view["risk_rank"].astype(str)
                        + " | "
                        + filtered_view["outcome"].astype(str)
                        + " | p="
                        + filtered_view["y_prob"].round(4).astype(str)
                        + extra
                        + " | dens="
                        + filtered_view["density_group"].astype(str)
                        + " | "
                        + filtered_view["abnormality_type"].astype(str)
                        + " | "
                        + filtered_view["view"].astype(str)
                    )

                    idx = st.selectbox(
                        "Select case",
                        options=filtered_view.index.tolist(),
                        format_func=lambda i: filtered_view.loc[i, "display"],
                        key="cr_case_select",
                    )

                    row = filtered_view.loc[idx]
                    selected_series_dir = Path(row["image_dir"])
                    meta = row.to_dict()

                    try:
                        y_true = int(meta.get("y_true")) if "y_true" in meta and not pd.isna(meta.get("y_true")) else None
                    except Exception:
                        y_true = None

                    try:
                        per_case_eval_thr = (
                            float(meta.get("threshold_used"))
                            if "threshold_used" in meta and not pd.isna(meta.get("threshold_used"))
                            else None
                        )
                    except Exception:
                        per_case_eval_thr = None

                    with st.expander("Technical metadata (raw)", expanded=False):
                        st.json(meta)

        elif source == "Manual folder path":
            p = st.text_input("Folder path containing DICOMs for ONE case", value="", key="cr_manual_path")
            if p.strip():
                selected_series_dir = Path(p.strip())
                if not selected_series_dir.exists():
                    st.error("That folder path does not exist.")

        else:
            st.write("Upload a ZIP containing the DICOM files for a single case.")
            up = st.file_uploader("ZIP of DICOMs", type=["zip"], key="cr_zip")

            if st.button("Clear uploaded ZIP", key="cr_zip_clear"):
                cleanup_zip_session()
                st.rerun()

            zip_obj: ExtractedZip | None = st.session_state.get("uploaded_zip_obj", None)
            zip_name: str | None = st.session_state.get("uploaded_zip_name", None)

            if up is not None:
                cleanup_zip_session()
                data = up.getvalue()
                zip_obj = extract_zip_to_temp(data)
                st.session_state["uploaded_zip_obj"] = zip_obj
                st.session_state["uploaded_zip_name"] = up.name
                zip_name = up.name

            if zip_obj is not None:
                selected_series_dir = zip_obj.series_dir
                st.success(f"Using uploaded ZIP: {zip_name}")
                st.caption(f"Extracted to: {selected_series_dir}")
                y_true = None
            else:
                st.info("No ZIP uploaded yet.")

    with colB:
        if selected_series_dir is None:
            st.info("Select a case, enter a folder path, or upload a ZIP.")
        else:
            try:
                prep = preprocess_series_dir(
                    series_dir=selected_series_dir,
                    img_size=img_size,
                    num_channels=3,
                    crop_foreground=True,
                )

                if use_eval_threshold:
                    thr = per_case_eval_thr if per_case_eval_thr is not None else eval_threshold
                    if thr is None:
                        thr = threshold_slider
                    mode_text = "Eval operating point"
                else:
                    thr = threshold_slider
                    mode_text = "Manual"

                prob = predict_proba(model, prep.x, device=device, tta=tta)
                pred = int(prob >= float(thr))

                # ---- Density policy override (TEST-set only) ----
                if use_density_policy and meta and source == "From case_index (test set)":
                    thr_pol = meta.get("threshold_density_policy", None)
                    pred_pol = meta.get("y_pred_density_policy", None)

                    if pd.notna(thr_pol):
                        thr = float(thr_pol)
                        mode_text = "Density-aware policy"
                        pred = int(prob >= float(thr))

                    if pd.notna(pred_pol):
                        pred = int(pred_pol)

                decision_panel(prob=prob, pred=pred, thr=float(thr), mode_text=mode_text, y_true=y_true)
                why_panel(viz_mode=viz_mode, show_gradcam=show_gradcam)

                if meta and (meta.get("audit_flag_any") is not None):
                    st.write("#### Shortcut-risk indicators (audit)")
                    m1, m2, m3 = st.columns(3)
                    try:
                        m1.metric("Audit flagged", f"{int(float(meta.get('audit_flag_any', 0)))}")
                    except Exception:
                        m1.metric("Audit flagged", "n/a")

                    try:
                        m2.metric("CAM outside-breast", f"{float(meta.get('audit_cam_outside_ratio')):.3f}")
                    except Exception:
                        m2.metric("CAM outside-breast", "n/a")

                    try:
                        m3.metric("Δ prob (masked - orig)", f"{float(meta.get('audit_delta_masked')):.3f}")
                    except Exception:
                        m3.metric("Δ prob (masked - orig)", "n/a")

                st.write("#### Image evidence")
                if viz_mode == "Input only":
                    st.image(prep.img_display, caption="Preprocessed mammogram (model input)", use_container_width=True)
                else:
                    if not show_gradcam:
                        st.image(prep.img_display, caption="Preprocessed mammogram (model input)", use_container_width=True)
                        st.warning("Grad-CAM is off, so heatmap views are unavailable.")
                    else:
                        cam = gradcam_resnet18(model, prep.x, device=device)
                        if viz_mode == "Heatmap only":
                            st.pyplot(heatmap_fig(cam))
                        else:
                            st.pyplot(overlay_fig(prep.img_display, cam, alpha=cam_alpha))

            except Exception as e:
                st.error(str(e))

# -----------------------------
# Tab 2: Batch Triage
# -----------------------------
with tab2:
    st.subheader("Batch Triage (Test Set)")

    if case_index is None or case_index.empty:
        st.warning("case_index.csv not found. Run: `py -m SRC.case_index`")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            outc = st.multiselect(
                "Outcome",
                sorted(case_index["outcome"].unique().tolist()),
                default=sorted(case_index["outcome"].unique().tolist()),
                key="bt_outcome",
            )
        with c2:
            dens = st.multiselect(
                "Density group",
                sorted(case_index["density_group"].unique().tolist()),
                default=sorted(case_index["density_group"].unique().tolist()),
                key="bt_density",
            )
        with c3:
            abn = st.multiselect(
                "Abnormality",
                sorted(case_index["abnormality_type"].unique().tolist()),
                default=sorted(case_index["abnormality_type"].unique().tolist()),
                key="bt_abn",
            )
        with c4:
            view = st.multiselect(
                "View",
                sorted(case_index["view"].unique().tolist()),
                default=sorted(case_index["view"].unique().tolist()),
                key="bt_view",
            )

        df = case_index[
            case_index["outcome"].isin(outc)
            & case_index["density_group"].isin(dens)
            & case_index["abnormality_type"].isin(abn)
            & case_index["view"].isin(view)
        ].copy()

        audit_available = ("audit_flag_any" in df.columns) and df["audit_flag_any"].notna().any()
        if audit_available:
            with st.expander("Shortcut-risk filters (audit)", expanded=False):
                a1, a2, a3 = st.columns(3)
                with a1:
                    only_flagged = st.checkbox("Flagged only", value=False, key="bt_audit_only")
                with a2:
                    min_outside = st.slider("Min CAM outside-breast", 0.0, 1.0, 0.0, 0.01, key="bt_min_outside")
                with a3:
                    use_delta = st.checkbox("Use mask ablation filter", value=False, key="bt_use_delta")

                max_delta = None
                if use_delta:
                    max_delta = st.slider("Max Δ prob (masked - original)", -1.0, 1.0, -0.05, 0.01, key="bt_max_delta")

            df = apply_shortcut_filters(
                df,
                only_flagged=only_flagged,
                min_outside=min_outside,
                use_delta_filter=use_delta,
                max_delta=max_delta,
            )

        sort_options = ["Risk score (p)"]
        if "audit_delta_masked" in df.columns and df["audit_delta_masked"].notna().any():
            sort_options.append("Largest drop when masked")
        if "audit_cam_outside_ratio" in df.columns and df["audit_cam_outside_ratio"].notna().any():
            sort_options.append("Highest CAM outside breast")

        sort_mode = st.selectbox("Sort by", sort_options, index=0, key="bt_sort")
        df = sort_triage(df, sort_mode)

        # If policy is enabled and policy preds exist, compute a policy outcome
        if use_density_policy and "y_pred_density_policy" in df.columns:
            y_true = df["y_true"].astype(int)
            y_pred_pol = df["y_pred_density_policy"].fillna(df["y_pred"]).astype(int)

            df["outcome_density_policy"] = "TN"
            df.loc[(y_true == 1) & (y_pred_pol == 1), "outcome_density_policy"] = "TP"
            df.loc[(y_true == 0) & (y_pred_pol == 1), "outcome_density_policy"] = "FP"
            df.loc[(y_true == 1) & (y_pred_pol == 0), "outcome_density_policy"] = "FN"

        st.write(f"Rows: **{len(df)}**")

        cols = [
            "risk_rank",
            "patient_id",
            "abnormality_id",
            "view",
            "laterality",
            "density_group",
            "abnormality_type",
            "y_true",
            "y_prob",
            "y_pred",
            "outcome",
        ]
        for extra in ["audit_flag_any", "audit_cam_outside_ratio", "audit_delta_masked"]:
            if extra in df.columns:
                cols.append(extra)

        if use_density_policy:
            for extra in ["threshold_density_policy", "y_pred_density_policy", "outcome_density_policy"]:
                if extra in df.columns:
                    cols.append(extra)

        cols = [c for c in cols if c in df.columns]

        st.dataframe(df[cols], use_container_width=True, height=520)

        st.download_button(
            "Download filtered triage CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="triage_filtered.csv",
            mime="text/csv",
        )

# -----------------------------
# Tab 3: Subgroup Performance
# -----------------------------
with tab3:
    st.subheader("Subgroup Performance")

    if case_index is None or case_index.empty:
        st.warning("case_index.csv not found. Run: `py -m SRC.case_index`")
    else:
        group_by = st.selectbox(
            "Group by",
            ["density_group", "breast_density", "abnormality_type", "view", "laterality"],
            key="sg_group_by_live",
        )
        df = compute_subgroup_from_case_index(case_index, group_by)
        if df.empty:
            st.warning("Could not compute subgroup metrics from case_index.")
        else:
            st.dataframe(df, use_container_width=True)

            labels = df["group"].astype(str).tolist()
            sens = df["sensitivity"].astype(float).tolist()
            spec = df["specificity"].astype(float).tolist()

            fig1, ax1 = plt.subplots()
            ax1.bar(labels, sens)
            ax1.set_title(f"Sensitivity by {group_by}")
            ax1.set_ylabel("Sensitivity")
            ax1.set_xlabel(group_by)
            plt.xticks(rotation=30, ha="right")
            fig1.tight_layout()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.bar(labels, spec)
            ax2.set_title(f"Specificity by {group_by}")
            ax2.set_ylabel("Specificity")
            ax2.set_xlabel(group_by)
            plt.xticks(rotation=30, ha="right")
            fig2.tight_layout()
            st.pyplot(fig2)

# -----------------------------
# Tab 4: Model Info
# -----------------------------
with tab4:
    st.subheader("Model & Run Information")

    c1, c2 = st.columns(2)
    with c1:
        st.write("### run_info.json")
        if run_info:
            st.json(run_info)
        else:
            st.warning("run_info.json not found in run dir.")

    with c2:
        st.write("### Latest test_metrics.json")
        if test_metrics:
            st.json(test_metrics)
        else:
            st.warning("test_metrics.json not found (run eval).")

    st.divider()
    st.write("### Evaluation Figures (latest)")
    mdir = find_latest_eval_metrics_dir(run_dir)
    if mdir is None:
        st.info("No eval_outputs found for this run.")
    else:
        fig_dir = mdir.parent / "figures"
        if not fig_dir.exists():
            st.info("No figures folder found.")
        else:
            imgs = [
                ("ROC", fig_dir / "roc_curve.png"),
                ("PR", fig_dir / "pr_curve.png"),
                ("Confusion matrix", fig_dir / "confusion_matrix.png"),
                ("Calibration", fig_dir / "calibration_curve.png"),
            ]
            cols = st.columns(2)
            for i, (name, p) in enumerate(imgs):
                if p.exists():
                    with cols[i % 2]:
                        st.image(str(p), caption=name, use_container_width=True)