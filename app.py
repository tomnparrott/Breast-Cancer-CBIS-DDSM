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


def find_latest_eval_metrics_dir(run_dir: Path):
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
    """
    Big, layman-friendly “what is the system recommending?” panel.
    """
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
        st.caption(
            "Important: Grad-CAM shows **influence**, not a confirmed tumour location. It can highlight broader tissue patterns or artefacts."
        )


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
    p1 = run_dir / "case_index.csv"
    if p1.exists():
        return pd.read_csv(p1)
    mdir = find_latest_eval_metrics_dir(run_dir)
    if mdir is not None:
        p2 = mdir / "case_index.csv"
        if p2.exists():
            return pd.read_csv(p2)
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
    st.write(
        "This page is designed for non-technical readers. It summarises what the system does and how well it performed on the test set."
    )

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
        ap = float(test_metrics.get("avg_precision", np.nan))
        thr = float(test_metrics.get("threshold", eval_threshold if eval_threshold is not None else 0.5))

        st.info(
            f"**At the chosen operating point (threshold ≈ {thr:.3f})** the system is tuned to reduce false alarms "
            "and will flag fewer cases for review."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test cases", f"{len(case_index)}")
        c2.metric("Sensitivity (cancers caught)", f"{sens:.3f}")
        c3.metric("Specificity (benign not flagged)", f"{spec:.3f}")
        c4.metric("Precision (flagged that are cancer)", f"{prec:.3f}")
        c5.metric("AUC (ranking quality)", f"{auc:.3f}")

        st.write("#### What this means in practice")
        st.write(
            f"- Out of **{tp + fn} malignant** test cases, the system **flagged {tp}** and **missed {fn}** at this strict setting.\n"
            f"- Out of **{tn + fp} benign** test cases, the system **correctly did not flag {tn}**, but **flagged {fp}** as false alarms.\n"
            "- If you lower the threshold, the system will catch more cancers but will also produce more false alarms.\n"
        )

        with st.expander("Important limitations (plain English)", expanded=False):
            st.write(
                "- This model was trained and tested on a historical dataset (CBIS-DDSM). Performance may change on new hospitals/scanners.\n"
                "- The heatmap is an explainability tool; it does not prove the tumour location.\n"
                "- In a real setting this would only be used to support a radiologist, not replace clinical judgment."
            )

# -----------------------------
# Tab 1: Case Review
# -----------------------------
with tab1:
    st.subheader("Case Review (per image)")
    st.caption("This page is designed to explain each decision clearly and show supporting evidence (image + heatmap).")

    colA, colB = st.columns([1, 1])

    with colA:
        source = st.radio(
            "Input source",
            ["From case_index (test set)", "Manual folder path", "Upload ZIP (unseen)"],
            horizontal=True,
            key="cr_source",
        )

        selected_series_dir = None
        meta = {}
        y_true: int | None = None
        per_case_eval_thr = None

        if source == "From case_index (test set)":
            if case_index is None or case_index.empty:
                st.warning("case_index.csv not found for this run. Run: `py -m SRC.case_index`")
            else:
                f1, f2, f3 = st.columns(3)
                with f1:
                    outcome_filter = st.multiselect(
                        "Outcome (TP/FP/TN/FN)",
                        options=sorted(case_index["outcome"].unique().tolist()),
                        default=sorted(case_index["outcome"].unique().tolist()),
                        key="cr_outcome",
                        help="TP=True positive, FP=False positive, TN=True negative, FN=False negative.",
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

                st.write(f"Filtered cases: **{len(filtered)}**")

                max_show = min(500, len(filtered))
                filtered_view = filtered.head(max_show).copy()
                filtered_view["display"] = (
                    filtered_view["risk_rank"].astype(str)
                    + " | "
                    + filtered_view["outcome"].astype(str)
                    + " | p="
                    + filtered_view["y_prob"].round(4).astype(str)
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

                if "y_true" in meta and not pd.isna(meta["y_true"]):
                    try:
                        y_true = int(meta["y_true"])
                    except Exception:
                        y_true = None

                if "threshold_used" in meta and not pd.isna(meta["threshold_used"]):
                    try:
                        per_case_eval_thr = float(meta["threshold_used"])
                    except Exception:
                        per_case_eval_thr = None

                # Make image_dir easy to copy
                st.write("**Image folder path (for ZIP tests):**")
                st.code(str(meta.get("image_dir", "")), language="text")

                # Keep raw metadata out of the way (but accessible)
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

                # Big, layman-friendly block FIRST
                decision_panel(prob=prob, pred=pred, thr=float(thr), mode_text=mode_text, y_true=y_true)
                why_panel(viz_mode=viz_mode, show_gradcam=show_gradcam)

                # Visualisation
                st.write("#### Image evidence")
                if viz_mode == "Input only":
                    st.image(prep.img_display, caption="Preprocessed mammogram (model input)", clamp=True, width="stretch")
                else:
                    if not show_gradcam:
                        st.image(prep.img_display, caption="Preprocessed mammogram (model input)", clamp=True, width="stretch")
                        st.warning("Grad-CAM is off, so heatmap views are unavailable.")
                    else:
                        cam = gradcam_resnet18(model, prep.x, device=device)
                        if viz_mode == "Heatmap only":
                            st.pyplot(heatmap_fig(cam))
                            st.caption("Grad-CAM heatmap: regions that most influenced the risk score.")
                        else:
                            st.pyplot(overlay_fig(prep.img_display, cam, alpha=cam_alpha))
                            st.caption("Overlay: input + Grad-CAM (influential regions).")

                # Optional: light summary of the case if available
                if meta:
                    st.write("#### Case summary")
                    summary = {
                        "View": meta.get("view"),
                        "Laterality": meta.get("laterality"),
                        "Density group": meta.get("density_group"),
                        "Abnormality type": meta.get("abnormality_type"),
                        "Outcome (eval threshold)": meta.get("outcome"),
                        "Saved probability (if test case)": meta.get("y_prob"),
                    }
                    st.table(pd.DataFrame([summary]))

            except Exception as e:
                st.error(str(e))


# -----------------------------
# Tab 2: Batch Triage
# -----------------------------
with tab2:
    st.subheader("Batch Triage (Test Set)")
    st.caption("Filter and export predictions from case_index.csv (risk-sorted).")

    if case_index is None or case_index.empty:
        st.warning("case_index.csv not found. Run: `py -m SRC.case_index`")
    else:
        tp = int((case_index["outcome"] == "TP").sum())
        fp = int((case_index["outcome"] == "FP").sum())
        tn = int((case_index["outcome"] == "TN").sum())
        fn = int((case_index["outcome"] == "FN").sum())
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        thr = float(test_metrics.get("threshold", eval_threshold if eval_threshold is not None else 0.5))

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Cases", f"{len(case_index)}")
        s2.metric("Sensitivity", f"{sens:.3f}")
        s3.metric("Specificity", f"{spec:.3f}")
        s4.metric("Precision", f"{prec:.3f}")
        s5.metric("Eval threshold", f"{thr:.3f}")

        st.divider()

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

        df = df.sort_values("y_prob", ascending=False)

        st.write(f"Rows: **{len(df)}**")
        st.dataframe(
            df[
                [
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
            ],
            width="stretch",
            height=520,
        )

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
    st.caption("Uses subgroup_metrics.csv if valid; otherwise computes subgroup metrics from case_index (recommended).")

    subgroup_use_file = False
    if subgroup_file is not None and not subgroup_file.empty and "group" in subgroup_file.columns:
        unknown_rate = (subgroup_file["group"].astype(str).str.lower() == "unknown").mean()
        subgroup_use_file = unknown_rate < 0.80

    prefer_live = st.toggle(
        "Compute subgroup metrics from case_index (live)",
        value=not subgroup_use_file,
        key="sg_live_toggle",
        help="Live computation guarantees it matches the case_index shown in the dashboard.",
    )

    if case_index is None or case_index.empty:
        st.warning("case_index.csv not found. Run: `py -m SRC.case_index`")
    else:
        if prefer_live or not subgroup_use_file:
            group_by = st.selectbox(
                "Group by",
                ["density_group", "breast_density", "abnormality_type", "view", "laterality"],
                key="sg_group_by_live",
            )
            df = compute_subgroup_from_case_index(case_index, group_by)
            if df.empty:
                st.warning("Could not compute subgroup metrics from case_index.")
            else:
                st.dataframe(df, width="stretch")

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
        else:
            group_by = st.selectbox(
                "Group by",
                sorted(subgroup_file["group_by"].unique().tolist()),
                key="sg_group_by_file",
            )
            df = subgroup_file[subgroup_file["group_by"] == group_by].copy()
            st.dataframe(df, width="stretch")


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
                        st.image(str(p), caption=name, width="stretch")