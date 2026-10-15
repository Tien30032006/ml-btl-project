"""
streamlit_app.py
TV4 - Evaluation + Demo + Report Lead

Demo 3 trang: Overview / Single Scoring / Batch Scoring.
Model: models/best_model.pkl + models/best_model_meta.json
Quy ước: P(default) cao -> rủi ro cao. Cost matrix: FN = 5, FP = 1.

Chạy: streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import (  # noqa: E402
    COST_FN,
    COST_FP,
    compute_shap_values,
    expected_cost,
    metrics_from_proba,
)

MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR    = PROJECT_ROOT / "data"

TARGET_COL        = "credit_risk"
GOOD_LABEL_IN_CSV = 1   # trong CSV: 1 = good, 0 = bad

st.set_page_config(page_title="Credit Risk Demo", page_icon="💳", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    model_path = MODELS_DIR / "best_model.pkl"
    meta_path  = MODELS_DIR / "best_model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), meta


@st.cache_data
def load_reference():
    """Dữ liệu train dùng để dựng form nhập (danh sách lựa chọn, min/max/median)."""
    df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    features = json.loads(
        (DATA_DIR / "processed" / "final_features.json").read_text(encoding="utf-8")
    )
    return df, [c for c in features if c in df.columns]


@st.cache_data
def load_comparison():
    path = REPORTS_DIR / "final_comparison.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


# ------------------------------------------------------------------ helpers
def decide(p_default: float, threshold: float) -> str:
    return "REJECT" if p_default >= threshold else "APPROVE"


def risk_band(p: float, threshold: float) -> str:
    """Mức rủi ro tính theo ngưỡng từ chối để nhất quán với quyết định."""
    if p >= threshold:
        return "🔴 Cao"
    return "🟡 Trung bình" if p >= 0.5 * threshold else "🟢 Thấp"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("💳 Credit Risk")
page = st.sidebar.radio("Trang", ["Overview", "Single Scoring", "Batch Scoring"])

model, meta = load_model()
if model is None:
    st.error("Chưa có `models/best_model.pkl`. Chạy `python src/evaluate.py` để tạo.")
    st.stop()

ref_df, FEATURES = load_reference()

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{meta['name']}**")
default_thr = float(meta["threshold"])
threshold = st.sidebar.slider(
    "Ngưỡng từ chối (P(default) ≥ t)", 0.05, 0.95, default_thr, 0.01,
    help=f"Mặc định = ngưỡng tối ưu theo cost ({default_thr:.2f}). Lý thuyết 1/6 ≈ 0.17.",
)
st.sidebar.caption(f"Cost matrix: FN = {COST_FN}, FP = {COST_FP}")

# =================================================================== OVERVIEW
if page == "Overview":
    st.title("Dự đoán rủi ro vỡ nợ - German Credit Data")
    st.write(
        "Hệ thống chấm điểm hồ sơ vay: ước lượng **xác suất vỡ nợ** và đề xuất "
        "**duyệt / từ chối** theo ngưỡng tối thiểu hóa chi phí."
    )

    tm = meta["test_metrics"]
    # st.metric cắt tên dài thành "Random Fores...", nên hiển thị tên model riêng
    st.info(f"🏆 Best model: **{meta['name']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ngưỡng tối ưu", f"{default_thr:.2f}")
    c2.metric("ROC-AUC (test)", f"{tm['ROC-AUC']:.3f}")
    c3.metric("Expected cost (test)", f"{tm['Cost @t* (test)']:.3f}",
              delta=f"{tm['Cost @t* (test)'] - tm['Cost @0.5']:+.3f} so với ngưỡng 0.5",
              delta_color="inverse")

    st.subheader("Vì sao không dùng Accuracy?")
    st.write(
        f"Dữ liệu mất cân bằng (~30% default) và chi phí bất đối xứng: bỏ sót một khách "
        f"default (FN) tốn **{COST_FN}**, từ chối nhầm khách tốt (FP) tốn **{COST_FP}**. "
        "Vì vậy dùng PR-AUC, Recall và Expected Cost, đồng thời hạ ngưỡng quyết định "
        "xuống thấp hơn 0.5."
    )

    cmp_df = load_comparison()
    if cmp_df is not None:
        st.subheader("So sánh các mô hình (test set, positive = default)")
        # reset_index + hide_index: cột Model không còn che các cột bên cạnh
        tbl = cmp_df.reset_index()
        st.dataframe(
            tbl.style.format("{:.3f}", subset=list(cmp_df.columns)).highlight_min(
                subset=["Cost @0.5", "Cost @t* (test)"], color="#c8e6c9"),
            hide_index=True,
        )
        st.caption("Threshold* chọn bằng xác suất out-of-fold trên train; "
                   "best model chọn theo cost OOF trên train (không nhìn test).")

    figs = [("roc_pr_curves.png", "ROC & PR"),
            ("cost_curves.png", "Cost curve"),
            ("confusion_default.png", "Confusion @0.5"),
            ("confusion_best_threshold.png", "Confusion @ngưỡng tối ưu"),
            ("shap_summary.png", "SHAP")]
    for fname, cap in figs:
        if (FIGURES_DIR / fname).exists():
            st.subheader(cap)
            st.image(str(FIGURES_DIR / fname))

# ============================================================ SINGLE SCORING
elif page == "Single Scoring":
    st.title("Chấm điểm một hồ sơ")

    with st.form("single"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURES):
            col = cols[i % 3]
            s = ref_df[feat]
            if pd.api.types.is_numeric_dtype(s):
                lo, hi = int(s.min()), int(s.max())
                inputs[feat] = col.number_input(
                    feat,
                    min_value=lo,
                    value=int(s.median()),
                    step=1,
                    help=f"Khoảng trong train: {lo} – {hi}",
                )
            else:
                opts = sorted(s.dropna().unique().tolist())
                inputs[feat] = col.selectbox(
                    feat, opts, index=opts.index(s.mode()[0])
                )
        submitted = st.form_submit_button("Chấm điểm", type="primary")

    if submitted:
        row = pd.DataFrame([inputs])[FEATURES]
        p = float(model.predict_proba(row)[:, 1][0])
        verdict = decide(p, threshold)

        a, b, c = st.columns(3)
        a.metric("P(default)", f"{p:.1%}")
        b.metric("Mức rủi ro", risk_band(p, threshold))
        c.metric("Quyết định", verdict)
        st.progress(min(max(p, 0.0), 1.0))

        if verdict == "REJECT":
            st.error(f"P(default) = {p:.1%} ≥ ngưỡng {threshold:.2f} → đề xuất **TỪ CHỐI**.")
        else:
            st.success(f"P(default) = {p:.1%} < ngưỡng {threshold:.2f} → đề xuất **DUYỆT**.")

        st.caption(
            f"Chi phí kỳ vọng nếu DUYỆT ≈ {COST_FN}·p = {COST_FN * p:.2f}; "
            f"nếu TỪ CHỐI ≈ {COST_FP}·(1−p) = {COST_FP * (1 - p):.2f}. "
            "(Mang tính tham khảo: xác suất của SVM/RF có thể chưa hiệu chỉnh tốt.)"
        )

        with st.expander("Giải thích (SHAP) - đặc trưng nào đẩy rủi ro lên/xuống"):
            try:
                sv, names = compute_shap_values(model, row, background=ref_df[FEATURES])
                contrib = (pd.Series(sv[0], index=names)
                           .sort_values(key=np.abs, ascending=False).head(10))
                st.bar_chart(contrib.iloc[::-1])
                st.caption("Dương = tăng P(default); âm = giảm P(default).")
            except Exception as exc:
                st.info(f"Không tạo được giải thích cho model này ({exc}).")

# ============================================================ BATCH SCORING
else:
    st.title("Chấm điểm hàng loạt (CSV)")
    st.write(
        f"Tải lên file CSV có đủ {len(FEATURES)} cột đặc trưng. "
        f"Nếu có thêm cột `{TARGET_COL}` (1 = good, 0 = bad) hệ thống sẽ tính luôn metric."
    )
    st.download_button(
        "⬇️ Tải file mẫu",
        ref_df[FEATURES + [TARGET_COL]].head(10).to_csv(index=False).encode("utf-8"),
        "sample_input.csv", "text/csv",
    )

    up = st.file_uploader("Chọn file CSV", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as exc:
            st.error(f"Không đọc được file: {exc}")
            st.stop()

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Thiếu cột: {', '.join(missing)}")
            st.stop()

        # Cảnh báo giá trị categorical chưa từng thấy (model sẽ bỏ qua âm thầm)
        unseen = {}
        for c in FEATURES:
            if not pd.api.types.is_numeric_dtype(ref_df[c]):
                new_vals = set(df[c].dropna().astype(str)) - set(ref_df[c].astype(str))
                if new_vals:
                    unseen[c] = sorted(new_vals)
        if unseen:
            detail = "; ".join(f"`{c}`: {', '.join(v)}" for c, v in unseen.items())
            st.warning(f"Có giá trị chưa xuất hiện trong dữ liệu train (sẽ bị bỏ qua khi mã hóa): {detail}")

        try:
            p = model.predict_proba(df[FEATURES])[:, 1]
        except Exception as exc:
            st.error(f"Dữ liệu không hợp lệ (sai kiểu/giá trị): {exc}")
            st.stop()

        out = df.copy()
        out["p_default"] = p.round(4)
        out["decision"]  = ["REJECT" if v >= threshold else "APPROVE" for v in p]
        out["risk_band"] = [risk_band(v, threshold) for v in p]

        a, b, c = st.columns(3)
        a.metric("Số hồ sơ", len(out))
        b.metric("Từ chối", int((out["decision"] == "REJECT").sum()))
        c.metric("P(default) trung bình", f"{p.mean():.1%}")

        if TARGET_COL in df.columns:
            y = (df[TARGET_COL] != GOOD_LABEL_IN_CSV).astype(int)
            st.subheader("Đánh giá (có nhãn)")
            if y.nunique() == 2:
                m = metrics_from_proba(y, p, threshold)
                st.dataframe(pd.DataFrame(m, index=["@ngưỡng hiện tại"]).round(4))
            else:
                st.dataframe(pd.DataFrame(
                    {"Expected Cost": [expected_cost(y, (p >= threshold).astype(int))]}
                ).round(4))
                st.caption("Chỉ có một lớp nhãn nên không tính được AUC/Precision/Recall.")

        st.subheader("Kết quả")
        st.dataframe(out)
        st.download_button(
            "⬇️ Tải kết quả",
            out.to_csv(index=False).encode("utf-8"),
            "scored_applications.csv",
            "text/csv",
        )