import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES = ["Study_Hours", "Attendance", "Previous_Score"]
TARGET = "Final_Score"

# Column names used by the Kaggle dataset (and common variants) -> our names
ALIASES = {
    "Study_Hours": ["Hours_Studied", "Study_Hours", "Hours Studied", "StudyHours"],
    "Attendance": ["Attendance", "Attendance_Percentage"],
    "Previous_Score": ["Previous_Scores", "Previous_Score", "Previous Scores"],
    "Final_Score": ["Exam_Score", "Final_Score", "Performance Index", "Final_Exam_Score"],
}

# ---------- Styling ----------
st.markdown(
    """
<style>
    .block-container {padding-top: 2rem;}
    .result-card {
        padding: 1.6rem; border-radius: 16px; text-align: center;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; margin-top: 0.5rem;
        box-shadow: 0 8px 24px rgba(79,70,229,.25);
    }
    .result-card h1 {color: white; font-size: 3.4rem; margin: 0;}
    .result-card p {margin: 0; opacity: 0.9;}
    .source-pill {
        display:inline-block; padding:.2rem .7rem; border-radius:999px;
        background:#eef2ff; color:#4338ca; font-size:.82rem; font-weight:600;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Data loading ----------
def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Pick the needed columns from a CSV and rename them to our schema."""
    out = {}
    lower = {c.lower(): c for c in raw.columns}
    for new, options in ALIASES.items():
        for opt in options:
            if opt.lower() in lower:
                out[new] = pd.to_numeric(raw[lower[opt.lower()]], errors="coerce")
                break
        else:
            raise ValueError(
                f"Required column for '{new}' not found. Expected one of: {options}"
            )
    return pd.DataFrame(out).dropna().reset_index(drop=True)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return standardise(pd.read_csv(path))


@st.cache_data
def synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    hours = rng.uniform(1, 8, n)
    att = rng.uniform(40, 100, n)
    prev = rng.uniform(30, 95, n)
    final = np.clip(5 * hours + 0.25 * att + 0.40 * prev + rng.normal(0, 5, n), 0, 100)
    return pd.DataFrame(
        {"Study_Hours": hours, "Attendance": att, "Previous_Score": prev, "Final_Score": final}
    )


def find_local_csv():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    for f in files:  # prefer the Kaggle file by name
        if "studentperformancefactors" in os.path.basename(f).lower().replace("_", ""):
            return f
    return files[0] if files else None


# ---------- Sidebar: dataset source ----------
st.sidebar.header("🗂 Dataset")
uploaded = st.sidebar.file_uploader("Upload Kaggle CSV (optional)", type="csv")

df, source = None, ""
try:
    if uploaded is not None:
        df = standardise(pd.read_csv(uploaded))
        source = f"Uploaded: {uploaded.name}"
    else:
        local = find_local_csv()
        if local:
            df = load_csv(local)
            source = f"Kaggle dataset: {os.path.basename(local)}"
except Exception as e:
    st.sidebar.error(f"Could not read dataset: {e}")

using_kaggle = df is not None
if not using_kaggle:
    df = synthetic_dataset()
    source = "Synthetic sample data (Kaggle CSV not found)"
    st.sidebar.warning(
        "No Kaggle CSV found. Put `StudentPerformanceFactors.csv` in the `data/` "
        "folder (see README) or upload it above."
    )

hours_unit = "per week" if using_kaggle else "per day"

# ---------- Model ----------
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

# ---------- Header ----------
st.title("📊 Student Performance Prediction")
st.write(
    "Predict a student's estimated final exam score from study hours, "
    "attendance, and previous exam score using **Linear Regression**."
)
st.markdown(f'<span class="source-pill">{source} · {len(df):,} students</span>', unsafe_allow_html=True)

# ---------- Sidebar: inputs (ranges follow the dataset) ----------
st.sidebar.markdown("---")
st.sidebar.header("🎓 Student Details")


def rng_of(col):
    return float(np.floor(df[col].min())), float(np.ceil(df[col].max()))


h_lo, h_hi = rng_of("Study_Hours")
a_lo, a_hi = rng_of("Attendance")
p_lo, p_hi = rng_of("Previous_Score")

study_hours = st.sidebar.slider(
    f"Study Hours ({hours_unit})", h_lo, h_hi, float(round((h_lo + h_hi) / 2)), 1.0
)
attendance = st.sidebar.slider("Attendance (%)", a_lo, a_hi, float(round((a_lo + a_hi) / 2)), 1.0)
previous_score = st.sidebar.slider(
    "Previous Exam Score", p_lo, p_hi, float(round((p_lo + p_hi) / 2)), 1.0
)
predict_clicked = st.sidebar.button("🔮 Predict Score", use_container_width=True)
st.sidebar.caption("Algorithm: Linear Regression (only)")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prediction", "📈 Model Performance", "🗂 Dataset", "ℹ️ About"]
)

# ----- Tab 1: Prediction -----
with tab1:
    st.subheader("Predicted Final Score")
    input_df = pd.DataFrame([[study_hours, attendance, previous_score]], columns=FEATURES)
    prediction = float(np.clip(model.predict(input_df)[0], 0, 100))

    if predict_clicked:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""<div class="result-card">
                    <p>Estimated Final Score</p>
                    <h1>{prediction:.1f}</h1>
                    <p>out of 100</p></div>""",
                unsafe_allow_html=True,
            )
        with col_b:
            if prediction >= 85:
                grade, msg = "A — Excellent", "Outstanding expected performance."
            elif prediction >= 70:
                grade, msg = "B — Good", "Good performance; a bit more effort can reach the top."
            elif prediction >= 55:
                grade, msg = "C — Average", "Improve study hours and attendance to do better."
            else:
                grade, msg = "D — Needs Improvement", "Needs more study time and regular attendance."
            st.metric("Estimated Grade", grade)
            st.progress(int(prediction))
            st.info(msg)

        avg = float(y.mean())
        st.metric("Dataset average score", f"{avg:.1f}", delta=f"{prediction - avg:+.1f} vs average")
        st.markdown("#### Your Inputs")
        st.dataframe(input_df.round(2), use_container_width=True, hide_index=True)
    else:
        st.info("👈 Enter the student's details in the sidebar and click **Predict Score**.")

# ----- Tab 2: Model performance -----
with tab2:
    st.subheader("Model Evaluation (Test Data)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.3f}")
    m2.metric("Mean Absolute Error", f"{mae:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.5, color="#4f46e5", s=14)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with c2:
        st.markdown("#### Feature Coefficients")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(FEATURES, model.coef_, color="#7c3aed")
        ax2.set_xlabel("Coefficient value")
        ax2.grid(alpha=0.3, axis="x")
        st.pyplot(fig2)
        st.caption(f"Intercept: {model.intercept_:.2f}")

    st.markdown("#### Feature Correlation with Final Score")
    corr = df.corr()[TARGET].drop(TARGET).round(3)
    st.dataframe(corr.rename("Correlation").to_frame(), use_container_width=True)

# ----- Tab 3: Dataset -----
with tab3:
    st.subheader(f"Dataset ({len(df):,} students)")
    st.caption(source)
    st.dataframe(df.round(2), use_container_width=True, height=350)
    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)
    st.markdown("#### Score Distribution")
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.hist(df[TARGET], bins=25, color="#4f46e5", alpha=0.85)
    ax3.set_xlabel("Final Score")
    ax3.set_ylabel("Students")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)
    st.download_button(
        "⬇️ Download dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "student_data.csv",
        "text/csv",
    )

# ----- Tab 4: About -----
with tab4:
    st.subheader("About this Project")
    st.write(
        "This project predicts a student's final exam score using "
        "**Linear Regression**, the only Machine Learning algorithm used."
    )
    st.markdown(
        f"""
- **Inputs:** Study hours, attendance %, previous exam score
- **Output:** Estimated final score (0–100)
- **Data:** {source} ({len(df):,} rows), 80/20 train-test split
- **Kaggle source:** *Student Performance Factors* (`lainguyn123/student-performance-factors`)
- **Libraries:** Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib
"""
    )
