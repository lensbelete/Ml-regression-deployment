import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Parkinson's UPDRS Monitor", layout="wide")
st.title("Parkinson's Disease Severity Monitor")
st.caption(
    "Predicts a patient's total UPDRS score over time from their clinician-scored "
    "baseline recordings plus a personalised trend. Monitoring mode only: "
    "a brand-new patient with no scored recordings cannot be predicted reliably."
)
 
model = joblib.load("parkinsons_model.joblib")
GLOBAL_COEF = model["global_coef"]
W = model["w"]
CAL_FRAC = model["cal_frac"]
 
with st.sidebar:
    st.header("How it works")
    st.markdown(
        f"""
1. Upload a CSV with columns `test_time` and `total_UPDRS`
   (optional `subject#` for several patients).
2. The first **{int(CAL_FRAC*100)}%** of each patient's recordings (by time)
   are the calibration window and **must** have `total_UPDRS`.
3. For later recordings the app predicts:
   `baseline + (1-w)*global_slope*dt + w*patient_slope*dt`
   with w = {W}.
4. If later rows also have `total_UPDRS`, accuracy is reported.
"""
    )
    st.info(
        "Held-out test result (9 unseen patients): RMSE 4.77, MAE 3.23, R2 0.67."
    )
    sample = pd.DataFrame(
        {
            "subject#": [1] * 10,
            "test_time": [5, 12, 19, 26, 33, 40, 47, 54, 61, 68],
            "total_UPDRS": [34.4, 34.9, 35.4, 35.8, None, None, None, None, None, None],
        }
    )
    st.download_button(
        "Download sample CSV", sample.to_csv(index=False), "sample.csv", "text/csv"
    )
 
 
def predict_patient(g):
    g = g.sort_values("test_time").reset_index(drop=True)
    n = len(g)
    k = max(8, int(n * CAL_FRAC))
    cal = g.iloc[:k]
    ev = g.iloc[k:].copy()
    if cal["total_UPDRS"].isna().any():
        raise ValueError(
            f"First {k} rows (calibration window) need total_UPDRS values."
        )
    t0 = cal["test_time"].mean()
    base = cal["total_UPDRS"].mean()
    own_slope = np.polyfit(cal["test_time"], cal["total_UPDRS"], 1)[0]
    dt = ev["test_time"].values - t0
    ev["predicted_UPDRS"] = base + (1 - W) * GLOBAL_COEF * dt + W * own_slope * dt
    return cal, ev, base, own_slope
 
 
file = st.file_uploader("Upload patient CSV", type=["csv"])
 
if file is None:
    st.write("Upload a CSV to get started, or download the sample from the sidebar.")
    st.stop()
 
df = pd.read_csv(file)
needed = {"test_time", "total_UPDRS"}
if not needed.issubset(df.columns):
    st.error(f"CSV must contain columns: {sorted(needed)}")
    st.stop()
if "subject#" not in df.columns:
    df["subject#"] = 1
 
all_preds = []
for sid, g in df.groupby("subject#"):
    st.subheader(f"Patient {sid}")
    if len(g) < 20:
        st.warning("Need at least 20 recordings (8+ for calibration). Skipped.")
        continue
    try:
        cal, ev, base, slope = predict_patient(g)
    except ValueError as e:
        st.error(str(e))
        continue
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline UPDRS", f"{base:.1f}")
    c2.metric("Personal slope (pts/day)", f"{slope:.3f}")
    c3.metric("Predicted rows", len(ev))
 
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(cal["test_time"], cal["total_UPDRS"], "o", label="Calibration (given)")
    ax.plot(ev["test_time"], ev["predicted_UPDRS"], "-", label="Predicted")
    if ev["total_UPDRS"].notna().any():
        ax.plot(ev["test_time"], ev["total_UPDRS"], ".", alpha=0.4, label="Actual")
        m = ev["total_UPDRS"].notna()
        err = ev.loc[m, "total_UPDRS"] - ev.loc[m, "predicted_UPDRS"]
        st.write(
            f"RMSE: **{np.sqrt((err**2).mean()):.2f}**  |  MAE: **{err.abs().mean():.2f}**"
        )
    ax.set_xlabel("test_time (days)")
    ax.set_ylabel("total_UPDRS")
    ax.legend()
    st.pyplot(fig)
 
    out = ev[["test_time", "predicted_UPDRS"]].copy()
    out["subject#"] = sid
    all_preds.append(out)
 
if all_preds:
    result = pd.concat(all_preds)
    st.download_button(
        "Download predictions CSV", result.to_csv(index=False), "predictions.csv", "text/csv"
    )
 