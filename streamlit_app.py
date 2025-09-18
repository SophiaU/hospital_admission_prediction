# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import boto3
import os
import io
from prophet.serialize import model_from_json, model_to_json
import xgboost as xgb
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Hospital Admissions Forecast")

# ---------------------------
# CONFIG - set secrets in Streamlit or replace placeholders
# ---------------------------
MINIO_ENDPOINT = st.secrets.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = st.secrets.get("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = st.secrets.get("MINIO_SECRET_KEY", "admin123")
MINIO_BUCKET = st.secrets.get("MINIO_BUCKET", "gold")
# BUCKET = st.secrets.get("BUCKET", "silver")

# Create boto3 client for MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(ttl=3600)
def download_from_minio(bucket: str, key: str) -> bytes:
    """Download object as bytes. Raises on failure."""
    bio = io.BytesIO()
    s3.download_fileobj(bucket, key, bio)
    bio.seek(0)
    return bio.read()

@st.cache_data(ttl=3600)
def load_prophet_model_from_minio(bucket: str, key: str):
    """
    Try multiple ways to deserialize a saved Prophet model:
      - joblib / pickle bytes (pickle.loads)
      - JSON (model_from_json) if bytes decode to JSON text
    Returns model or raises.
    """
    b = download_from_minio(bucket, key)
    # try joblib (file-like)
    try:
        return joblib.load(io.BytesIO(b))
    except Exception:
        pass
    # try json text -> model_from_json
    try:
        text = b.decode("utf-8")
        return model_from_json(text)
    except Exception as e:
        raise RuntimeError(f"Failed to load Prophet model from {bucket}/{key}: {e}")

@st.cache_data(ttl=3600)
def load_joblib_from_minio(bucket: str, key: str):
    b = download_from_minio(bucket, key)
    return joblib.load(io.BytesIO(b))

def safe_read_parquet_bytes(b: bytes):
    return pd.read_parquet(io.BytesIO(b))

# Build iterative future features (lags and rolling averages)
def build_future_features(history_df, future_dates, prophet_yhat_series, features_list):
    hist = history_df.sort_values("ds").copy()
    # keep 'ds' and 'y' at front if present
    other_cols = [c for c in hist.columns if c not in ['ds', 'y']]
    hist = hist[['ds', 'y'] + other_cols]
    hist.set_index('ds', inplace=True)

    # create sim_y mapping (pandas.Timestamp -> float)
    sim_y = {pd.to_datetime(idx): float(val) for idx, val in hist['y'].dropna().items()}

    rows = []
    for ds in pd.to_datetime(future_dates):
        dow = ds.dayofweek
        is_weekend = 1 if dow >= 5 else 0
        month = ds.month

        prev1 = ds - pd.Timedelta(days=1)
        prev7 = ds - pd.Timedelta(days=7)
        lag_1 = sim_y.get(prev1, np.nan)
        lag_7 = sim_y.get(prev7, np.nan)

        vals = [sim_y[d] for k in range(1, 8) for d in [ds - pd.Timedelta(days=k)] if d in sim_y]
        ma_7 = float(np.mean(vals)) if len(vals) > 0 else np.nan

        # operational features fallback
        op_features = {}
        for op in [
            'scheduled_appointments', 'scheduled_surgeries', 'occupancy',
            'doctors_on_shift', 'nurses_on_shift', 'helpers_on_shift'
        ]:
            if op in hist.columns:
                # try to get value for exact ds index; if not, NaN
                op_val = hist[op].get(ds, np.nan)

                # If Series, reduce to scalar
                if isinstance(op_val, pd.Series):
                    if op_val.empty:
                        op_val = np.nan
                    else:
                        op_val = op_val.iloc[0]   # or .mean() if that makes more sense

                # If still NaN, fallback to forward-fill last known
                if pd.isna(op_val):
                    filled = hist[op].ffill()
                    op_val = filled.iloc[-1] if not filled.empty else 0.0

                op_features[op] = float(op_val)
            else:
                op_features[op] = 0.0


        row = {}
        for f in features_list:
            if f == 'lag_1':
                row[f] = 0.0 if pd.isna(lag_1) else float(lag_1)
            elif f == 'lag_7':
                row[f] = 0.0 if pd.isna(lag_7) else float(lag_7)
            elif f == 'ma_7':
                row[f] = 0.0 if pd.isna(ma_7) else float(ma_7)
            elif f in ('day_of_week', 'dow'):
                row[f] = int(dow)
            elif f == 'is_weekend':
                row[f] = int(is_weekend)
            elif f == 'month':
                row[f] = int(month)
            elif f in op_features:
                row[f] = op_features[f]
            else:
                row[f] = 0.0

        # use Prophet baseline as simulated y for next-day lags
        try:
            prophet_val = float(prophet_yhat_series.loc[pd.to_datetime(ds)])
        except Exception:
            prophet_val = 0.0
        sim_y[ds] = prophet_val

        rows.append((ds, row))

    feat_df = pd.DataFrame([r for _, r in rows], index=[r[0] for r in rows])
    feat_df.index.name = 'ds'
    feat_df.reset_index(inplace=True)
    return feat_df


@st.cache_data
# Get the department ids | Load the parquet file with Spark
# Read just the dept_id column from the parquet
def get_departments(bucket: str, key: str):
    # download parquet file from MinIO
    data_bytes = download_from_minio(bucket, key)
    # read parquet directly from bytes
    df = pd.read_parquet(io.BytesIO(data_bytes), columns=["dept_id"], engine="pyarrow")
    return sorted(df["dept_id"].unique().tolist())
# MinIO bucket and key
BUCKET = "silver"
KEY = "hospital/curated/dept_daily_train_ready/part-00000-7ca1d62f-5425-4dbd-9515-f63ce63d507f-c000.snappy.parquet"
# get unique department IDs
dept_ids = get_departments(BUCKET, KEY)

# Streamlit selectbox
# dept_id = st.selectbox("Select Department", options=dept_ids, index=0)



# ---------------------------
# UI
# ---------------------------
st.title("🏥 Hospital Admissions: Hybrid Forecast Dashboard")

# left column for controls
with st.sidebar:
    st.header("Settings")
    # dept_id = st.selectbox("Select Department", options=["101", "102", "103", "104"], index=0)
    dept_id = st.selectbox("Select Department", options=dept_ids, index=0)
    days_ahead = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=14, step=1)
    avg_los = st.slider("Average Length of Stay (days)", min_value=1, max_value=14, value=3)
    nurses_per_patient = st.slider("Nurses per patient (FTE)", min_value=0.05, max_value=1.0, value=0.2, step=0.01)
    doctors_per_patient = st.slider("Doctors per patient (FTE)", min_value=0.01, max_value=0.5, value=0.06, step=0.01)
    load_history_from_minio = st.checkbox("Load historical dept data from MinIO (preferred)", value=True)
    st.markdown("---")
    st.markdown("**MinIO / model paths expected:**")
    st.text(f"{MINIO_BUCKET}/hospital/models/prophet_dept_{dept_id}.pkl")
    st.text(f"{MINIO_BUCKET}/hospital/models/xgb_residual.pkl")
    st.text(f"{MINIO_BUCKET}/hospital/models/features.pkl")
    st.text(f"{BUCKET}/hospital/curated/dept_daily_train_ready/part-00000-7ca1d62f-5425-4dbd-9515-f63ce63d507f-c000.snappy.parquet")

# load models & features
col1, col2 = st.columns([2, 1])

# --- Column 1: Hospital image ---
with col1:
    st.image(
        "https://www.philips.com/c-dam/corporate/innovation-and-you/global/sept-2021/4-Departmental-dashboard-body.jpg",
        caption="",
        use_container_width=True
    )

# --- Column 2: Model Operations ---
with col2:
    st.markdown("### Model ops")
    # features list
    try:
        st.write("Loading features list...")
        features_bytes = download_from_minio(MINIO_BUCKET, "hospital/models/features.pkl")
        features_list = joblib.load(io.BytesIO(features_bytes))
        if not isinstance(features_list, (list, tuple)):
            raise ValueError("features.pkl content is not a list")
        st.success("features.pkl loaded")
    except Exception as e:
        st.error(f"Could not load features.pkl from MinIO: {e}")
        fallback = st.text_input(
            "Enter feature names comma-separated (fallback)",
            value="lag_1,lag_7,ma_7,scheduled_appointments,scheduled_surgeries,occupancy,doctors_on_shift,nurses_on_shift,helpers_on_shift,day_of_week,is_weekend,month"
        )
        features_list = [f.strip() for f in fallback.split(",")]

    # XGB model
    try:
        st.write("Loading XGB residual model...")
        xgb_model = load_joblib_from_minio(MINIO_BUCKET, "hospital/models/xgb_residual.pkl")
        st.success("XGB residual model loaded")
    except Exception as e:
        st.warning(f"Could not load xgb_residual.pkl: {e}")
        xgb_model = None

    # Prophet model
    try:
        st.write("Loading Prophet model for dept...")
        prophet_key = f"hospital/models/prophet_dept_{dept_id}.pkl"
        prophet_model = load_prophet_model_from_minio(MINIO_BUCKET, prophet_key)
        st.success(f"Prophet model for dept {dept_id} loaded")
    except Exception as e:
        st.warning(f"Could not load Prophet model for dept {dept_id}: {e}")
        prophet_model = None

# ---------------------------
# Load historical data for department (from MinIO if available, else user upload)
history_df = None
if load_history_from_minio:
    try:
        hist_key = "hospital/curated/dept_daily_train_ready/part-00000-7ca1d62f-5425-4dbd-9515-f63ce63d507f-c000.snappy.parquet"
        hist_bytes = download_from_minio(BUCKET, hist_key)
        history_df = safe_read_parquet_bytes(hist_bytes)
        # Ensure correct column names for Prophet
        history_df = history_df.rename(
            columns={'admission_date': 'ds', 'admissions_next7': 'y'}
        )
        # Filter history to selected dept:
        history_df = history_df[history_df["dept_id"] == dept_id].copy()
        # Ensure datetime
        history_df['ds'] = pd.to_datetime(history_df['ds'])
        # Sort and reset index
        history_df = history_df.sort_values('ds').reset_index(drop=True)
        
        st.success("Historical data loaded from MinIO")
    except Exception as e:
        st.warning(f"No history file in MinIO for dept {dept_id} or failed to read: {e}")

if history_df is None:
    st.info("Please upload historical department data (CSV/Parquet). Must include columns: ds (date) or admission_date, and y (actual) or admissions_next7.")
    uploaded = st.file_uploader("Upload history CSV/parquet", type=["csv", "parquet"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                tmp = pd.read_csv(uploaded)
            else:
                tmp = pd.read_parquet(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.stop()

        # Normalize column names: accept either ds or admission_date; y or admissions_next7
        col_map = {}
        if 'ds' not in tmp.columns and 'admission_date' in tmp.columns:
            col_map['admission_date'] = 'ds'
        if 'y' not in tmp.columns and 'admissions_next7' in tmp.columns:
            col_map['admissions_next7'] = 'y'
        tmp = tmp.rename(columns=col_map)

        # If ds exists but not datetime, try parse
        if 'ds' in tmp.columns:
            tmp['ds'] = pd.to_datetime(tmp['ds'])
        else:
            st.error("Uploaded file must include a date column named 'ds' or 'admission_date'.")
            st.stop()

        if 'y' not in tmp.columns:
            st.error("Uploaded file must include a target column named 'y' or 'admissions_next7'.")
            st.stop()

        history_df = tmp.copy()
        st.success("Uploaded historical data loaded")

if history_df is None:
    st.stop()  # cannot proceed

# ensure correct dtype & ordering
history_df['ds'] = pd.to_datetime(history_df['ds'])
history_df = history_df.sort_values('ds').reset_index(drop=True)

# ---------------
# Guard: ensure prophet_model available
if prophet_model is None:
    st.error("Prophet model is not loaded — cannot generate baseline forecast. Upload model to MinIO or disable MinIO loading.")
    st.stop()

# Generate Prophet future & baseline
try:
    future = prophet_model.make_future_dataframe(periods=days_ahead, freq='D')
    prophet_fc = prophet_model.predict(future)
    # ensure prophet_future is a Series indexed by ds
    prophet_future = prophet_fc[['ds', 'yhat']].tail(days_ahead).set_index('ds')['yhat']
except Exception as e:
    st.error(f"Prophet prediction failed: {e}")
    st.stop()

# Build future features
# future_dates = prophet_future.index
feat_future = build_future_features(history_df, prophet_future.index, prophet_future, features_list)

# align by ds explicitly
feat_future = feat_future.set_index('ds')
prophet_series = prophet_future.reindex(feat_future.index)  # align by index
feat_future = feat_future.assign(prophet_yhat = prophet_series.values)

# Prepare X for XGB in correct order
X_future = feat_future[features_list].fillna(0)

# Predict residuals with XGB (if available)
if xgb_model is not None:
    try:
        xgb_resid_pred = xgb_model.predict(X_future)
    except Exception as e:
        st.warning(f"XGB prediction failed, falling back to zero residuals: {e}")
        xgb_resid_pred = np.zeros(len(X_future))
else:
    xgb_resid_pred = np.zeros(len(X_future))

########################

# # Compose final forecast
result_df = feat_future.reset_index()[['ds']].copy()
result_df['prophet_yhat'] = feat_future['prophet_yhat'].values
result_df['xgb_resid_pred'] = xgb_resid_pred
result_df['hybrid_yhat'] = result_df['prophet_yhat'] + result_df['xgb_resid_pred']
result_df['dept_id'] = dept_id


# ---------------------------
# Visualisations
# ---------------------------
st.write("", history_df.tail(28))

# Forecast Per Department
st.header(f"Forecasts — Department {dept_id}")

# top KPIs (safe indexing)
if len(result_df) > 0:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Next day (Hybrid)", f"{result_df['hybrid_yhat'].iloc[0]:.1f}")
    col_b.metric("7-day avg (Hybrid)", f"{result_df['hybrid_yhat'].head(7).mean():.2f}")
    col_c.metric("Peak (next horizon)", f"{result_df['hybrid_yhat'].max():.1f}")
else:
    st.info("No forecast rows generated.")

# ----- Department-level comparison [Time series plot]----------#
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history_df['ds'].tail(28), history_df['y'].tail(28), label="Actual (history)", linewidth=2)
ax.plot(result_df['ds'], result_df['prophet_yhat'], label="Prophet baseline")
ax.plot(result_df['ds'], result_df['hybrid_yhat'], label="Hybrid forecast", linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Admissions")
ax.legend()
st.pyplot(fig)

# Forecast table
st.subheader("Forecast table (next days)")
st.dataframe(result_df.set_index('ds')[['prophet_yhat', 'xgb_resid_pred', 'hybrid_yhat']].round(2))

# Occupancy estimate
st.subheader("Estimated bed occupancy (approx.)")
result_df['occupancy_est'] = result_df['hybrid_yhat'].rolling(window=avg_los, min_periods=1).sum()
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(result_df['ds'], result_df['occupancy_est'], label='Estimated Occupied Beds (approx)')
ax2.set_xlabel("Date")
ax2.set_ylabel("Beds")
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.3)
ax2.legend()
st.pyplot(fig2)

# Staffing estimates
st.subheader("Staffing estimates (FTE) from forecast")
result_df['nurses_needed'] = result_df['hybrid_yhat'] * nurses_per_patient
result_df['doctors_needed'] = result_df['hybrid_yhat'] * doctors_per_patient
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(result_df['ds'], result_df['nurses_needed'], label='Nurses FTE')
ax3.plot(result_df['ds'], result_df['doctors_needed'], label='Doctors FTE')
ax3.set_xlabel("Date")
ax3.set_ylabel("FTE")
ax3.legend()
st.pyplot(fig3)

# --------------------------------------
# Residual histogram
# --------------------------------------
st.subheader("Residual (histogram) - Prophet")
# compute prophet predictions on historical ds
hist_pred = prophet_model.predict(history_df[['ds']])
hist_pred = hist_pred[['ds','yhat']].rename(columns={'yhat':'prophet_yhat'})
history_eval = history_df.merge(hist_pred, on='ds', how='left')

# residuals
history_eval['prophet_resid'] = history_eval['y'] - history_eval['prophet_yhat']

# metrics
mae = history_eval['prophet_resid'].abs().mean()
rmse = np.sqrt((history_eval['prophet_resid']**2).mean())
col_a, col_b = st.columns(2)
col_a.metric("Prophet historical MAE", f"{mae:.2f}")
col_b.metric("Prophet historical RMSE", f"{rmse:.2f}")
# st.write(f"Prophet historical MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# plot residual histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(history_eval['prophet_resid'], bins=30, edgecolor='black')
ax.set_title("Prophet Residuals Distribution")
ax.set_xlabel("Residuals (y - yhat)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# --------------------------------------
# Feature importance
# --------------------------------------
if xgb_model is not None:
    try:
        st.subheader("XGBoost Feature Importances")
        importances = xgb_model.feature_importances_
        fi = pd.DataFrame({"feature": features_list, "importance": importances}).sort_values("importance", ascending=False)
        st.table(fi.head(20))
    except Exception as e:
        st.warning(f"Could not display XGB feature importances: {e}")

# Option to download forecast CSV
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("Download forecast CSV", csv, file_name=f"forecast_dept_{dept_id}.csv", mime="text/csv")
