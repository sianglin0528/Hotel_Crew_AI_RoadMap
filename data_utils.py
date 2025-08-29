# data_utils.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

# Prophet
from prophet import Prophet

# XGBoost + sklearn
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ---------- I/O ----------
def load_occupancy_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "occupancy_pct" not in df.columns:
        raise ValueError("CSV 需要包含 columns: 'date','occupancy_pct'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ---------- Prophet ----------
def fit_prophet_and_forecast(df: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
    """fit Prophet on occupancy history and forecast next N days"""
    tmp = df.rename(columns={"date":"ds","occupancy_pct":"y"})[["ds","y"]].copy()
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(tmp)
    future = model.make_future_dataframe(periods=periods)
    fcst = model.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
    fcst = fcst.tail(periods).rename(columns={
        "ds":"date",
        "yhat":"occ_pred",
        "yhat_lower":"occ_lo",
        "yhat_upper":"occ_hi",
    })
    # clip 0~100
    for c in ["occ_pred","occ_lo","occ_hi"]:
        fcst[c] = fcst[c].clip(0, 100)
    return fcst

def summarize_forecast(fcst: pd.DataFrame) -> dict:
    return {
        "avg_occ": float(fcst["occ_pred"].mean()),
        "min_occ": float(fcst["occ_pred"].min()),
        "max_occ": float(fcst["occ_pred"].max()),
        "start": fcst["date"].min().date().isoformat(),
        "end": fcst["date"].max().date().isoformat(),
    }

# ---------- XGBoost pricing ----------
def _build_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # derive competitor mean if provided min/max
    if "comp_mean" not in out.columns:
        if "comp_min" in out.columns and "comp_max" in out.columns:
            out["comp_mean"] = (out["comp_min"] + out["comp_max"]) / 2.0
        else:
            out["comp_mean"] = np.nan  # will be imputed at inference
    # date features
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dow"]>=4).astype(int)
    out["month"] = out["date"].dt.month
    return out

def train_xgb_pricing_model(history: pd.DataFrame) -> tuple[Pipeline, float] | None:
    """
    Train an XGB regressor if 'price' exists in history.
    Returns (pipeline, mae) or None if price column missing.
    """
    if "price" not in history.columns:
        return None

    df = _build_pricing_features(history)
    feats = ["occupancy_pct","comp_mean","dow","is_weekend","month"]
    df = df.dropna(subset=["price"])  # require price labels
    X = df[feats]
    y = df["price"].astype(float)

    pre = ColumnTransformer(
        transformers=[("passthrough","passthrough", ["occupancy_pct","comp_mean","dow","is_weekend","month"])]
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    pipe = Pipeline(steps=[("prep", pre), ("model", model)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    return pipe, mae

def infer_price_range(
    model_or_none,
    avg_occ: float,
    comp_min: float | None,
    comp_max: float | None,
    anchor_comp_mean: float | None = None
) -> dict:
    """
    If model available → use it to predict; else use hybrid rule.
    Returns dict with price_mid, lo, hi.
    """
    # derive competitor mean
    comp_mean = anchor_comp_mean
    if comp_mean is None:
        if comp_min is not None and comp_max is not None:
            comp_mean = (comp_min + comp_max) / 2.0

    if model_or_none is not None and comp_mean is not None:
        # ML prediction
        X = pd.DataFrame([{
            "occupancy_pct": avg_occ,
            "comp_mean": comp_mean,
            "dow": 5, "is_weekend": 1, "month": 8  # neutral placeholders for "next week"
        }])
        price_mid = float(model_or_none.predict(X)[0])
        band = max(8.0, price_mid * 0.08)  # ±8% band
        return {"price_mid": price_mid, "lo": price_mid - band, "hi": price_mid + band, "basis": "xgboost"}
    else:
        # rule-of-thumb pricing if no label history
        # base on competitor mean ± occupancy pressure
        base = comp_mean if comp_mean is not None else 150.0
        pressure = 0.6 * (avg_occ/100.0)  # 0 ~ 0.6
        price_mid = base * (0.9 + pressure)  # 90% ~ 150% of comp_mean
        band = max(10.0, price_mid * 0.12)
        return {"price_mid": price_mid, "lo": price_mid - band, "hi": price_mid + band, "basis": "rule"}



# ================== WRAPPERS FOR STREAMLIT ==================

def simple_occupancy_forecast(
    df: pd.DataFrame,
    lookback_days: int | None = None,
    boost: float | None = None,         # streamlit 可能用的命名
    boost_pct: float | None = None,     # 也支援這個命名
    periods: int = 7,
    forecast_days: int | None = None,   # 若 UI 用 forecast_days
    **kwargs,                           # 防未預期參數
):
    """
    回傳 (forecast_df, summary_dict)

    - lookback_days：訓練視窗（只取最近 N 天做 Prophet 訓練）
    - boost / boost_pct：整體百分比調整（+10 → *1.10）
    - periods / forecast_days：預測未來天數（優先使用 forecast_days）
    """
    # 參數對齊
    if forecast_days is not None:
        periods = int(forecast_days)

    if boost_pct is None:
        boost_pct = float(boost or 0.0)
    else:
        boost_pct = float(boost_pct)

    work = df.copy()

    # 只用最近 lookback_days 天
    if lookback_days is not None and lookback_days > 0:
        cutoff = work["date"].max() - pd.Timedelta(days=int(lookback_days) - 1)
        work = work[work["date"] >= cutoff].copy()

    # Prophet 預測
    fcst = fit_prophet_and_forecast(work, periods=periods)

    # 活動/季節加成
    if boost_pct:
        factor = 1.0 + (boost_pct / 100.0)
        for c in ["occ_pred", "occ_lo", "occ_hi"]:
            fcst[c] = (fcst[c] * factor).clip(0, 100)

    summary = summarize_forecast(fcst)
    return fcst, summary


def dynamic_pricing(
    history: pd.DataFrame,
    comp_min: float | None = None,
    comp_max: float | None = None,
    anchor_comp_mean: float | None = None,
) -> dict:
    """
    訓練 XGB（若有 price 標籤），否則用規則法。
    回傳 {price_mid, lo, hi, basis, avg_occ, model_mae}
    """
    trained = train_xgb_pricing_model(history)
    model, mae = (trained if trained is not None else (None, None))

    # 近一週平均入住率（fallback: 全體平均 or 70）
    if "occupancy_pct" in history.columns:
        recent = history.sort_values("date").tail(7)
        avg_occ = float(recent["occupancy_pct"].mean())
        if np.isnan(avg_occ):
            avg_occ = float(history["occupancy_pct"].mean())
    else:
        avg_occ = 70.0

    out = infer_price_range(
        model_or_none=model,
        avg_occ=avg_occ,
        comp_min=comp_min,
        comp_max=comp_max,
        anchor_comp_mean=anchor_comp_mean,
    )
    out["avg_occ"] = avg_occ
    out["model_mae"] = mae
    return out


def dynamic_price_suggestion(
    history: pd.DataFrame,
    comp_min: float | None = None,
    comp_max: float | None = None,
    anchor_comp_mean: float | None = None,
) -> dict:
    """與 dynamic_pricing 相同；提供給 streamlit_app.py 的相容名稱。"""
    return dynamic_pricing(
        history=history,
        comp_min=comp_min,
        comp_max=comp_max,
        anchor_comp_mean=anchor_comp_mean,
    )


# ================== STABLE WRAPPERS ==================

def simple_occupancy_forecast(
    df,
    lookback_days=None,
    boost=None,
    boost_pct=None,
    periods=7,
    forecast_days=None,
    **kwargs,
):
    # 參數對齊
    if forecast_days is not None:
        periods = int(forecast_days)
    if boost_pct is None:
        boost_pct = float(boost or 0.0)
    else:
        boost_pct = float(boost_pct)

    work = df.copy()
    if lookback_days is not None and lookback_days > 0:
        cutoff = work["date"].max() - pd.Timedelta(days=int(lookback_days) - 1)
        work = work[work["date"] >= cutoff].copy()

    fcst = fit_prophet_and_forecast(work, periods=periods)

    if boost_pct:
        factor = 1.0 + (boost_pct / 100.0)
        for c in ["occ_pred", "occ_lo", "occ_hi"]:
            fcst[c] = (fcst[c] * factor).clip(0, 100)

    summary = summarize_forecast(fcst)
    return fcst, summary


def dynamic_pricing(history, comp_min=None, comp_max=None, anchor_comp_mean=None):
    """回傳 dict：{'price_mid','lo','hi','basis','avg_occ','model_mae'}"""
    trained = train_xgb_pricing_model(history)
    model, mae = (trained if trained is not None else (None, None))

    if "occupancy_pct" in history.columns:
        recent = history.sort_values("date").tail(7)
        avg_occ = float(recent["occupancy_pct"].mean())
        if np.isnan(avg_occ):
            avg_occ = float(history["occupancy_pct"].mean())
    else:
        avg_occ = 70.0

    out = infer_price_range(
        model_or_none=model,
        avg_occ=avg_occ,
        comp_min=comp_min,
        comp_max=comp_max,
        anchor_comp_mean=anchor_comp_mean,
    )
    out["avg_occ"] = avg_occ
    out["model_mae"] = mae
    return out


def dynamic_price_suggestion(history, comp_min=None, comp_max=None, anchor_comp_mean=None):
    """
    ⚓ 固定回兩個值，給 streamlit 用：
      ((lo, hi), reason)
    """
    info = dynamic_pricing(
        history=history,
        comp_min=comp_min,
        comp_max=comp_max,
        anchor_comp_mean=anchor_comp_mean,
    )
    lo = float(info["lo"])
    hi = float(info["hi"])
    reason = f"basis={info.get('basis','rule')}, avg_occ={info.get('avg_occ',0):.1f}%"
    return (lo, hi), reason
