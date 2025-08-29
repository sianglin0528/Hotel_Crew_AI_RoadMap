from data_utils import (
    load_occupancy_csv,
    fit_prophet_and_forecast, summarize_forecast,     # 可能成功（Prophet）
    train_xgb_pricing_model, infer_price_range        # XGBoost 或規則定價
)

# 可改
CSV_PATH = "sample_data/occupancy_history.csv"
COMP_MIN, COMP_MAX = 120.0, 180.0
PERIODS = 7  # 預測未來天數

def try_forecast(hist):
    """先 Prophet，失敗則 Holt-Winters（需在 data_utils.py 中已加入 fit_hw_and_forecast）。"""
    try:
        fcst = fit_prophet_and_forecast(hist, periods=PERIODS)
        fsum = summarize_forecast(fcst)
        src = "Prophet"
        return fsum, src
    except Exception:
        # 改用 Holt-Winters（在 data_utils.py 裡實作 fit_hw_and_forecast + summarize_forecast_generic）
        from data_utils import fit_hw_and_forecast, summarize_forecast_generic
        fcst = fit_hw_and_forecast(hist, periods=PERIODS)
        fsum = summarize_forecast_generic(fcst)
        src = "Holt-Winters"
        return fsum, src

def main():
    hist = load_occupancy_csv(CSV_PATH)

    # 1) 住房率預測（真模型）
    fsum, occ_src = try_forecast(hist)
    avg_occ = round(fsum["avg_occ"], 1)

    # 2) 定價（有 price → XGBoost，沒有 → 規則）
    xgb_tuple = train_xgb_pricing_model(hist)  # 需 CSV 有 'price' 才會生效，否則 None
    price = infer_price_range(xgb_tuple, avg_occ=avg_occ, comp_min=COMP_MIN, comp_max=COMP_MAX)

        # 3) 輸出摘要
    print("\n=== OCCUPANCY FORECAST ===")
    print(f"source     : {occ_src}")
    print(f"window     : {fsum['start']} ~ {fsum['end']}")
    print(f"avg/min/max: {avg_occ}% / {fsum['min_occ']:.1f}% / {fsum['max_occ']:.1f}%")

    print("\n=== PRICING SUGGESTION ===")
    print(f"comp range : USD {COMP_MIN:.0f} ~ {COMP_MAX:.0f}")
    print(f"price mid  : USD {price['price_mid']:.1f}")
    print(f"price band : USD {price['lo']:.1f} ~ {price['hi']:.1f}")
    print(f"method     : {price['basis']}")  # 'xgboost_mae_...' 或 'rule'

if __name__ == "__main__":
    main()