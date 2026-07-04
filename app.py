import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

app = FastAPI(title="Nifty 500 Production Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    requested_date: str

BACKUP_FILE = "nifty_universe_backup.csv"

def load_local_nifty_universe() -> list[str]:
    if os.path.exists(BACKUP_FILE):
        return pd.read_csv(BACKUP_FILE)['Ticker'].tolist()
    raise FileNotFoundError("nifty_universe_backup.csv missing from root directory.")

def sanitize_value(val) -> float:
    if val is None or np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)

@app.post("/api/analyze-all")
async def analyze_all_stocks(request: PredictionRequest):
    try:
        target_date = datetime.strptime(request.requested_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Use YYYY-MM-DD format.")
    
    # Loads the full list of 500 stocks from your local CSV file
    tickers = load_local_nifty_universe()
    
    # Set the historical data window boundaries
    start_date = (target_date - timedelta(days=90)).strftime('%Y-%m-%d')
    # Extend end_date to the current day to ensure we get the live real-time price row
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Downloading bulk market data for all {len(tickers)} stocks...")
    try:
        raw_data = yf.download(
            tickers=tickers, 
            start=start_date, 
            end=end_date, 
            group_by='ticker', 
            progress=False, 
            timeout=35
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data Stream Error: {str(e)}")
        
    if raw_data.empty:
        raise HTTPException(status_code=500, detail="No data returned from the market interface.")

    results = []
    
    for ticker in tickers:
        try:
            if ticker not in raw_data.columns.levels[0]:
                continue
                
            # Drop rows missing closing prices
            df = raw_data[ticker].dropna(subset=['Close'])
            if df.empty:
                continue
                
            # 1. Extract the Live Current Price (the most recent row in the entire dataset)
            live_current_price = float(df['Close'].iloc[-1])
            
            # 2. Filter the data to isolate rows up to the user's selected date
            target_df = df[df.index <= target_date]
            if len(target_df) < 20:
                continue
            
            # Extract rows for the selected date and the previous business day
            target_row = target_df.iloc[-1]
            prev_row = target_df.iloc[-2]
            actual_eval_date = target_df.index[-1]
            
            selected_day_close = float(target_row['Close'])
            prev_day_close = float(prev_row['Close'])
            
            # Technical Indicators
            close_series = target_df['Close']
            sma5 = close_series.rolling(window=5).mean().iloc[-1]
            sma20 = close_series.rolling(window=20).mean().iloc[-1]
            
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            log_ret = np.log(close_series / close_series.shift(1))
            volatility = log_ret.rolling(window=10).std().iloc[-1]
            if np.isnan(volatility): 
                volatility = 0.02
                
            if pd.isna(sma5) or pd.isna(sma20) or sma20 == 0:
                continue
                
            # Trend Calculations
            sma_ratio = sma5 / sma20
            rsi_factor = (rsi / 100) if rsi < 75 else (1.5 - (rsi / 100))
            predicted_profit_score = ((sma_ratio - 1) * 50) + (rsi_factor * 5)
            
            momentum_direction = (sma_ratio - 1)
            predicted_change_pct = (momentum_direction * 100) * (1.0 + (rsi / 50))
            max_bound = max(float(volatility * 100 * 1.5), 2.5)
            predicted_change_pct = np.clip(predicted_change_pct, -max_bound, max_bound)
            
            # Historical daily change on the selected day
            daily_change_pct = ((selected_day_close - prev_day_close) / prev_day_close) * 100
            
            # Confidence Metrics
            volatility_penalty = (volatility * 100) * 12.0 
            rsi_instability_penalty = abs(50 - rsi) * 0.3 if abs(50 - rsi) < 10 else 0
            confidence_score = np.clip(90.0 - volatility_penalty - rsi_instability_penalty, 10.0, 98.0)
            
            results.append({
                "ticker": ticker.replace(".NS", ""),
                "previous_close": sanitize_value(prev_day_close),
                "selected_close": sanitize_value(selected_day_close),
                "current_price": sanitize_value(live_current_price),
                "daily_change_pct": sanitize_value(daily_change_pct),
                "rsi": sanitize_value(rsi),
                "predicted_profit_score": sanitize_value(predicted_profit_score),
                "predicted_change_pct": sanitize_value(predicted_change_pct),
                "confidence_score": sanitize_value(confidence_score),
                "actual_date_used": actual_eval_date.strftime('%Y-%m-%d')
            })
        except Exception:
            continue

    print(f"Processing complete. Displaying all {len(results)} valid stocks.")
    return {"status": "success", "total_processed": len(results), "data": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)