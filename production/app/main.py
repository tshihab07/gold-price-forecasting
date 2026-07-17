""""""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import pandas as pd
import json

try:
    from .config import API_HOST, API_PORT, API_DEBUG, DATA_DIR, FINAL_MODEL_INFO, get_best_model_path
    from .utils.logger import get_logger
    from .services.feature_engineer import GoldFeatureEngineer
    from .services.data_ingestion import MarketDataFetcher
    from .models.predictor import GoldPricePredictor

except ImportError:
    from app.config import API_HOST, API_PORT, API_DEBUG, DATA_DIR, FINAL_MODEL_INFO, get_best_model_path
    from app.utils.logger import get_logger
    from app.services.feature_engineer import GoldFeatureEngineer
    from app.services.data_ingestion import MarketDataFetcher
    from app.models.predictor import GoldPricePredictor

logger = get_logger(__name__)

# global instances
feature_engineer: Optional[GoldFeatureEngineer] = None
data_fetcher: Optional[MarketDataFetcher] = None
predictor: Optional[GoldPricePredictor] = None
prediction_history: List[Dict] = []

# jinja2 templates
templates = Jinja2Templates(directory="app/templates")


# pydantic models for request/response
class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Dictionary with all engineered features")
    include_debug: bool = Field(default=False, description="Include debug information in response")


class MarketDataRequest(BaseModel):
    returns: Dict[str, float] = Field(..., description="Dictionary with returns for each asset (SPX_Return, USO_Return, etc.)")
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp")


class PredictionResponse(BaseModel):
    predicted_return: float
    direction: str
    confidence: float
    timestamp: str
    model_version: str
    feature_count: int
    debug_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    predictor_ready: bool
    feature_engineer_ready: bool
    data_fetcher_ready: bool
    prediction_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Starting Gold Price Forecasting API")
    
    global feature_engineer, data_fetcher, predictor
    
    try:
        feature_engineer = GoldFeatureEngineer()
        logger.info("✓ Feature Engineer initialized")
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize Feature Engineer: {e}")
    
    try:
        data_fetcher = MarketDataFetcher()
        logger.info("✓ Data Fetcher initialized")
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize Data Fetcher: {e}")
    
    try:
        # load model dynamically based on FinalModel.csv
        model_path = get_best_model_path()
        predictor = GoldPricePredictor(model_path=model_path)
        logger.info(f"✓ Predictor initialized with model: {model_path.name}")
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize Predictor: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # shutdown
    logger.info("Shutting down Gold Price Forecasting API")
    logger.info(f"Total predictions made: {len(prediction_history)}")


# create app
app = FastAPI(
    title="Gold Price Forecasting API",
    description="Production-grade real-time streaming prediction system for gold price forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def _build_forecast_payload() -> Dict[str, Any]:
    if predictor is None:
        raise RuntimeError("Predictor is not initialized")

    price_path = DATA_DIR / "price_data.csv"
    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")

    df = pd.read_csv(price_path)
    df = df.dropna(subset=["Date", "GLD"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).tail(180).reset_index(drop=True)

    price_series = df["GLD"].astype(float)
    market_returns = df[["SPX", "GLD", "USO", "SLV", "EUR/USD"]].pct_change().dropna()
    market_returns = market_returns.rename(columns={"EUR/USD": "EURUSD"})

    engineer = GoldFeatureEngineer()
    for _, row in market_returns.iterrows():
        engineer.update({
            "SPX_Return": float(row["SPX"]),
            "USO_Return": float(row["USO"]),
            "SLV_Return": float(row["SLV"]),
            "EURUSD_Return": float(row["EURUSD"]),
            "GLD_Return": float(row["GLD"]),
        })

    features_df = engineer.extract_features()
    prediction = predictor.predict(features_df)

    last_price = float(price_series.iloc[-1])
    projected_price = last_price * (1 + prediction["predicted_return"])

    history_prices = [float(value) for value in price_series.tail(60)]
    history_labels = [d.strftime("%b %d") for d in df["Date"].tail(60)]
    chart_svg = _build_trend_chart(history_prices, projected_price, history_labels)

    return {
        "prediction": prediction,
        "direction": prediction["direction"],
        "confidence": prediction["confidence"],
        "predicted_return": prediction["predicted_return"],
        "timestamp": prediction["timestamp"],
        "projected_price": projected_price,
        "history_prices": history_prices,
        "history_labels": history_labels,
        "chart_svg": chart_svg,
        "latest_price": last_price,
        "model_name": FINAL_MODEL_INFO.get("Model", "CatBoost (Optuna)"),
    }


def _build_trend_chart(history_prices: List[float], projected_price: float, labels: List[str]) -> str:
    if not history_prices:
        return ""

    width = 760
    height = 280
    padding = 30
    min_val = min(history_prices + [projected_price]) * 0.98
    max_val = max(history_prices + [projected_price]) * 1.02

    def scale_x(index: int, total: int) -> float:
        return padding + (index / max(total - 1, 1)) * (width - 2 * padding)

    def scale_y(value: float) -> float:
        return height - padding - ((value - min_val) / max(max_val - min_val, 1e-6)) * (height - 2 * padding)

    points = []
    
    for idx, value in enumerate(history_prices):
        points.append((scale_x(idx, len(history_prices)), scale_y(value), value, labels[idx]))

    last_x = scale_x(len(history_prices), len(history_prices) + 1)
    last_y = scale_y(projected_price)
    last_label = labels[-1] + " (proj)"

    # build shadow path
    shadow_points = " ".join(f"{x:.1f},{y + 2:.1f}" for x, y, _, _ in points)
    shadow_points += f" {last_x:.1f},{last_y + 2:.1f}"

    # build main line path
    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
    line_points += f" {last_x:.1f},{last_y:.1f}"

    # build gradient fill area under the line
    fill_points = f"{padding},{height - padding} "
    fill_points += line_points
    fill_points += f" {width - padding},{height - padding}"

    # tooltip markers
    tooltip_data = []
    for x, y, value, label in points:
        tooltip_data.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="transparent" stroke="transparent" data-value="{value:.2f}" data-label="{label}" class="tooltip-trigger" />')
    tooltip_data.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="16" fill="transparent" stroke="transparent" data-value="{projected_price:.2f}" data-label="{last_label}" class="tooltip-trigger" />')

    # visible markers
    marker_points = []
    for x, y, value, _ in points:
        marker_points.append(f"""
            <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--color-primary)" stroke="var(--color-background)" stroke-width="2" class="data-point" data-value="{value:.2f}" />
        """)
    
    marker_points.append(f"""
        <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="6" fill="var(--color-primary)" stroke="var(--color-background)" stroke-width="2" class="data-point projected" data-value="{projected_price:.2f}" />
    """)

    # X-axis labels
    labels_markup = []
    step = max(1, len(labels) // 6)
    for idx in range(0, len(labels), step):
        x = scale_x(idx, len(labels))
        labels_markup.append(f'<text x="{x:.1f}" y="{height - 8}" font-size="11" fill="var(--color-outline-variant)" text-anchor="middle" font-family="var(--font-data-mono)">{labels[idx]}</text>')

    import json

    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="320" role="img" aria-label="Gold price trend chart with interactive tooltips" style="background: var(--color-surface-container-lowest); border-radius: 12px;" data-history-prices='{json.dumps(history_prices)}' data-last-price='{history_prices[-1]}'>
      <defs>
        <!-- Gradient for area fill -->
        <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="var(--color-primary)" stop-opacity="0.12"/>
          <stop offset="100%" stop-color="var(--color-primary)" stop-opacity="0.02"/>
        </linearGradient>
        <!-- Tooltip style -->
        <style>
          <![CDATA[
            .tooltip {{ pointer-events: none; font-family: var(--font-data-mono); font-size: 11px; }}
            .tooltip-bg {{ fill: var(--color-surface-container-high); filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4)); }}
            .tooltip-text {{ fill: var(--color-on-surface); }}
            .tooltip-label {{ fill: var(--color-on-surface-variant); font-family: var(--font-label-caps); font-size: 9px; }}
            .data-point {{ transition: r 0.15s ease; cursor: crosshair; }}
            .data-point:hover {{ r: 8; }}
            .data-point.projected {{ animation: pulse 2s ease-in-out infinite; }}
            @keyframes pulse {{
              0%, 100% {{ opacity: 1; }}
              50% {{ opacity: 0.6; }}
            }}
            .grid-line {{ stroke: var(--color-outline-variant); stroke-width: 0.5; opacity: 0.3; }}
            .y-label {{ font-family: var(--font-data-mono); font-size: 10px; fill: var(--color-outline-variant); }}
          ]]>
        </style>
      </defs>

      <!-- Background -->
      <rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="var(--color-surface-container-lowest)" />

      <!-- Horizontal grid lines -->
      <g class="grid-lines">
        <line x1="{padding}" y1="{padding + (height - 2*padding) * 0.25}" x2="{width - padding}" y2="{padding + (height - 2*padding) * 0.25}" class="grid-line" />
        <line x1="{padding}" y1="{padding + (height - 2*padding) * 0.5}" x2="{width - padding}" y2="{padding + (height - 2*padding) * 0.5}" class="grid-line" />
        <line x1="{padding}" y1="{padding + (height - 2*padding) * 0.75}" x2="{width - padding}" y2="{padding + (height - 2*padding) * 0.75}" class="grid-line" />
      </g>

      <!-- Y-axis labels (price values) -->
      <g class="y-labels">
        <text x="{padding - 8}" y="{padding + 4}" font-size="10" fill="var(--color-outline-variant)" text-anchor="end" class="y-label">${max_val:.0f}</text>
        <text x="{padding - 8}" y="{padding + (height - 2*padding) * 0.25 + 4}" font-size="10" fill="var(--color-outline-variant)" text-anchor="end" class="y-label">${(max_val - (max_val - min_val) * 0.25):.0f}</text>
        <text x="{padding - 8}" y="{padding + (height - 2*padding) * 0.5 + 4}" font-size="10" fill="var(--color-outline-variant)" text-anchor="end" class="y-label">${(max_val - (max_val - min_val) * 0.5):.0f}</text>
        <text x="{padding - 8}" y="{padding + (height - 2*padding) * 0.75 + 4}" font-size="10" fill="var(--color-outline-variant)" text-anchor="end" class="y-label">${(max_val - (max_val - min_val) * 0.75):.0f}</text>
        <text x="{padding - 8}" y="{height - padding + 4}" font-size="10" fill="var(--color-outline-variant)" text-anchor="end" class="y-label">${min_val:.0f}</text>
      </g>

      <!-- Axes -->
      <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="var(--color-outline-variant)" stroke-width="1" />
      <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="var(--color-outline-variant)" stroke-width="1" />

      <!-- Area fill under line -->
      <polygon points="{fill_points}" fill="url(#areaGradient)" />

      <!-- Shadow line (subtle, no glow) -->
      <polyline points="{shadow_points}" fill="none" stroke="var(--color-primary)" stroke-width="8" stroke-linejoin="round" stroke-linecap="round" opacity="0.1" />

      <!-- Main line -->
      <polyline points="{line_points}" fill="none" stroke="var(--color-primary)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />

      <!-- Tooltip triggers (invisible, larger hit area) -->
      <g class="tooltip-triggers">
        {''.join(tooltip_data)}
      </g>

      <!-- Visible data point markers -->
      <g class="markers">
        {''.join(marker_points)}
      </g>

      <!-- X-axis labels -->
      <g class="x-labels">
        {''.join(labels_markup)}
      </g>

      <!-- Title -->
      <text x="{padding}" y="18" font-size="12" fill="var(--color-on-surface-variant)" font-family="var(--font-label-caps)">Historical trend and next-step projection</text>

      <!-- Interactive tooltip (updated via external JS) -->
      <g id="tooltip" class="tooltip" style="display: none;">
        <rect id="tooltip-bg" class="tooltip-bg" x="0" y="0" width="110" height="52" rx="6" />
        <text id="tooltip-label" class="tooltip-label" x="8" y="18"></text>
        <text id="tooltip-value" class="tooltip-text" x="8" y="36"></text>
        <text id="tooltip-price" class="tooltip-text" x="8" y="48" font-weight="600" fill="var(--color-primary)"></text>
      </g>
    </svg>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:

    return HealthResponse(
        status="healthy" if predictor and feature_engineer else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        predictor_ready=predictor is not None,
        feature_engineer_ready=feature_engineer is not None and feature_engineer.is_ready(),
        data_fetcher_ready=data_fetcher is not None,
        prediction_count=len(prediction_history)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_with_features(request: PredictionRequest) -> PredictionResponse:

    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        import pandas as pd
        
        # convert features dict to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # make prediction
        result = predictor.predict(features_df)
        
        # store in history
        prediction_history.append({
            **result,
            "source": "direct"
        })
        
        # prepare response
        response = PredictionResponse(
            predicted_return=result["predicted_return"],
            direction=result["direction"],
            confidence=result["confidence"],
            timestamp=result["timestamp"],
            model_version=result["model_version"],
            feature_count=result["feature_count"],
            debug_info={"features_received": len(request.features)} if request.include_debug else None
        )
        
        logger.info(f"Prediction endpoint: {response.direction} ({response.confidence:.4f})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/live", response_model=PredictionResponse)
async def predict_with_live_data(request: MarketDataRequest) -> PredictionResponse:
    if not feature_engineer or not predictor:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Parse timestamp if provided
        timestamp = None
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp)
        
        # Update feature engineer
        feature_engineer.update(request.returns, timestamp)
        
        if not feature_engineer.is_ready():
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data. Current history: {len(feature_engineer.spx_returns)}, Required: 10"
            )
        
        # Extract features
        features_df = feature_engineer.extract_features()
        
        # Make prediction
        result = predictor.predict(features_df)
        
        # Store in history
        prediction_history.append({
            **result,
            "source": "live"
        })
        
        # Prepare response
        response = PredictionResponse(
            predicted_return=result["predicted_return"],
            direction=result["direction"],
            confidence=result["confidence"],
            timestamp=result["timestamp"],
            model_version=result["model_version"],
            feature_count=result["feature_count"],
        )
        
        logger.info(f"Live prediction: {response.direction} ({response.confidence:.4f})")
        return response
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Live prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
async def get_current_features() -> Dict[str, Any]:
    if not feature_engineer:
        raise HTTPException(status_code=503, detail="Feature Engineer not initialized")
    
    if not feature_engineer.is_ready():
        raise HTTPException(
            status_code=400,
            detail="Not enough historical data to extract features"
        )
    
    try:
        features_df = feature_engineer.extract_features()
        history = feature_engineer.get_history()
        
        return {
            "current_features": features_df.iloc[0].to_dict(),
            "history": history,
            "ready": feature_engineer.is_ready(),
            "history_size": len(feature_engineer.spx_returns)
        }
    
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history")
async def get_prediction_history(limit: int = 100) -> List[Dict]:
    return prediction_history[-limit:]


@app.get("/stats")
async def get_statistics() -> Dict[str, Any]:
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    predictor_stats = predictor.get_stats()
    
    return {
        "predictor": predictor_stats,
        "prediction_history_size": len(prediction_history),
        "timestamp": datetime.utcnow().isoformat(),
        "feature_engineer_ready": feature_engineer is not None and feature_engineer.is_ready(),
        "final_model_info": FINAL_MODEL_INFO,
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Render the professional landing page for the forecasting platform."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/forecast", response_class=HTMLResponse)
async def forecasting_page(request: Request) -> HTMLResponse:
    """Render the forecasting page with a chart and model output."""
    try:
        payload = _build_forecast_payload()
        return templates.TemplateResponse("forecast.html", {"request": request, **payload})
    except Exception as exc:
        logger.error(f"Forecast page error: {exc}")
        return HTMLResponse(content=f"<html><body><h1>Forecasting unavailable</h1><p>{exc}</p></body></html>", status_code=500)


@app.get("/favicon.ico")
async def favicon() -> HTMLResponse:
    """Serve a lightweight favicon to avoid repeated missing-favicon requests."""
    return HTMLResponse(content="<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='12' fill='#0d0e12'/><path d='M18 44c8-10 20-20 28-24' stroke='#e9c349' stroke-width='5' fill='none' stroke-linecap='round'/><path d='M18 32c6 2 12 2 18 0' stroke='#d4af37' stroke-width='5' fill='none' stroke-linecap='round'/></svg>", media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )