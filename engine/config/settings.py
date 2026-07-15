"""
Quant EOD Engine — Configuration
All secrets loaded from environment variables (.env file).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (Docker passes env vars directly)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# ─── OANDA ────────────────────────────────────────────────
OANDA_API_TOKEN = os.environ.get("OANDA_API_TOKEN", "")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_BASE_URL = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
# Practice: https://api-fxpractice.oanda.com
# Live:     https://api-fxtrade.oanda.com

# ─── FRED ─────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ─── Perplexity ───────────────────────────────────────────
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_MODEL = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# ─── Discord ──────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ─── PostgreSQL ───────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "quant_eod")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ─── Trading Instruments ──────────────────────────────────
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
PRIMARY_INSTRUMENT = "EUR_USD"

# ─── Kronos bridge ────────────────────────────────────────
# The md-schema series that feeds the engine's foundation-model features.
# During the FX→CME transition this differs from PRIMARY_INSTRUMENT (which is
# still the FX symbol), so it is configured separately. Defaults to the CME
# micro-EUR front month the forecasting pipeline has been persisting.
KRONOS_INSTRUMENT = os.environ.get("KRONOS_INSTRUMENT", "M6EH6")

# L3 microstructure series: the intraday puller's PARENT symbol (md.v_l3_latest
# is keyed by it, e.g. "M6E.FUT" / "ES.FUT") — not the md series name above.
L3_INSTRUMENT = os.environ.get("L3_INSTRUMENT", "M6E.FUT")

# ─── FRED Series IDs ─────────────────────────────────────
FRED_US_2Y_SERIES = "DGS2"          # US 2-Year Treasury yield
FRED_DE_2Y_SERIES = "IRLTLT01DEM156N" # Proxy — using Long-Term Interest Rate for Germany instead of missing 2Y
# Note: FRED does not carry German 2Y directly. Options:
#   1. Use "IRLTLT01DEM156N" (long-term) as proxy (Current)
#   2. Use ECB Statistical Data Warehouse API
#   3. Use Trading Economics API
#   4. Hardcode a manual feed initially
# For MVP, we'll use what FRED has and document the gap.

# ─── Logging ──────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ─── Model Storage ──────────────────────────────────────────
MODELDIR = os.environ.get("MODELDIR", "/tmp/modelartifacts")
os.makedirs(MODELDIR, exist_ok=True)

# ─── Signal Calibration ───────────────────────────────────
# Retail sentiment fade thresholds (fraction long, 0..1)
SENTIMENT_EXTREME_HIGH = float(os.environ.get("SENTIMENT_EXTREME_HIGH", "0.72"))
SENTIMENT_EXTREME_LOW = float(os.environ.get("SENTIMENT_EXTREME_LOW", "0.28"))
