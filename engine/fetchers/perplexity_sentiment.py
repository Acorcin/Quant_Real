"""
Fetcher: Perplexity Sonar API — AI Macro Sentiment Scoring.

Calls Perplexity's structured output API to generate a quantified
macro sentiment score from the last 24 hours of financial news.

This is the system's biggest informational edge — no other retail
quant system reads the internet daily via LLM and converts it to
a structured feature for an ML model.
"""
import requests
import json
import logging
from datetime import date
from config.settings import PERPLEXITY_API_KEY, PERPLEXITY_MODEL, PERPLEXITY_BASE_URL
from models.database import get_connection

logger = logging.getLogger(__name__)

# ─── Prompt Template ──────────────────────────────────────
DAILY_PROMPT = """
Search financial news from the last 24 hours about EUR/USD, the Federal Reserve,
the ECB, Eurozone economic data, and US economic data.

Analyze:
1. Any central bank commentary or rate decision signals from the last 24 hours
2. Economic data releases today and their surprises vs expectations
3. Geopolitical events affecting EUR or USD
4. Market positioning shifts or notable institutional flows mentioned in news

Based on this analysis, provide your structured assessment of the EUR/USD outlook
for the next 24 hours.
"""

# ─── Response Schema ──────────────────────────────────────
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "macro_sentiment",
        "schema": {
            "type": "object",
            "properties": {
                "macro_sentiment_score": {
                    "type": "number",
                    "description": "Score from -1.0 (extremely bearish EUR/USD, bullish USD) to 1.0 (extremely bullish EUR/USD, bearish USD). 0.0 = neutral."
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the score from 0.0 to 1.0. Lower if news is mixed or few events occurred."
                },
                "dominant_driver": {
                    "type": "string",
                    "description": "Primary driver of the score. One of: fed_hawkish, fed_dovish, ecb_hawkish, ecb_dovish, economic_data_usd_strong, economic_data_usd_weak, economic_data_eur_strong, economic_data_eur_weak, geopolitical, risk_sentiment, technical_flow, no_major_driver"
                },
                "key_events": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3 events or news items driving the score today."
                },
                "rationale": {
                    "type": "string",
                    "description": "2-3 sentence explanation of the score and key reasoning."
                },
                "fed_stance": {
                    "type": "string",
                    "enum": ["hawkish", "neutral", "dovish"],
                    "description": "Current perceived Federal Reserve policy stance based on today's news."
                },
                "ecb_stance": {
                    "type": "string",
                    "enum": ["hawkish", "neutral", "dovish"],
                    "description": "Current perceived ECB policy stance based on today's news."
                },
                "risk_sentiment": {
                    "type": "string",
                    "enum": ["risk_on", "neutral", "risk_off"],
                    "description": "Overall market risk sentiment today."
                },
            },
            "required": [
                "macro_sentiment_score", "confidence", "dominant_driver",
                "key_events", "rationale", "fed_stance", "ecb_stance", "risk_sentiment"
            ],
            "additionalProperties": False,
        }
    }
}


def fetch_ai_sentiment() -> dict:
    """
    Call Perplexity Sonar API with structured output to get macro sentiment.

    Returns parsed sentiment dict or fallback on failure.
    """
    if not PERPLEXITY_API_KEY:
        logger.warning("No Perplexity API key configured. Using fallback.")
        return _fallback_sentiment("no_api_key")

    try:
        response = requests.post(
            f"{PERPLEXITY_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": PERPLEXITY_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior Forex macro analyst. Provide structured analysis of EUR/USD based on today's financial news."
                    },
                    {
                        "role": "user",
                        "content": DAILY_PROMPT,
                    }
                ],
                "response_format": RESPONSE_SCHEMA,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Extract the structured content
        content = data["choices"][0]["message"]["content"]
        sentiment = json.loads(content)

        # Add metadata
        sentiment["model_used"] = PERPLEXITY_MODEL
        sentiment["fallback_used"] = False
        sentiment["date"] = str(date.today())
        sentiment["sources_consulted"] = len(data.get("citations", []))
        sentiment["raw_response"] = data

        logger.info(
            f"AI Sentiment: score={sentiment['macro_sentiment_score']}, "
            f"confidence={sentiment['confidence']}, "
            f"driver={sentiment['dominant_driver']}"
        )
        return sentiment

    except requests.exceptions.HTTPError as e:
        logger.error(f"Perplexity API HTTP error: {e}")
        return _fallback_sentiment(f"http_error_{e.response.status_code}")
    except Exception as e:
        logger.error(f"Perplexity API error: {e}")
        return _fallback_sentiment(str(e))


def _fallback_sentiment(reason: str) -> dict:
    """
    Fallback when Perplexity API is unavailable.
    Returns a neutral score with low confidence.
    """
    logger.info(f"Using fallback sentiment (reason: {reason})")
    return {
        "date": str(date.today()),
        "macro_sentiment_score": 0.0,
        "confidence": 0.1,
        "dominant_driver": "no_major_driver",
        "key_events": [f"AI sentiment unavailable: {reason}"],
        "rationale": "Perplexity API was unavailable. Score set to neutral with low confidence. System should rely more heavily on quantitative signals today.",
        "fed_stance": "neutral",
        "ecb_stance": "neutral",
        "risk_sentiment": "neutral",
        "sources_consulted": 0,
        "model_used": "fallback",
        "fallback_used": True,
        "raw_response": None,
    }


def store_sentiment(data: dict):
    """Store AI sentiment in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ai_sentiment
                    (date, macro_sentiment_score, confidence, dominant_driver,
                     key_events, rationale, fed_stance, ecb_stance, risk_sentiment,
                     sources_consulted, model_used, fallback_used, raw_response)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date)
                DO UPDATE SET
                    macro_sentiment_score = EXCLUDED.macro_sentiment_score,
                    confidence = EXCLUDED.confidence,
                    dominant_driver = EXCLUDED.dominant_driver,
                    key_events = EXCLUDED.key_events,
                    rationale = EXCLUDED.rationale,
                    fed_stance = EXCLUDED.fed_stance,
                    ecb_stance = EXCLUDED.ecb_stance,
                    risk_sentiment = EXCLUDED.risk_sentiment,
                    sources_consulted = EXCLUDED.sources_consulted,
                    model_used = EXCLUDED.model_used,
                    fallback_used = EXCLUDED.fallback_used,
                    raw_response = EXCLUDED.raw_response,
                    fetched_at = NOW()
            """, (
                data["date"],
                data["macro_sentiment_score"],
                data["confidence"],
                data["dominant_driver"],
                json.dumps(data.get("key_events", [])),
                data["rationale"],
                data["fed_stance"],
                data["ecb_stance"],
                data["risk_sentiment"],
                data.get("sources_consulted", 0),
                data["model_used"],
                data["fallback_used"],
                json.dumps(data.get("raw_response")),
            ))
        conn.commit()
        logger.info(f"Stored AI sentiment for {data['date']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing AI sentiment: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store() -> dict:
    """Fetch AI sentiment and store in DB."""
    data = fetch_ai_sentiment()
    store_sentiment(data)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store()
    print(json.dumps(result, indent=2, default=str))
