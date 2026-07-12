"""
Fetcher/Notifier: Discord Webhook — Signal Delivery.

Sends formatted trading signals and pipeline status updates
to a Discord channel (or DM-forwarded webhook) via webhook.

Replaces the Telegram bot approach with a simpler, credential-light
Discord webhook that requires no bot token — just a webhook URL.
"""
import requests
import json
import logging
from datetime import datetime, timezone, date
from config.settings import DISCORD_WEBHOOK_URL

logger = logging.getLogger(__name__)


def send_signal(snapshot: dict, status: str) -> bool:
    """
    Send the daily signal summary to Discord.

    Args:
        snapshot: The assembled daily snapshot dict.
        status: Pipeline run status ('success', 'partial', 'failed').

    Returns:
        True if message sent successfully, False otherwise.
    """
    if not DISCORD_WEBHOOK_URL:
        logger.warning("No Discord webhook URL configured. Skipping notification.")
        return False

    try:
        embed = _build_embed(snapshot, status)
        payload = {
            "username": "Quant EOD Engine",
            "embeds": [embed],
        }

        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        logger.info("Discord signal sent successfully")
        return True

    except requests.exceptions.HTTPError as e:
        logger.error(f"Discord webhook HTTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"Discord webhook error: {e}")
        return False


def _build_embed(snapshot: dict, status: str) -> dict:
    """Build a rich Discord embed from the daily snapshot."""
    today = snapshot.get("date", str(date.today()))
    ai = snapshot.get("ai_sentiment", {})
    macro = snapshot.get("macro", {})
    bars = snapshot.get("bars_summary", {})
    sentiment = snapshot.get("sentiment", {})
    swaps = snapshot.get("swap_rates", {})

    # Status emoji
    status_map = {
        "success": "\u2705",   # ✅
        "partial": "\u26a0\ufe0f",   # ⚠️
        "failed": "\u274c",    # ❌
    }
    status_icon = status_map.get(status, "\u2753")

    # AI sentiment formatting
    ai_score = ai.get("macro_sentiment_score", 0.0)
    ai_confidence = ai.get("confidence", 0.0)
    ai_driver = ai.get("dominant_driver", "n/a")
    ai_rationale = ai.get("rationale", "No AI analysis available.")
    fed_stance = ai.get("fed_stance", "n/a")
    ecb_stance = ai.get("ecb_stance", "n/a")
    risk = ai.get("risk_sentiment", "n/a")
    fallback = ai.get("fallback_used", True)

    # Directional arrow for sentiment score
    if ai_score > 0.3:
        direction = "\U0001f7e2 BULLISH EUR/USD"     # 🟢
    elif ai_score < -0.3:
        direction = "\U0001f534 BEARISH EUR/USD"     # 🔴
    else:
        direction = "\U0001f7e1 NEUTRAL"              # 🟡

    # Key events
    events = ai.get("key_events", [])
    events_text = "\n".join(f"• {e}" for e in events[:5]) if events else "None reported"

    # Yield spread
    us_2y = macro.get("us_2y_yield", "n/a")
    spread = macro.get("yield_spread_bps", "n/a")

    # Bar counts
    bar_lines = []
    for key, val in bars.items():
        if not key.endswith("_error"):
            bar_lines.append(f"`{key}`: {val} bars")

    # Phase 2 data
    regime = snapshot.get("regime", {})
    regime_label = regime.get("state_label", "unknown")
    regime_conf = regime.get("confidence", 0)
    regime_days = regime.get("days_in_regime", 0)

    prediction = snapshot.get("prediction", {})
    pred_dir = prediction.get("direction", "flat")
    pred_prob = prediction.get("probability", 0.5)
    pred_size = prediction.get("size_multiplier", 0)

    signals_data = snapshot.get("signals", {})
    composite = signals_data.get("composite", {})
    comp_dir = composite.get("composite_direction", "flat")
    comp_strength = composite.get("composite_strength", 0)

    # Prediction direction emoji
    if pred_dir == "long":
        pred_icon = "\U0001f7e2 LONG"   # Green
    elif pred_dir == "short":
        pred_icon = "\U0001f534 SHORT"  # Red
    else:
        pred_icon = "\u26aa FLAT"

    # Regime emoji
    regime_icons = {
        "low_vol": "\U0001f4c8",
        "high_vol_choppy": "\U0001f300",
        "high_vol_crash": "\u26a1",
    }
    regime_icon = regime_icons.get(regime_label, "\u2753")

    # Build embed
    embed = {
        "title": f"{status_icon} EOD Signal \u2014 {today}",
        "color": 0x00ff88 if status == "success" else (0xffaa00 if status == "partial" else 0xff4444),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "Quant EOD Engine \u2022 Phase 2"},
        "fields": [
            {
                "name": "\U0001f3af Meta-Model Prediction",
                "value": (
                    f"**{pred_icon} EUR/USD**\n"
                    f"Probability: `{pred_prob:.1%}` | Size: `{pred_size}x`\n"
                    f"Composite: `{comp_dir}` (strength `{comp_strength:.3f}`)"
                ),
                "inline": False,
            },
            {
                "name": f"{regime_icon} Regime",
                "value": f"`{regime_label}` (conf `{regime_conf:.3f}`)\nDays in regime: `{regime_days}`",
                "inline": True,
            },
            {
                "name": "\U0001f3af Macro Sentiment",
                "value": (
                    f"**{direction}**\n"
                    f"Score: `{ai_score:.3f}` | Confidence: `{ai_confidence:.3f}`\n"
                    f"Driver: `{ai_driver}`"
                    + (f"\n\u26a0\ufe0f *Fallback \u2014 AI unavailable*" if fallback else "")
                ),
                "inline": True,
            },
            {
                "name": "\U0001f3e6 Central Banks",
                "value": f"Fed: `{fed_stance}` | ECB: `{ecb_stance}`\nRisk: `{risk}`",
                "inline": True,
            },
            {
                "name": "\U0001f4c8 Yields",
                "value": f"US 2Y: `{us_2y}`\nSpread: `{spread}` bps",
                "inline": True,
            },
            {
                "name": "\U0001f4f0 Key Events",
                "value": events_text,
                "inline": False,
            },
            {
                "name": "\U0001f6e0\ufe0f Pipeline",
                "value": f"Status: **{status.upper()}**\nFriday: {'Yes' if snapshot.get('is_friday') else 'No'}",
                "inline": True,
            },
        ],
    }

    return embed


def send_error_alert(error_msg: str) -> bool:
    """Send a critical error alert to Discord."""
    if not DISCORD_WEBHOOK_URL:
        return False

    try:
        payload = {
            "username": "Quant EOD Engine",
            "embeds": [{
                "title": "\u274c Pipeline Critical Error",
                "description": f"```\n{error_msg[:1800]}\n```",
                "color": 0xff0000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "Quant EOD Engine • Alert"},
            }],
        }
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Discord error alert: {e}")
        return False
