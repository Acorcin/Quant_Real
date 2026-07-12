"""
Database-backed Model Persistence.
Implements write-through caching to a local directory so ephemeral dynos
don't hit scaling limits while allowing models to be persistent.
"""

import os
import io
import joblib
import logging
from typing import Any
from models.database import get_connection, fetch_one
from config.settings import MODELDIR

logger = logging.getLogger(__name__)

def save_model_to_db(model_name: str, version: str, obj: Any):
    """Serialize a Python object using joblib into a BytesIO buffer and upsert to Postgres bytea."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    raw_bytes = buffer.getvalue()
    
    # Upsert to PostgreSQL
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_artifacts (model_name, version, artifact, created_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (model_name) DO UPDATE SET
                    version = EXCLUDED.version,
                    artifact = EXCLUDED.artifact,
                    created_at = EXCLUDED.created_at
            """, (model_name, version, raw_bytes))
        conn.commit()
        logger.info(f"Model '{model_name}' (version {version}) saved to PostgreSQL model_artifacts.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save model '{model_name}' to DB: {e}")
        raise
    finally:
        conn.close()

    # Write-through to local cache
    local_path = os.path.join(MODELDIR, f"{model_name}.joblib")
    try:
        with open(local_path, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        logger.warning(f"Failed to cache model '{model_name}' locally to {local_path}: {e}")

def load_model_from_db(model_name: str) -> Any:
    """Load object, preferring local cache within dyno session, otherwise fetch from DB and cache locally."""
    local_path = os.path.join(MODELDIR, f"{model_name}.joblib")
    
    # Check local cache first
    if os.path.exists(local_path):
        try:
            logger.info(f"Loading '{model_name}' from local cache {local_path}.")
            return joblib.load(local_path)
        except Exception as e:
            logger.warning(f"Failed to load '{model_name}' from local cache: {e}. Falling back to DB.")

    # Fetch from PostgreSQL
    row = fetch_one("SELECT version, artifact FROM model_artifacts WHERE model_name = %s", (model_name,))
    if not row or not row.get("artifact"):
        logger.warning(f"No model artifact found in DB for '{model_name}'.")
        return None
        
    raw_bytes = row["artifact"]
    # Handle psycopg2 memoryview
    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
        
    logger.info(f"Loaded '{model_name}' (version {row['version']}) from DB, caching locally.")
    
    # Write to local cache
    try:
        with open(local_path, "wb") as f:
            f.write(raw_bytes)
    except Exception as e:
        logger.warning(f"Failed to write DB artifact for '{model_name}' to local cache: {e}")

    # Deserialize
    buffer = io.BytesIO(raw_bytes)
    return joblib.load(buffer)
