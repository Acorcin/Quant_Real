# Phase 1 Setup Guide

---

## Deploy on VPS

### 1. Clone the repo

```bash
git clone https://github.com/Acorcin/quant-eod-engine.git
cd quant-eod-engine
```

### 2. Create your .env file

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys and credentials. See `.env.example` for the required variables.

Lock it down:
```bash
chmod 600 .env
```

### 3. Start Postgres and run

```bash
docker compose up -d postgres
docker compose ps  # wait for healthy
docker compose run --rm quant-engine
```

### 4. Check results

```bash
cat logs/daily_$(date +%Y-%m-%d).log
docker exec -it quant_postgres psql -U postgres -d quant_eod -c "SELECT * FROM pipeline_runs;"
```

### 5. Set up cron (after test succeeds)

```bash
crontab -e
# 4:30 PM EST = 20:30 UTC (EDT) / 21:30 UTC (EST)
30 20 * * 1-5  cd ~/quant-eod-engine && docker compose run --rm quant-engine >> /var/log/quant-cron.log 2>&1
```

---

## Known Gaps

1. **German 2Y Yield** — FRED doesn't carry it. System runs on US 2Y alone.
2. **Economic Calendar** — Placeholder. Perplexity AI partially covers it (once enabled).
3. **OANDA Sentiment** — Endpoint deprecated (403). Fallback to neutral coded.
4. **Perplexity AI** — Add key when ready. Fallback: neutral score, 0.1 confidence.
