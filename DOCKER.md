# Running the Quant_Real stack in containers

One image, five services. Postgres reuses the existing `engine_pgdata`
volume, so all md history (bars, forecasts, gate decisions, L3 state)
carries straight over.

## First run

```bash
cd Quant_Real
cp .env.example .env          # fill in DATABENTO_API_KEY + TRADOVATE_* (or keep
                              # the generated .env if it's already there)

# the old engine compose stack must be stopped once — it owns port 5433 and
# runs deprecated FX-era services (physics-engine / live-trader):
docker compose -f engine/docker-compose.yml down

docker compose up -d          # postgres + bootstrap + the three data feeds
docker compose logs -f live_loop
```

`bootstrap` applies every schema idempotently and, on a fresh volume, seeds
md from `data/<INSTRUMENT>_trades.parquet` if present.

## Services

| service | what it does | cadence |
|---|---|---|
| `postgres` | the md + engine database (host port 5433) | — |
| `bootstrap` | schemas + optional seed, then exits | once |
| `live_loop` | volume bars → conditioning → one Kronos forecast per bar | 60 s |
| `l3_puller` | MBP-1 book imbalance + VPIN buckets | 300 s |
| `physics_feed` | per-tick spike-filter + Kalman conditioning | 60 s |
| `demo_loop` | meta-model → veto/sizing → Tradovate **demo** orders | 60 s, **opt-in** |

The trading loop is behind a profile so plain `up` never trades:

```bash
docker compose --profile trade up -d
```

## Controls

```bash
docker compose ps                      # status
docker compose logs -f physics_feed    # any service's log
docker compose down                    # stop everything (data survives)
type nul > data\KILL                   # kill switch: demo_loop flattens + halts
```

## Rules of the road

* **Containers and host loops are mutually exclusive.** The single-instance
  locks cannot see across the PID namespace, so stop host-launched loops
  (`live_loop` / `l3_puller` / `physics_feed` via `Get-Process python`)
  before `docker compose up`, and vice versa.
* Secrets live only in `.env` (git-ignored). The demo rail is unchanged:
  orders require `TRADOVATE_ENV=demo`; live additionally requires editing
  `ALLOW_LIVE` in `engine/execution/tradovate_client.py`.
* Kronos weights download once into the `hf_cache` volume (~100 MB) on the
  first `live_loop` start; later restarts reuse it.
* Every Databento pull is still cost-quoted first and aborts if not ~$0.
