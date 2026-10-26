# Docker Desktop 全自動部署（tidemark）

目標：別台 Docker **開機即跑** — 盤中 Paper 模擬下單、每晚覆盤、網頁看訊號／策略／下單。

## 一次設定

1. 安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Linux containers）
2. 拷貝本 repo（建議一併拷 `data/cache`）
3. `copy .env.example .env`，填入：
   - `SHIOAJI_SIM_API_KEY`
   - `SHIOAJI_SIM_SECRET_KEY`
4. （可選）憑證放 `certs/`，`.env`：`SHIOAJI_CERT_PATH=/app/certs/xxx.pfx`
5. 排程在 `config/ops.yaml`（預設 08:40–13:50 盤中、18:00 覆盤）

## 啟動（全自動）

```bash
docker compose up -d --build
```

| 服務 | 作用 | 重啟 |
|------|------|------|
| `ops` | 日循環 daemon：盤中訂閱 Paper → 晚間覆盤 | unless-stopped |
| `ui` | 戰情頁 http://localhost:8080 | unless-stopped |

```bash
docker compose logs -f ops
docker compose logs -f ui
docker compose down          # 停止
```

Docker Desktop 設成開機啟動後，這兩個 container 會跟 `restart: unless-stopped` 一起回來。

## 網頁

開啟 **http://localhost:8080**

- **訊號** — ComplementarySignal（orb/rsi/flat）
- **策略** — ComplementaryRiskStrategy（一口／風控）
- **下單** — Paper 模擬成交事件（非 LIVE）
- **journal** — `outputs/ops/journal.jsonl` 系統記錄

API：`/api/status` `/api/signal` `/api/strategy` `/api/orders` `/api/journal`

## 記錄落盤

| 路徑 | 內容 |
|------|------|
| `outputs/ops/status.json` | 目前 phase |
| `outputs/ops/journal.jsonl` | 追加事件 |
| `outputs/ops/days/YYYYMMDD.json` | 當日訊號／策略／下單／覆盤 |
| `outputs/ops/evening_latest.json` | 最近一次晚間覆盤 |
| `outputs/live_tick_*.json` | 盤中即時結果 |

## 行為時間表（Asia/Taipei）

| 時間 | 動作 |
|------|------|
| 08:40 | 啟動 tick 訂閱（需 SIM key）；Paper 模擬 |
| 13:50 | 停止盤中、封存當日 |
| 18:00 | 用 cache 1m 重播覆盤（無 cache 則 sim_day） |

`place_sim: false`（預設）— 只 Paper。要對永豐 SIM 真下單：`config/ops.yaml` 設 `place_sim: true` 或環境變數 `OPS_PLACE_SIM=1`（仍非 LIVE）。

## 工具指令

```bash
docker compose --profile tools run --rm smoke
DATE=2026-10-16 docker compose --profile tools run --rm live-tick-replay
# 本機測覆盤：
python scripts/run_ops_daemon.py --once-evening --date 2026-10-16
```

## 掛載

| 主機 | 容器 |
|------|------|
| `./data` | `/app/data` |
| `./outputs` | `/app/outputs` |
| `./config` | `/app/config`（唯讀） |
| `./certs` | `/app/certs` |
| `.env` | env_file |

## 注意

- 禁止 LIVE；UI 唯讀、不下單。
- 無 SIM key 時盤中訂閱會略過，**晚間覆盤仍會跑**（需 `data/cache`）。
- image 不含 `.env`／cache／outputs。
