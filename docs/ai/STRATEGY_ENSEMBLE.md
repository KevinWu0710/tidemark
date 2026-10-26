# tidemark 雙軌策略設定與回測說明

> 專案：`tidemark`（潮線）  
> 標的：微型台指（TMF，1點 = 10元）  
> 資金規範：本金 10,000 元，單一倉位上限 `max_lots: 1`  
> 設定檔對應：`config/settings.yaml`

---

## 一、 核心哲學：為什麼不能「每天盲目進場」？

在期貨量化交易中，「天天有單做」往往是新手最大的心理陷阱。回測與實盤數據證明：**盲目追求交易頻率是導致帳戶虧損的主因**。

### 1. 交易摩擦成本（Friction）的致命性
* **手續費與稅**：雙邊手續費約 40 元 + 期交稅 ≈ 42~45 元。
* **滑價成本**：買賣各 1 tick 滑價（約 10~20 元）。
* **單筆固定損耗**：每進出一趟至少損耗 **45 元（相當於 4.5 個指數點）**。
* **歷史教訓**：先前測試 500+ 組「每天無條件開盤抓 10~15 點小波動」的策略，因震盪雜訊過大與摩擦耗損，最終皆累積**虧損 -25,000 ～ -65,000 元**；300 組「每天 MACD 交叉就追」更是大虧 **-94,000 元**。

### 2. 台指期的盤型分布
* 台股約有 **40%～50% 的日子屬於「窄幅死魚盤」**（全日高低差不足 60 點）。
* 在這種盤型下任何突破都是假突破、任何均線交叉都是兩面挨巴掌。
* **高勝率與高盈虧比的精髓**：**「學會空手等待」**。嚴格透過跳空幅度、開盤區間與量能確認，濾除死魚盤，只在大肉行情出現時揮棒。

---

## 二、 雙軌並行組合（Multi-Strategy Ensemble）

為同時滿足「高勝率踏實感」與「吃到大波段暴利」，`tidemark` 採用雙軌並行設計：

```text
                                  【08:45 開盤】
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
               【軌道 A：穩健高勝率】           【軌道 B：極致暴利波段】
                 tmf_daily_secure                 tmf_daily_trend
                         │                               │
        • 跳空 ≥ 40 點 + OR 突破        • 跳空 ≥ 40 點 + OR 突破
        • 量能爆發確認 (≥ 1.2x)         • 全天候捕捉突破 (不限爆量)
        • 停損 50 點 / 停利 200 點      • 停損 50 點 / 停利 200 點
        • 獲利達 +50 點 立即鎖保本       • 不移保本，硬抱 200 點
        • 45 分鐘無動能時間停損         • 45 分鐘無動能時間停損
                         │                               │
                         ▼                               ▼
               勝率 59.9% (OOS 63.2%)            勝率 44.3% (OOS 39.5%)
               淨利 +13,061 元（HL 覆核）         淨利 +30,802 元（HL 覆核）
               心理較舒適（保本）                 大單邊利潤極大化
               ※ secure ⊆ trend，非互補雙軌
```

---

## 三、 完整參數設定表（`config/settings.yaml`）

### 軌道 A：穩健高勝率旗艦（`tmf_daily_secure`）
```yaml
- id: tmf_daily_secure
  enabled: true
  symbol: TMF
  params:
    inner: open_gap_orb
    probe_mins: 15            # 觀察開盤前 15 分鐘 (08:45-09:00) 建立高低區間
    cutoff_mins: 105          # 超過 10:30 (開盤 105 分鐘) 後不再建立新倉
    gap_th: 40.0              # 隔夜跳空需 ≥ 40 點，確保具備動能方向
    max_or_range: 250.0       # 開盤區間 ≤ 250 點 (適應 2026 波動放大，避免已噴出追高)
    vol_spike_mult: 1.2       # 突破 K 線量能必須 ≥ 前 5 根均量 1.2 倍
    allow_short: false        # 專注台股強大多頭動能，不逆勢作空
    sessions: [day]           # 日盤專用
    take_profit_points: 200   # 目標停利 200 點 (+2,000 元)
    stop_points: 50           # 初始保護停損 50 點 (-500 元)
    breakeven_points: 50      # ★核心關鍵：盤中獲利達 +50 點時，停損直接移至成本價+5點
    time_stop_bars: 45        # 進場 45 分鐘若未達標且無動能，平倉離場
    max_daily_loss: 0
    flatten_off_session: true
    flatten_eod: true
    max_lots: 1
    timeframe: 1m
    prefilter_enabled: false
```

### 軌道 B：極致暴利波段旗艦（`tmf_daily_trend`）
```yaml
- id: tmf_daily_trend
  enabled: true
  symbol: TMF
  params:
    inner: open_gap_orb
    probe_mins: 15            # 觀察開盤前 15 分鐘
    cutoff_mins: 105          # 10:30 後不追單
    gap_th: 40.0              # 隔夜跳空 ≥ 40 點
    max_or_range: 250.0       # 開盤區間 ≤ 250 點
    vol_spike_mult: 0.0       # 降低量能門檻，確保所有單邊突破行情不漏接
    allow_short: false
    sessions: [day]
    take_profit_points: 200   # 完整追求 200 點大行情
    stop_points: 50           # 嚴格 50 點停損（風報比 1 : 4）
    breakeven_points: 0       # 不提早移保本，容忍洗盤以賺取最大波段
    time_stop_bars: 45        # 45 分鐘時間停損
    max_daily_loss: 0
    flatten_off_session: true
    flatten_eod: true
    max_lots: 1
    timeframe: 1m
    prefilter_enabled: false
```

---

## 四、 回測成效數據比較

| 指標 | 軌道 A（穩健型 secure） | 軌道 B（暴利型 trend） | 原版 RSI 回歸（tmf_v1） |
| :--- | :--- | :--- | :--- |
| **交易訊號原理** | 15m 開盤突破 + 量能 + 保本 | 15m 開盤突破 + 趨勢長抱 | 28期 RSI 超賣反彈 |
| **總淨損益（研究 HL 觸價）** | **+13,061 元**（舊文 +15,411 已覆核下修） | **+30,802 元** | +10,132 元 |
| **總淨損益（正式 close 觸價）** | +21,961 元 | +22,712 元 | — |
| **全期勝率（HL）** | **59.9%** | **44.3%** | 82.1% |
| **樣本外 (OOS) 勝率（HL）**| **63.2%** | **39.5%**（舊文 42.1%） | 80.0% |
| **單筆風報比** | 動態保本保護（保本後風險為0）| **1 : 4**（賠 50 點 / 賺 200 點）| 7.5 : 1（賠 300 / 賺 40）|
| **每週交易次數** | 約 1.5～2 次 | 約 2.5～3 次 | 約 0.5 次（2週1次）|
| **最大拉回 (MaxDD)**| 研究引擎未畫權益曲線 → 報告中 DD=0 **不可信**；需另算 | 同上 | 較大（單次被抓停損 300 點）|

> 覆核指令：`python scripts/verify_ensemble.py` → `outputs/ensemble_verify_report.json`（2026-10-26）。
> **結構事實**：secure 進場日 = trend 子集（157/158 重疊）；不是互補雙軌。YAML 雙 `enabled` 尚無 ensemble 編排器；`run_backtest`／paper 預設仍走 `tmf_v1`。

---

## 五、 「多策略並行」的倉位與資金排他規則

1. **資金與口數鐵律**：
   * 專案初始本金為 **10,000 元**。
   * 遵守 `AGENTS.md` 硬規則：任何下單路徑 `max_lots: 1`。
2. **排他保護機制**：
   * 若當天 09:01 同時符合軌道 A 與軌道 B 的進場條件，因 `max_lots: 1` 限制，**系統僅會由優先觸發的策略持有一口**，絕不超額槓桿下第二口。
   * 兩套策略若在不同日觸發，則各自獨立執行，完美兼顧「日常保護」與「大行情爆發」。

---

## 六、 未來拉高交易頻率的正確擴展途徑

若未來希望達到「每天都有機會交易」，正統且安全的進階路徑如下：

1. **時段擴展（日盤 + 夜盤，推薦）**：
   * 現有軌道專注於 **日盤（08:45–13:45）** 開盤動能。
   * 下一階段可開闢 **夜盤（15:00–05:00）** 美股開盤動能模組（21:30 美股開盤跳空突破），每週交易時段由 5 個翻倍至 10 個。
2. **盤型分流器（Regime Switcher）**
   * **v1 偽互補**（secure vs trend，同族 ORB）：`research_ensemble_selector.py` → `ensemble_selector_report.json`
   * **v2 真互補**（ORB 動能 vs RSI28 回歸）：`research_complementary_selector.py` → `complementary_selector_report.json`
   * **紙上主路徑（一口）**：`trading.paper_strategy: complementary`；鎖 `gap≥120→orb else rsi` + ORB 保本 50pt → `run_paper_standard.py`；鎖檔 `best_complementary_lock.json`
   * 硬化掃描：`research_complementary_harden.py`；SIM 探測：`run_paper_sim_probe.py`；日流程：`run_sim_day.py`；即時 tick：`run_live_tick.py`（tick→1m 關棒→互補；`--subscribe`）
   * 時機：探針結束（≈09:00）才分流／下單；RSI 只計 09:00 後進場。基線對照：`paper_strategy: tmf_v1`。
