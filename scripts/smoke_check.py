#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-1 sanity check for tidemark."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import SecondBarBacktester
from src.core.config import load_settings, notify_prefix
from src.data.local_store import LocalStore
from src.strategy.tmf_session import TmfSessionStrategy


def main() -> int:
    settings = load_settings()
    brand = settings["brand"]["name"]
    trading = settings["trading"]
    assert brand == "tidemark", brand
    assert trading["instrument"] == "TMF"
    assert trading["capital"] == 10000
    assert trading["max_lots"] == 1
    assert set(trading["sessions"]) == {"day", "night"}

    store = LocalStore(settings)
    assert store.cache_dir.exists()

    # synthetic 1s series — engine must step one row at a time
    n = 5
    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-02 09:00:00", periods=n, freq="s"),
            "close": [18000 + i for i in range(n)],
        }
    )
    strat = TmfSessionStrategy(
        max_lots=trading["max_lots"],
        capital=trading["capital"],
    )
    bt = SecondBarBacktester(
        strat,
        capital=trading["capital"],
        point_value=trading["point_value"],
        fee_per_side=trading.get("fee_per_side", 20),
        tax_rate=trading.get("tax_rate", 0.00002),
    )
    result = bt.run(bars)
    assert result.n_steps == n, result.n_steps
    assert result.final_equity == trading["capital"]

    # Phase 2 gate: no look-ahead
    from tests.test_no_lookahead import main as lookahead_main

    la = lookahead_main()
    assert la == 0, "look-ahead tests failed"

    from tests.test_paper_ledger import main as paper_main

    pa = paper_main()
    assert pa == 0, "paper ledger tests failed"

    from tests.test_session_win_rates import main as sess_wr_main

    sw = sess_wr_main()
    assert sw == 0, "session win-rate tests failed"

    from tests.test_day_prefilters import main as day_pf_main

    dp = day_pf_main()
    assert dp == 0, "day prefilter tests failed"

    from tests.test_open_gap_orb import main as orb_main

    ob = orb_main()
    assert ob == 0, "open gap orb tests failed"

    from tests.test_ensemble_selector import main as ens_sel_main

    es = ens_sel_main()
    assert es == 0, "ensemble selector tests failed"

    from tests.test_complementary_fast import main as comp_main

    cf = comp_main()
    assert cf == 0, "complementary fast tests failed"

    from tests.test_sim_day_loop import main as sim_day_main

    sd = sim_day_main()
    assert sd == 0, "sim day loop tests failed"

    from tests.test_minute_bars import main as minute_main

    mb = minute_main()
    assert mb == 0, "minute bar tests failed"

    from tests.test_live_tick_loop import main as live_tick_main

    lt = live_tick_main()
    assert lt == 0, "live tick loop tests failed"

    from tests.test_complementary_modules import main as comp_mod_main

    cm = comp_mod_main()
    assert cm == 0, "complementary module boundary tests failed"

    from tests.test_ops_ledger import main as ops_led_main

    ol = ops_led_main()
    assert ol == 0, "ops ledger tests failed"

    from tests.test_ui_app import main as ui_app_main

    ui_ok = ui_app_main()
    assert ui_ok == 0, "ui app tests failed"

    print(f"{notify_prefix(settings)} smoke ok | brand={brand} | steps={result.n_steps}")
    print(f"  instrument={trading['instrument']} capital={trading['capital']} max_lots={trading['max_lots']}")
    print(f"  sessions={trading['sessions']}")
    print(f"  cache={store.cache_dir}")
    print("  look-ahead tests: pass")
    print("  paper ledger tests: pass")
    print("  session win-rate split: pass")
    print("  day prefilter: pass")
    print("  open gap orb: pass")
    print("  ensemble selector: pass")
    print("  complementary fast: pass")
    print("  sim day loop: pass")
    print("  minute bars: pass")
    print("  live tick loop: pass")
    print("  complementary modules: pass")
    print("  ops ledger: pass")
    print("  ui app: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
