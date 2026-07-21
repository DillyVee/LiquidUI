"""
Time-Cycle Sweep - Yahoo Finance calendar-cycle backtest across many tickers.

For every ticker in a list, this pulls daily bars from Yahoo Finance once,
then backtests a pure calendar ON/OFF/SHIFT cycle over its full history and
records return, drawdown, profit factor, Sharpe and Sortino for every cycle
configuration.

The cycle
---------
A configuration is three integers ``(buy, out, offset)``:

  * ``buy``    - consecutive days held IN the market (long the asset)
  * ``out``    - consecutive days OUT of the market (flat / cash)
  * ``offset`` - how many days the whole ON/OFF pattern is shifted to the right

``3, 4, 2`` therefore means "3 days in, 4 days out, pattern shifted right by 2".
The cycle length is ``buy + out`` and there is one distinct shift for each day
of the cycle, so a 3/4 cycle has 7 possible offsets (0..6) - exactly the "7
overall shifts" described in the request.

Iteration order
---------------
Configurations are grouped by ``(buy, out)`` and ordered by cycle length
ascending, then by out-length ascending::

    (1,1) -> offsets 0,1
    (2,1) -> offsets 0,1,2
    (1,2) -> offsets 0,1,2
    (3,1) -> offsets 0,1,2,3
    (2,2) -> ...
    (1,3) -> ...

i.e. 1,1,0 / 1,1,1 / 2,1,0 / 2,1,1 / 2,1,2 / 1,2,0 / ...

The outer loop walks the ``(buy, out)`` groups; the inner loop walks every
ticker. So every ticker is scored on the (1,1) cycle before any ticker moves
on to (2,1) - matching "all through the 500 and then goes back to AAPL to run
2,1,0". Downloaded price data is cached to disk, so coming back to a ticker for
the next group re-uses the data instead of hitting Yahoo again.

Output
------
One master CSV (default ``data_output/cycle_backtest_results.csv``) with one
row per ``(ticker, buy, out, offset)``, appended and flushed as each result is
computed so it updates in real time. The run is resumable: on restart, rows
already present are skipped, and tickers Yahoo could not return are remembered
and skipped without re-downloading.

Monitoring
----------
A dependency-free local web dashboard (default http://localhost:8765) shows the
current cycle/ticker, progress and ETA, the best results so far and the most
recent results. Disable it with ``--no-ui``.

Usage
-----
    python cycle_backtest.py                 # full 10x10 sweep, all tickers
    python cycle_backtest.py --max-buy 7 --max-out 7
    python cycle_backtest.py --no-ui         # headless, console logs only
    python cycle_backtest.py --selftest      # verify the math, no network

Trading involves substantial risk. This is research tooling only.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    """
    Load a project module straight from its file, bypassing the package
    ``__init__`` (which imports matplotlib/optuna/etc.). We only need the
    Yahoo loader and the metrics engine, both of which depend on nothing but
    numpy/pandas/yfinance, so this keeps the sweep lightweight for a headless,
    days-long run.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, BASE_DIR / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the project's Yahoo loader (cleaning / quality checks) and its single
# source of truth for performance metrics, so numbers match the rest of the app.
DataLoader = _load_module("_cycle_loader", "data/loader.py").DataLoader
PerformanceMetrics = _load_module(
    "_cycle_metrics", "optimization/metrics.py"
).PerformanceMetrics
DEFAULT_TICKERS = BASE_DIR / "data" / "top_500_tickers.txt"
DEFAULT_OUTPUT = BASE_DIR / "data_output" / "cycle_backtest_results.csv"
CACHE_DIR = BASE_DIR / "data_output" / "cycle_cache"
UNAVAILABLE_FILE = CACHE_DIR / "_unavailable.json"

# Columns written to the results CSV, in order.
CSV_FIELDS = [
    "timestamp",
    "ticker",
    "buy",
    "out",
    "offset",
    "cycle_len",
    "n_days",
    "start_date",
    "end_date",
    "return_pct",
    "max_drawdown_pct",
    "profit_factor",
    "sharpe",
    "sortino",
    "cagr_pct",
    "calmar_ratio",
    "n_trades",
    "win_rate_pct",
    "buyhold_return_pct",
]

INITIAL_EQUITY = 1000.0
ANNUALIZATION = 252.0


# ============================================================
# TICKERS AND CONFIGURATIONS
# ============================================================
def load_tickers(path: Path) -> List[str]:
    """
    Read a ticker list. Tolerates a plain symbol per line or a numbered list
    (``1<TAB>NVDA``); the symbol is the last whitespace-delimited token. Handles
    CRLF endings, blank lines and duplicates (first occurrence wins).
    """
    tickers: List[str] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            symbol = line.split()[-1].strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                tickers.append(symbol)
    return tickers


def to_yahoo_symbol(symbol: str) -> str:
    """Yahoo uses '-' where tickers carry a class suffix (BRK.B -> BRK-B)."""
    return DataLoader.normalize_ticker(symbol).replace(".", "-")


def generate_config_groups(max_buy: int, max_out: int) -> List[Tuple[int, int]]:
    """
    Every ``(buy, out)`` pair with ``1 <= buy <= max_buy`` and
    ``1 <= out <= max_out``, ordered by cycle length (buy+out) ascending, then
    by out-length ascending. This reproduces the requested progression
    1,1 -> 2,1 -> 1,2 -> 3,1 -> 2,2 -> 1,3 -> ...
    """
    combos = [
        (buy, out)
        for buy in range(1, max_buy + 1)
        for out in range(1, max_out + 1)
    ]
    combos.sort(key=lambda bo: (bo[0] + bo[1], bo[1]))
    return combos


# ============================================================
# THE BACKTEST
# ============================================================
def cycle_in_market(n: int, buy: int, out: int, offset: int) -> np.ndarray:
    """
    Boolean in-market mask for ``n`` days under a ``(buy, out, offset)`` cycle.

    Day ``i`` behaves like day ``i - offset`` of the unshifted pattern (a
    right-shift), and the pattern is ``buy`` ON days followed by ``out`` OFF
    days, repeating. Anchoring the phase to position 0 of the series makes a
    given offset reproducible for that ticker.
    """
    cycle_len = buy + out
    pos = (np.arange(n) - offset) % cycle_len
    return pos < buy


def run_cycle_backtest(
    close: np.ndarray, buy: int, out: int, offset: int
) -> Optional[Dict[str, float]]:
    """
    Backtest one cycle configuration on a daily close series.

    On days the cycle is ON we earn the asset's close-to-close return; on OFF
    days we earn nothing (flat). The equity curve and per-trade returns (one
    "trade" = one contiguous ON block) are handed to the project's
    ``PerformanceMetrics`` so Sharpe/Sortino/profit-factor/drawdown match the
    rest of LiquidUI. The cycle is a fixed calendar schedule, known in advance,
    so holding through day ``i`` introduces no look-ahead.
    """
    n = len(close)
    if n < 3:
        return None

    daily_ret = np.zeros(n, dtype=np.float64)
    daily_ret[1:] = close[1:] / close[:-1] - 1.0

    in_market = cycle_in_market(n, buy, out, offset)
    strat_ret = np.where(in_market, daily_ret, 0.0)

    equity = INITIAL_EQUITY * np.cumprod(1.0 + strat_ret)

    # Per-trade returns: compound (1+ret) over each contiguous ON block.
    # cumprod with a leading 1.0 lets a block [s, e] be priced as
    # cp[e+1] / cp[s] - 1 in one vectorized shot.
    trade_returns: Optional[np.ndarray] = None
    im = in_market.astype(np.int8)
    edges = np.diff(np.concatenate(([0], im, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1  # inclusive
    if len(starts):
        cp = np.concatenate(([1.0], np.cumprod(1.0 + strat_ret)))
        trade_returns = (cp[ends + 1] / cp[starts] - 1.0) * 100.0

    return PerformanceMetrics.calculate_metrics(
        equity, annualization_factor=ANNUALIZATION, trade_returns=trade_returns
    )


# ============================================================
# DATA (download + disk cache)
# ============================================================
class PriceCache:
    """Download daily bars once per ticker and cache them on disk."""

    def __init__(self, history_days: int, refresh: bool = False, pause: float = 0.0):
        self.history_days = history_days
        self.refresh = refresh
        self.pause = pause  # polite delay after each live download
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.unavailable: Set[str] = self._load_unavailable()

    @staticmethod
    def _safe(symbol: str) -> str:
        return symbol.replace("/", "_").replace(".", "_").replace("-", "_")

    def _cache_path(self, symbol: str) -> Path:
        return CACHE_DIR / f"{self._safe(symbol)}.csv"

    def _load_unavailable(self) -> Set[str]:
        if UNAVAILABLE_FILE.exists():
            try:
                return set(json.loads(UNAVAILABLE_FILE.read_text()))
            except (ValueError, OSError):
                return set()
        return set()

    def _mark_unavailable(self, symbol: str) -> None:
        self.unavailable.add(symbol)
        try:
            UNAVAILABLE_FILE.write_text(json.dumps(sorted(self.unavailable)))
        except OSError:
            pass

    def get_close(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[pd.Timestamp], Optional[pd.Timestamp], str]:
        """
        Return ``(close_array, start_date, end_date, source)`` for a ticker.

        ``source`` is "cache", "download" or "skip". A None array means the
        ticker should be skipped (already known bad, or download failed).
        """
        if symbol in self.unavailable:
            return None, None, None, "skip"

        path = self._cache_path(symbol)
        if path.exists() and not self.refresh:
            try:
                df = pd.read_csv(path, parse_dates=["Datetime"])
                if len(df) >= 3 and "Close" in df.columns:
                    return (
                        df["Close"].to_numpy(dtype=np.float64),
                        df["Datetime"].iloc[0],
                        df["Datetime"].iloc[-1],
                        "cache",
                    )
            except (ValueError, OSError, KeyError):
                pass  # fall through and re-download

        df = self._download(symbol)
        if df is None or len(df) < 3:
            self._mark_unavailable(symbol)
            return None, None, None, "skip"

        try:
            df[["Datetime", "Close"]].to_csv(path, index=False)
        except OSError:
            pass
        return (
            df["Close"].to_numpy(dtype=np.float64),
            df["Datetime"].iloc[0],
            df["Datetime"].iloc[-1],
            "download",
        )

    def _download(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Daily-only download. Tries yfinance first (its curl_cffi backend avoids
        Yahoo's throttling of plain clients); if that yields nothing, falls back
        to a direct request against Yahoo's chart API. Only when both fail is a
        ticker treated as unavailable and skipped.
        """
        df = self._download_yfinance(symbol)
        if df is None or len(df) < 3:
            df = self._download_requests(symbol)
        if self.pause > 0:
            time.sleep(self.pause)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        df = df[df["Close"] > 0].reset_index(drop=True)
        return df if len(df) >= 3 else None

    def _download_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """Primary path: yfinance, cleaned via the project loader."""
        try:
            import yfinance as yf
        except ImportError:
            return None

        yahoo = to_yahoo_symbol(symbol)
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=self.history_days)
        try:
            raw = yf.download(
                yahoo,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception:
            return None
        if raw is None or raw.empty:
            return None
        try:
            df = DataLoader._process_dataframe(raw)
        except Exception:
            return None
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df

    def _download_requests(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fallback path: Yahoo v8 chart API via requests. Uses the split/dividend
        adjusted close when available, so returns match yfinance auto_adjust.
        """
        try:
            import requests
        except ImportError:
            return None

        yahoo = to_yahoo_symbol(symbol)
        end = int(time.time())
        start = end - self.history_days * 86400
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo}"
        params = {
            "period1": start,
            "period2": end,
            "interval": "1d",
            "includeAdjustedClose": "true",
            "events": "div,splits",
        }
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
            result = (payload.get("chart") or {}).get("result")
            if not result:
                return None
            res = result[0]
            timestamps = res.get("timestamp")
            if not timestamps:
                return None
            quote = res["indicators"]["quote"][0]
            close = quote.get("close")
            adj = res["indicators"].get("adjclose")
            adjclose = adj[0].get("adjclose") if adj else None
            series = adjclose if adjclose else close
            if not series:
                return None
            df = pd.DataFrame(
                {
                    "Datetime": pd.to_datetime(timestamps, unit="s"),
                    "Close": series,
                }
            ).dropna()
            return df.reset_index(drop=True) if len(df) >= 3 else None
        except Exception:
            return None


# ============================================================
# SHARED STATE (worker -> dashboard)
# ============================================================
class RunState:
    """Thread-safe snapshot of progress, best and recent results."""

    def __init__(self, total_grid: int):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_grid = total_grid  # upper bound on rows across the full grid
        self.session_rows = 0  # rows written this process
        self.total_rows = 0  # rows in the CSV incl. resumed
        self.current_group: Tuple[int, int] = (0, 0)
        self.group_index = 0
        self.group_total = 0
        self.current_ticker = ""
        self.ticker_index = 0
        self.ticker_total = 0
        self.downloaded = 0
        self.from_cache = 0
        self.skipped: List[str] = []
        self.recent: Deque[Dict] = deque(maxlen=25)
        self.best: List[Dict] = []  # top by return_pct
        self.finished = False

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def record_row(self, row: Dict) -> None:
        with self._lock:
            self.session_rows += 1
            self.total_rows += 1
            self.recent.appendleft(row)
            ret = row.get("return_pct")
            if ret is not None and ret == ret:  # not NaN
                self.best.append(row)
                self.best.sort(key=lambda r: r["return_pct"], reverse=True)
                del self.best[25:]

    def note_skip(self, symbol: str) -> None:
        with self._lock:
            if symbol not in self.skipped:
                self.skipped.append(symbol)

    def snapshot(self) -> Dict:
        with self._lock:
            elapsed = time.time() - self.started_at
            rate = self.session_rows / elapsed if elapsed > 0 else 0.0
            remaining = max(self.total_grid - self.total_rows, 0)
            eta = remaining / rate if rate > 0 else 0.0
            return {
                "started_at": self.started_at,
                "elapsed": elapsed,
                "eta": eta,
                "rate": rate,
                "total_grid": self.total_grid,
                "session_rows": self.session_rows,
                "total_rows": self.total_rows,
                "current_group": self.current_group,
                "group_index": self.group_index,
                "group_total": self.group_total,
                "current_ticker": self.current_ticker,
                "ticker_index": self.ticker_index,
                "ticker_total": self.ticker_total,
                "downloaded": self.downloaded,
                "from_cache": self.from_cache,
                "skipped": list(self.skipped),
                "recent": list(self.recent),
                "best": list(self.best),
                "finished": self.finished,
            }


# ============================================================
# WEB DASHBOARD (stdlib only)
# ============================================================
def _fmt_secs(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _num(value, digits=2) -> str:
    try:
        if value is None or value != value:  # None or NaN
            return "-"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _results_table(rows: List[Dict], title: str) -> str:
    head = (
        "<tr><th>Ticker</th><th>buy</th><th>out</th><th>off</th>"
        "<th>Return %</th><th>Max DD %</th><th>PF</th><th>Sharpe</th>"
        "<th>Sortino</th><th>Trades</th></tr>"
    )
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td class='sym'>{r.get('ticker','')}</td>"
            f"<td>{r.get('buy','')}</td><td>{r.get('out','')}</td>"
            f"<td>{r.get('offset','')}</td>"
            f"<td class='num pos'>{_num(r.get('return_pct'))}</td>"
            f"<td class='num neg'>{_num(r.get('max_drawdown_pct'))}</td>"
            f"<td class='num'>{_num(r.get('profit_factor'))}</td>"
            f"<td class='num'>{_num(r.get('sharpe'), 3)}</td>"
            f"<td class='num'>{_num(r.get('sortino'), 3)}</td>"
            f"<td class='num'>{r.get('n_trades','')}</td>"
            "</tr>"
        )
    if not body:
        body.append("<tr><td colspan='10' class='muted'>none yet</td></tr>")
    return (
        f"<h2>{title}</h2><div class='tablewrap'><table>"
        f"{head}{''.join(body)}</table></div>"
    )


def render_dashboard(snap: Dict) -> str:
    buy, out = snap["current_group"]
    pct = (snap["total_rows"] / snap["total_grid"] * 100) if snap["total_grid"] else 0.0
    status = "FINISHED" if snap["finished"] else "RUNNING"
    status_cls = "done" if snap["finished"] else "run"
    skipped = snap["skipped"]
    skipped_str = ", ".join(skipped[:60]) + ("  ..." if len(skipped) > 60 else "")

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<title>Cycle Sweep - {status}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,monospace; margin:0; padding:24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:#8b949e; margin:26px 0 8px; }}
  .sub {{ color:#8b949e; font-size:13px; margin-bottom:18px; }}
  .badge {{ padding:2px 10px; border-radius:12px; font-weight:600; font-size:12px; }}
  .badge.run {{ background:#1f6feb33; color:#58a6ff; }}
  .badge.done {{ background:#2ea04333; color:#3fb950; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:8px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 14px; }}
  .card .k {{ color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
  .card .v {{ font-size:20px; font-weight:600; margin-top:4px; }}
  .card .v.big {{ font-size:22px; color:#58a6ff; }}
  .bar {{ height:14px; background:#161b22; border:1px solid #30363d; border-radius:8px; overflow:hidden; margin:8px 0 4px; }}
  .bar > span {{ display:block; height:100%; background:linear-gradient(90deg,#1f6feb,#3fb950); width:{pct:.3f}%; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ padding:5px 9px; text-align:right; border-bottom:1px solid #21262d; }}
  th {{ color:#8b949e; font-weight:600; text-align:right; position:sticky; top:0; background:#0d1117; }}
  td.sym {{ text-align:left; font-weight:600; color:#e6edf3; }}
  th:first-child {{ text-align:left; }}
  .num {{ font-variant-numeric:tabular-nums; }}
  .pos {{ color:#3fb950; }} .neg {{ color:#f85149; }}
  .muted {{ color:#6e7681; text-align:center; }}
  .tablewrap {{ overflow-x:auto; border:1px solid #30363d; border-radius:8px; }}
  .skips {{ color:#8b949e; font-size:12px; word-break:break-word; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px 12px; }}
</style></head>
<body>
  <h1>Time-Cycle Sweep &nbsp; <span class="badge {status_cls}">{status}</span></h1>
  <div class="sub">Yahoo Finance daily-cycle backtest &middot; auto-refresh 3s &middot; started {_fmt_secs(snap['elapsed'])} ago</div>

  <div class="grid">
    <div class="card"><div class="k">Current cycle</div><div class="v big">{buy} / {out}</div></div>
    <div class="card"><div class="k">Cycle group</div><div class="v">{snap['group_index']} / {snap['group_total']}</div></div>
    <div class="card"><div class="k">Current ticker</div><div class="v">{snap['current_ticker'] or '-'}</div></div>
    <div class="card"><div class="k">Ticker</div><div class="v">{snap['ticker_index']} / {snap['ticker_total']}</div></div>
    <div class="card"><div class="k">Rows written</div><div class="v">{snap['total_rows']:,}</div></div>
    <div class="card"><div class="k">This session</div><div class="v">{snap['session_rows']:,}</div></div>
    <div class="card"><div class="k">Rate</div><div class="v">{snap['rate']:.1f}/s</div></div>
    <div class="card"><div class="k">ETA</div><div class="v">{_fmt_secs(snap['eta'])}</div></div>
    <div class="card"><div class="k">Downloaded</div><div class="v">{snap['downloaded']}</div></div>
    <div class="card"><div class="k">Skipped</div><div class="v">{len(skipped)}</div></div>
  </div>

  <div class="bar"><span></span></div>
  <div class="sub">{pct:.3f}% of the full grid ({snap['total_rows']:,} / {snap['total_grid']:,} configs)</div>

  {_results_table(snap['best'], 'Best return so far')}
  {_results_table(snap['recent'], 'Most recent')}

  <h2>Skipped tickers ({len(skipped)})</h2>
  <div class="skips">{skipped_str or 'none'}</div>
</body></html>"""


def start_dashboard(state: RunState, port: int) -> Optional[ThreadingHTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("", "/data"):
                if self.path.rstrip("/") == "/data":
                    payload = json.dumps(state.snapshot(), default=str).encode()
                    ctype = "application/json"
                else:
                    payload = render_dashboard(state.snapshot()).encode()
                    ctype = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)

        def log_message(self, *args):  # silence per-request logging
            pass

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError as exc:
        print(f"[ui] could not start dashboard on port {port}: {exc}")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[ui] dashboard live at http://localhost:{port}")
    return server


# ============================================================
# RESULTS CSV (append + resume)
# ============================================================
def load_done_keys(path: Path) -> Set[Tuple[str, int, int, int]]:
    """Set of (ticker, buy, out, offset) already present, for resuming."""
    done: Set[Tuple[str, int, int, int]] = set()
    if not path.exists():
        return done
    try:
        with open(path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    done.add(
                        (
                            row["ticker"],
                            int(row["buy"]),
                            int(row["out"]),
                            int(row["offset"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
    except OSError:
        pass
    return done


class ResultsWriter:
    """Append rows to the master CSV, flushing after each so it updates live."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: Dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


def build_row(
    ticker: str,
    buy: int,
    out: int,
    offset: int,
    n_days: int,
    start_date,
    end_date,
    metrics: Optional[Dict[str, float]],
    buyhold: float,
) -> Dict:
    """Flatten a metrics dict into a CSV row (blank metrics -> NaN-ish)."""
    m = metrics or {}
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "buy": buy,
        "out": out,
        "offset": offset,
        "cycle_len": buy + out,
        "n_days": n_days,
        "start_date": pd.Timestamp(start_date).date() if start_date is not None else "",
        "end_date": pd.Timestamp(end_date).date() if end_date is not None else "",
        "return_pct": m.get("Percent_Gain_%", float("nan")),
        "max_drawdown_pct": m.get("Max_Drawdown_%", float("nan")),
        "profit_factor": m.get("Profit_Factor", float("nan")),
        "sharpe": m.get("Sharpe_Ratio", float("nan")),
        "sortino": m.get("Sortino_Ratio", float("nan")),
        "cagr_pct": m.get("CAGR_%", float("nan")),
        "calmar_ratio": m.get("Calmar_Ratio", float("nan")),
        "n_trades": m.get("_n_trades", ""),
        "win_rate_pct": m.get("Win_Rate_%", ""),
        "buyhold_return_pct": round(buyhold, 2),
    }


# ============================================================
# MAIN SWEEP
# ============================================================
def run_sweep(args: argparse.Namespace) -> None:
    tickers = load_tickers(Path(args.tickers))
    if not tickers:
        print(f"No tickers found in {args.tickers}")
        return

    groups = generate_config_groups(args.max_buy, args.max_out)
    offsets_per_group = {g: g[0] + g[1] for g in groups}
    total_offsets = sum(offsets_per_group.values())
    total_grid = total_offsets * len(tickers)

    out_path = Path(args.output)
    done = load_done_keys(out_path)

    print("=" * 66)
    print("  TIME-CYCLE SWEEP")
    print("=" * 66)
    print(f"  tickers        : {len(tickers)}")
    print(f"  cycle groups   : {len(groups)}  (buy 1-{args.max_buy}, out 1-{args.max_out})")
    print(f"  configs/ticker : {total_offsets}")
    print(f"  full grid      : {total_grid:,} configs")
    print(f"  already done   : {len(done):,} (resuming)")
    print(f"  output         : {out_path}")
    print(f"  history        : up to {args.history_days} days of daily bars")
    print("=" * 66)

    cache = PriceCache(
        history_days=args.history_days, refresh=args.refresh, pause=args.pause
    )
    state = RunState(total_grid=total_grid)
    state.update(
        group_total=len(groups),
        ticker_total=len(tickers),
        total_rows=len(done),
    )

    server = None
    if not args.no_ui:
        server = start_dashboard(state, args.port)

    writer = ResultsWriter(out_path)
    interrupted = False

    try:
        for gi, (buy, out) in enumerate(groups, start=1):
            state.update(current_group=(buy, out), group_index=gi)
            cycle_len = buy + out
            for ti, ticker in enumerate(tickers, start=1):
                state.update(current_ticker=ticker, ticker_index=ti)

                # Is every offset for this (ticker, group) already recorded?
                if all(
                    (ticker, buy, out, off) in done for off in range(cycle_len)
                ):
                    continue

                close, start_date, end_date, source = cache.get_close(ticker)
                if source == "download":
                    state.update(downloaded=state.downloaded + 1)
                elif source == "cache":
                    state.update(from_cache=state.from_cache + 1)

                if close is None:
                    state.note_skip(ticker)
                    # Mark the whole group done for this ticker so we don't
                    # keep probing a symbol Yahoo can't return.
                    for off in range(cycle_len):
                        done.add((ticker, buy, out, off))
                    continue

                n_days = len(close)
                buyhold = float(close[-1] / close[0] - 1.0) * 100.0

                for offset in range(cycle_len):
                    if (ticker, buy, out, offset) in done:
                        continue
                    metrics = run_cycle_backtest(close, buy, out, offset)
                    if metrics is not None:
                        # count trades = contiguous ON blocks (for the row)
                        metrics["_n_trades"] = _count_trades(n_days, buy, out, offset)
                    row = build_row(
                        ticker, buy, out, offset, n_days,
                        start_date, end_date, metrics, buyhold,
                    )
                    writer.write(row)
                    state.record_row(row)
                    done.add((ticker, buy, out, offset))

                if ti % 25 == 0 or ti == len(tickers):
                    snap = state.snapshot()
                    print(
                        f"  cycle {buy}/{out} [{gi}/{len(groups)}]  "
                        f"ticker {ti}/{len(tickers)} ({ticker})  "
                        f"rows={snap['total_rows']:,}  "
                        f"eta={_fmt_secs(snap['eta'])}"
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] finishing current write and shutting down...")
    finally:
        writer.close()
        state.update(finished=not interrupted)
        if server is not None:
            if interrupted:
                server.shutdown()
            else:
                print(f"[ui] run complete - dashboard still live at "
                      f"http://localhost:{args.port} (Ctrl-C to exit)")
                try:
                    while True:
                        time.sleep(3600)
                except KeyboardInterrupt:
                    server.shutdown()

    snap = state.snapshot()
    print("-" * 66)
    print(f"  rows this session : {snap['session_rows']:,}")
    print(f"  rows total        : {snap['total_rows']:,}")
    print(f"  downloaded        : {snap['downloaded']}")
    print(f"  skipped tickers   : {len(snap['skipped'])}")
    print(f"  results CSV       : {out_path}")
    print("-" * 66)


def _count_trades(n: int, buy: int, out: int, offset: int) -> int:
    """Number of contiguous ON blocks (trades) over ``n`` days."""
    im = cycle_in_market(n, buy, out, offset).astype(np.int8)
    return int(np.sum(np.diff(np.concatenate(([0], im))) == 1))


# ============================================================
# SELF-TEST (no network)
# ============================================================
def selftest() -> int:
    """Validate the cycle mask, vectorized trade returns and metric wiring."""
    ok = True

    # 1) in-market mask for 2/1 cycle, offsets 0,1,2 over 9 days
    expect = {
        0: [1, 1, 0, 1, 1, 0, 1, 1, 0],
        1: [0, 1, 1, 0, 1, 1, 0, 1, 1],
        2: [1, 0, 1, 1, 0, 1, 1, 0, 1],
    }
    for off, exp in expect.items():
        got = cycle_in_market(9, 2, 1, off).astype(int).tolist()
        status = "ok" if got == exp else "FAIL"
        ok &= got == exp
        print(f"  mask 2/1 off={off}: {got}  [{status}]")

    # number of offsets for a 3/4 cycle == 7
    n_off = len(range(3 + 4))
    print(f"  3/4 cycle offsets = {n_off}  [{'ok' if n_off == 7 else 'FAIL'}]")
    ok &= n_off == 7

    # config ordering matches the requested progression
    order = generate_config_groups(3, 3)[:6]
    exp_order = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3)]
    print(f"  group order      = {order}  [{'ok' if order == exp_order else 'FAIL'}]")
    ok &= order == exp_order

    # 2) vectorized trade returns vs a plain-Python reference
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 500))
    for buy, out, off in [(2, 1, 0), (3, 4, 2), (1, 5, 3), (5, 5, 4)]:
        n = len(close)
        daily = np.zeros(n)
        daily[1:] = close[1:] / close[:-1] - 1
        im = cycle_in_market(n, buy, out, off)
        strat = np.where(im, daily, 0.0)
        # reference: loop the ON blocks
        ref = []
        i = 0
        while i < n:
            if im[i]:
                comp, j = 1.0, i
                while j < n and im[j]:
                    comp *= 1 + strat[j]
                    j += 1
                ref.append((comp - 1) * 100)
                i = j
            else:
                i += 1
        m = run_cycle_backtest(close, buy, out, off)
        n_tr = _count_trades(n, buy, out, off)
        match = (n_tr == len(ref)) and (m is not None)
        print(
            f"  backtest {buy}/{out}/{off}: trades={n_tr} ref={len(ref)} "
            f"ret={m['Percent_Gain_%'] if m else 'None'}% "
            f"sharpe={m['Sharpe_Ratio'] if m else 'None'} "
            f"[{'ok' if match else 'FAIL'}]"
        )
        ok &= match

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ============================================================
# CLI
# ============================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Yahoo Finance calendar-cycle backtest sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers", default=str(DEFAULT_TICKERS), help="Ticker list file")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Results CSV path")
    p.add_argument("--max-buy", type=int, default=10, help="Max in-market length")
    p.add_argument("--max-out", type=int, default=10, help="Max out-of-market length")
    p.add_argument("--history-days", type=int, default=10 * 365,
                   help="Days of daily history to pull per ticker")
    p.add_argument("--port", type=int, default=8765, help="Dashboard port")
    p.add_argument("--no-ui", action="store_true", help="Disable the web dashboard")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore cached prices and re-download")
    p.add_argument("--pause", type=float, default=0.3,
                   help="Seconds to pause after each live download (be polite to Yahoo)")
    p.add_argument("--selftest", action="store_true",
                   help="Run offline correctness checks and exit")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    run_sweep(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
