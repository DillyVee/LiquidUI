"""
Fetch the real-market daily series used by the research experiments.

Downloads FRED CSVs into research/data/ (gitignored - the SP500 series
is licensed to FRED by S&P, so snapshots are not committed; re-fetch to
reproduce). Each file is observation_date,VALUE with "." for missing.

Usage:  python research/fetch_data.py
"""

import os
import urllib.request

SERIES = ("SP500", "NASDAQCOM", "CBBTCUSD")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for series in SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        out = os.path.join(DATA_DIR, f"{series}.csv")
        print(f"{series}: {url}")
        urllib.request.urlretrieve(url, out)
        with open(out) as f:
            n = sum(1 for _ in f)
        print(f"  -> {out} ({n} rows)")


if __name__ == "__main__":
    main()
