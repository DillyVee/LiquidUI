"""
Get Top 10 Stocks by Market Cap from Each Sector

This script fetches S&P 500 stocks and returns the top 10 by market cap 
from each GICS sector, saving them to a ticker list file.

Usage:
    python get_top_stocks_by_sector.py [--output filename.txt] [--per-sector N]
"""

import yfinance as yf
import pandas as pd
import argparse
from datetime import datetime
import time

def get_sp500_tickers():
    """Get list of S&P 500 tickers from Wikipedia"""
    print("📥 Fetching S&P 500 ticker list...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # Extract ticker and sector
        tickers_df = sp500_table[['Symbol', 'GICS Sector']].copy()
        tickers_df.columns = ['Ticker', 'Sector']
        
        # Clean tickers (some have periods instead of hyphens)
        tickers_df['Ticker'] = tickers_df['Ticker'].str.replace('.', '-', regex=False)
        
        print(f"✅ Found {len(tickers_df)} S&P 500 stocks")
        print(f"📊 Sectors: {tickers_df['Sector'].nunique()}")
        return tickers_df
    
    except Exception as e:
        print(f"❌ Error fetching S&P 500 list: {e}")
        return None

def get_market_caps(tickers):
    """Fetch market cap for each ticker"""
    print(f"\n💰 Fetching market caps for {len(tickers)} tickers...")
    
    results = []
    failed = []
    
    for idx, ticker in enumerate(tickers):
        if (idx + 1) % 50 == 0:
            print(f"   Progress: {idx + 1}/{len(tickers)}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try different market cap fields
            market_cap = info.get('marketCap') or info.get('market_cap')
            
            if market_cap and market_cap > 0:
                results.append({
                    'Ticker': ticker,
                    'MarketCap': market_cap
                })
            else:
                failed.append(ticker)
                
        except Exception as e:
            failed.append(ticker)
            continue
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    print(f"\n✅ Successfully fetched {len(results)} market caps")
    if failed:
        print(f"⚠️  Failed to fetch {len(failed)} tickers: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    
    return pd.DataFrame(results)

def get_top_stocks_by_sector(per_sector=10):
    """Get top N stocks by market cap from each sector"""
    
    # Get S&P 500 list with sectors
    sp500_df = get_sp500_tickers()
    if sp500_df is None:
        return None
    
    # Get market caps
    market_cap_df = get_market_caps(sp500_df['Ticker'].tolist())
    if market_cap_df.empty:
        print("❌ No market cap data retrieved")
        return None
    
    # Merge with sector information
    df = sp500_df.merge(market_cap_df, on='Ticker')
    
    print(f"\n📊 Processing {len(df)} stocks across sectors...")
    
    # Group by sector and get top N
    top_stocks = []
    sector_summary = []
    
    for sector in sorted(df['Sector'].unique()):
        sector_df = df[df['Sector'] == sector].copy()
        sector_df = sector_df.sort_values('MarketCap', ascending=False).head(per_sector)
        
        sector_summary.append({
            'Sector': sector,
            'Count': len(sector_df),
            'Total_Market_Cap': sector_df['MarketCap'].sum(),
            'Top_Stock': sector_df.iloc[0]['Ticker'] if len(sector_df) > 0 else 'N/A'
        })
        
        top_stocks.append(sector_df)
        
        print(f"\n{sector}:")
        print(f"  Top {len(sector_df)} stocks:")
        for _, row in sector_df.iterrows():
            print(f"    {row['Ticker']:6s} - ${row['MarketCap']/1e9:>8.2f}B")
    
    # Combine all
    result_df = pd.concat(top_stocks, ignore_index=True)
    summary_df = pd.DataFrame(sector_summary)
    
    return result_df, summary_df

def save_ticker_list(df, output_file, include_market_cap=True):
    """Save tickers to file"""
    print(f"\n💾 Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# Top Stocks by Market Cap per Sector\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total stocks: {len(df)}\n")
        f.write(f"# Format: TICKER,MARKET_CAP\n")
        f.write(f"#\n")
        
        # Group by sector for organized output
        for sector in sorted(df['Sector'].unique()):
            sector_df = df[df['Sector'] == sector].sort_values('MarketCap', ascending=False)
            
            f.write(f"\n# {sector} ({len(sector_df)} stocks)\n")
            
            for _, row in sector_df.iterrows():
                if include_market_cap:
                    f.write(f"{row['Ticker']},{int(row['MarketCap'])}\n")
                else:
                    f.write(f"{row['Ticker']}\n")
    
    print(f"✅ Saved {len(df)} tickers to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Get top stocks by market cap from each sector'
    )
    parser.add_argument(
        '--output', '-o',
        default='top_stocks_by_sector.txt',
        help='Output filename (default: top_stocks_by_sector.txt)'
    )
    parser.add_argument(
        '--per-sector', '-n',
        type=int,
        default=10,
        help='Number of stocks per sector (default: 10)'
    )
    parser.add_argument(
        '--no-market-cap',
        action='store_true',
        help='Exclude market cap from output (tickers only)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("TOP STOCKS BY SECTOR")
    print("="*70)
    print(f"Fetching top {args.per_sector} stocks per sector...")
    print()
    
    # Get the data
    result_df, summary_df = get_top_stocks_by_sector(per_sector=args.per_sector)
    
    if result_df is None or result_df.empty:
        print("❌ Failed to retrieve stock data")
        return
    
    # Save to file
    save_ticker_list(result_df, args.output, include_market_cap=not args.no_market_cap)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY BY SECTOR")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"Total stocks selected: {len(result_df)}")
    print(f"Total market cap: ${result_df['MarketCap'].sum()/1e12:.2f}T")
    print(f"Output file: {args.output}")
    print("="*70)

if __name__ == "__main__":
    main()