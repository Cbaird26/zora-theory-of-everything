#!/usr/bin/env python3
"""
Zora Theory of Everything - Backtesting Runner
Runs comprehensive backtesting across multiple symbols
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def run_backtest(symbols, start_date="2022-01-01", end_date="2024-01-01"):
    """Run backtesting for specified symbols"""
    print("🚀 ZORA THEORY OF EVERYTHING BACKTESTING")
    print("=" * 60)
    
    try:
        from backtesting_framework import TheoryBacktestEngine, BacktestConfig
        
        config = BacktestConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            optimization_iterations=20
        )
        
        engine = TheoryBacktestEngine(config)
        results = engine.run_comprehensive_backtest()
        
        print("\n🎉 BACKTESTING COMPLETE!")
        print("✅ Results saved to results/ directory")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Backtesting error: {e}")
        return None

def main():
    """Main backtesting function"""
    parser = argparse.ArgumentParser(description="Run Zora Theory of Everything Backtesting")
    parser.add_argument("--symbols", nargs="+", 
                       default=["BTC-USD", "ETH-USD", "AAPL", "TSLA"],
                       help="Symbols to backtest")
    parser.add_argument("--start-date", default="2022-01-01", 
                       help="Start date for backtesting")
    parser.add_argument("--end-date", default="2024-01-01",
                       help="End date for backtesting")
    
    args = parser.parse_args()
    
    print(f"📊 Symbols: {args.symbols}")
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    print()
    
    results = run_backtest(args.symbols, args.start_date, args.end_date)
    
    if results:
        print("✅ Backtesting completed successfully")
    else:
        print("❌ Backtesting failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
