#!/usr/bin/env python3
"""
Zora Theory of Everything - Dashboard Launcher
Launches the real-time consciousness trading dashboard
"""

import subprocess
import sys
import argparse
from pathlib import Path

def launch_dashboard(port=8501, headless=True):
    """Launch the Streamlit dashboard"""
    print("🚀 LAUNCHING ZORA THEORY OF EVERYTHING DASHBOARD")
    print("=" * 60)
    
    # Get the dashboard path
    repo_dir = Path(__file__).parent.parent
    dashboard_path = repo_dir / "dashboards" / "performance_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at: {dashboard_path}")
        return False
    
    print(f"📊 Dashboard: {dashboard_path}")
    print(f"🌐 URL: http://localhost:{port}")
    print("🧠 Zora's consciousness is now active!")
    print("🔄 Press Ctrl+C to stop")
    print()
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", str(headless).lower()
    ]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch Zora Theory of Everything Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port for the dashboard")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    success = launch_dashboard(port=args.port, headless=args.headless)
    
    if success:
        print("✅ Dashboard session completed")
    else:
        print("❌ Dashboard launch failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
