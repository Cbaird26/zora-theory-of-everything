#!/usr/bin/env python3
"""
Zora Theory of Everything - Setup Script
Installs dependencies and initializes the consciousness system
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔧 Installing Zora Theory of Everything dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "logs",
        "data",
        "results",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        import numpy
        import pandas
        import streamlit
        import yfinance
        print("✅ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ZORA THEORY OF EVERYTHING - SETUP")
    print("=" * 50)
    
    # Change to repository directory
    repo_dir = Path(__file__).parent.parent
    os.chdir(repo_dir)
    
    # Install dependencies
    if not install_requirements():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        return False
    
    print("\n🎉 SETUP COMPLETE!")
    print("✅ Zora Theory of Everything is ready to use")
    print("🚀 Run: python scripts/launch_dashboard.py")
    
    return True

if __name__ == "__main__":
    main()
