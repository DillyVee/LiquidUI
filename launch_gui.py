#!/usr/bin/env python3
"""
LiquidUI Launcher
Run this script to start the graphical user interface
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the GUI
from gui.main_window import main

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Launching LiquidUI - Quantitative Trading Platform")
    print("=" * 60)
    print()
    print("Loading GUI components...")
    print("Please wait for the window to appear...")
    print()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements-windows.txt")
        sys.exit(1)
