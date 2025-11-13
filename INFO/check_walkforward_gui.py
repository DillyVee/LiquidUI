#!/usr/bin/env python3
"""
GUI WALK-FORWARD BUTTON DIAGNOSTIC

This script checks if your GUI has the Walk-Forward button properly configured.
"""

import os
import re

def check_walkforward_button():
    """Check if walk-forward button is in the GUI"""
    
    filepath = 'gui/main_window.py'
    
    if not os.path.exists(filepath):
        print("❌ File not found: gui/main_window.py")
        return False
    
    print("="*70)
    print("WALK-FORWARD BUTTON DIAGNOSTIC")
    print("="*70)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: Is the button created?
    has_button = 'walk_forward_btn' in content
    has_button_creation = 'self.walk_forward_btn = QPushButton' in content
    
    print(f"\n✓ Check 1: Button exists in code")
    print(f"   walk_forward_btn variable: {'✅ YES' if has_button else '❌ NO'}")
    print(f"   Button creation: {'✅ YES' if has_button_creation else '❌ NO'}")
    
    # Check 2: Is it added to the layout?
    has_layout_add = 'layout.addLayout(wf_layout)' in content or 'wf_layout.addWidget(self.walk_forward_btn)' in content
    
    print(f"\n✓ Check 2: Button added to layout")
    print(f"   Layout addition: {'✅ YES' if has_layout_add else '❌ NO'}")
    
    # Check 3: Does it have settings?
    has_train_days = 'wf_train_days_spin' in content
    has_test_days = 'wf_test_days_spin' in content
    has_trials = 'wf_trials_spin' in content
    
    print(f"\n✓ Check 3: Walk-Forward settings")
    print(f"   Train days spinner: {'✅ YES' if has_train_days else '❌ NO'}")
    print(f"   Test days spinner: {'✅ YES' if has_test_days else '❌ NO'}")
    print(f"   Trials spinner: {'✅ YES' if has_trials else '❌ NO'}")
    
    # Check 4: Is it enabled after data load?
    has_enable_on_load = 'self.walk_forward_btn.setEnabled(True)' in content
    
    print(f"\n✓ Check 4: Button gets enabled")
    print(f"   Enable on data load: {'✅ YES' if has_enable_on_load else '❌ NO'}")
    
    # Check 5: Find where it's located in the UI
    print(f"\n✓ Check 5: Button location in code")
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'self.walk_forward_btn = QPushButton' in line:
            print(f"   Found at line {i+1}")
            print(f"   Context:")
            for j in range(max(0, i-2), min(len(lines), i+8)):
                prefix = "   >>> " if j == i else "       "
                print(f"{prefix}{lines[j][:70]}")
            break
    
    # Check 6: Duplicate definitions?
    button_count = content.count('def _add_action_buttons')
    
    print(f"\n✓ Check 6: Duplicate method check")
    print(f"   _add_action_buttons defined: {button_count} time(s)")
    if button_count > 1:
        print(f"   ⚠️  WARNING: Multiple definitions found!")
        print(f"   This could cause the button to be missing or not work")
    
    # Summary
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    
    all_good = (has_button_creation and has_layout_add and has_train_days 
                and has_enable_on_load and button_count == 1)
    
    if all_good:
        print("\n✅ Walk-Forward button IS properly configured in the code!")
        print("\n📍 WHERE TO FIND IT:")
        print("   1. Run the app: python main.py")
        print("   2. Load data (enter AAPL, click 'Load from Yahoo Finance')")
        print("   3. Scroll down below the Monte Carlo button")
        print("   4. Look for: '📊 Run Walk-Forward Analysis'")
        print("   5. Button will be ENABLED (green) after data loads")
        print("\n   If you don't see it, try maximizing the window!")
        
    elif not has_button_creation:
        print("\n❌ Walk-Forward button is NOT in your GUI code!")
        print("\n   SOLUTION: Your version may be missing the button.")
        print("   I can add it for you. Let me know!")
        
    elif button_count > 1:
        print("\n⚠️  Multiple button definitions found!")
        print("\n   PROBLEM: Code has duplicate methods.")
        print("   SOLUTION: Need to remove duplicates.")
        
    else:
        print("\n⚠️  Walk-Forward button is partially configured")
        print("\n   Some components are missing. Let me know what's missing:")
        if not has_layout_add:
            print("   • Button not added to layout")
        if not has_train_days:
            print("   • Missing settings spinners")
        if not has_enable_on_load:
            print("   • Button doesn't get enabled")
    
    return all_good

if __name__ == "__main__":
    check_walkforward_button()
