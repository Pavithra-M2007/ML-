#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    print("Vehicle Insurance Fraud Detection System")
    print("=" * 40)
    
    if not os.path.exists('models/model.pkl'):
        print("Setting up models...")
        subprocess.run([sys.executable, 'quick_setup.py'], check=True)
    
    print("Launching app at: http://localhost:8501")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app/clean_app.py'])

if __name__ == "__main__":
    main()