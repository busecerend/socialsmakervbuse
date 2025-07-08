#!/usr/bin/env python3
"""
Main entry point for the Meme Analyzer project.
"""

from scripts.analyze_meme import main as run_analysis

def start_application():
    print("🎭 Initializing Meme Analyzer application...")
    print("-" * 40)
    run_analysis()
    print("-" * 40)
    print("✅ Meme Analyzer application finished.")

if __name__ == "__main__":
    start_application()
