#!/usr/bin/env python3
"""Main entrypoint: python scripts/scan.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scanner.pipeline import run

if __name__ == "__main__":
    sys.exit(run())
