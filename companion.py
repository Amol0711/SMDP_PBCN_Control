#!/usr/bin/env python3
"""Computational companion. Explicit commands, immutable inputs, external outputs."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_smac.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
