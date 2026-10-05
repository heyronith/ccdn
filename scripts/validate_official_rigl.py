"""Compatibility entry point for the pinned RigL parity validation."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("validate_rigl_parity.py")),
               run_name="__main__")
