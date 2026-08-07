import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
# Repository root first so top-level modules (`app`, `main`) resolve like CI's
# `PYTHONPATH=.:src`, then `src/` for the `llm_survey` package.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
