"""Backward-compatible import shim. Prefer llm_survey.utils.preprocess."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from llm_survey.utils.preprocess import *  # noqa: F401,F403