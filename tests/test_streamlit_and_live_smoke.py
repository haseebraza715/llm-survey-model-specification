"""UI import smoke (no `streamlit run`) and optional live OpenRouter checks."""

from __future__ import annotations

import os

import pytest


def test_app_module_imports_without_executing_main() -> None:
    """
    Ensures `app.py` resolves `src/` on sys.path and `ui.dashboard` loads.
    Does not call `main()` (no Streamlit server or script run).
    """
    import app as app_module
    from ui import dashboard

    assert app_module.__doc__
    assert callable(dashboard.main)


@pytest.mark.live_api
@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_openrouter_chat_completion_smoke() -> None:
    """One tiny completion to verify key, base URL, and routing (billable)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": 'Reply with the single word "pong" and nothing else.'}],
        max_tokens=8,
        temperature=0,
    )
    text = (completion.choices[0].message.content or "").strip().lower()
    assert "pong" in text
