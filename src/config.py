from __future__ import annotations

import os
from typing import Optional

def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is not None:
        v = v.strip()
    return v or None

def get_secret(name: str, *, default: Optional[str] = None, required: bool = True) -> str:
    """
    Priority:
      1) Environment variables
      2) Streamlit secrets (if running under Streamlit)
      3) .env file via python-dotenv (if installed and .env exists)
    """
    # 1) env
    v = _get_env(name)
    if v:
        return v

    # 2) streamlit secrets (only if streamlit is installed + running)
    try:
        import streamlit as st  # type: ignore
        try:
            v2 = st.secrets.get(name)  # works even if not deployed, if secrets.toml exists
            if v2:
                return str(v2).strip()
        except Exception:
            pass
    except Exception:
        pass

    # 3) dotenv (optional)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        v3 = _get_env(name)
        if v3:
            return v3
    except Exception:
        pass

    if default is not None:
        return default

    if required:
        raise RuntimeError(
            f"Missing required secret '{name}'. "
            f"Set it as an environment variable, in .env, or in .streamlit/secrets.toml."
        )

    return ""
