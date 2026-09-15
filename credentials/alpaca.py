"""Load shared Alpaca credentials from credentials/.env."""

import os
from pathlib import Path


ENV_FILE = Path(__file__).with_name(".env")


def _load_env_file():
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name.strip(), value)


def get_alpaca_credentials():
    """Return Alpaca credentials shared by every strategy."""
    _load_env_file()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    missing = [
        name
        for name, value in (
            ("ALPACA_API_KEY", api_key),
            ("ALPACA_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Alpaca credentials: "
            + ", ".join(missing)
            + ". Copy credentials/.env.example to credentials/.env and fill it in."
        )

    return api_key, secret_key, base_url
