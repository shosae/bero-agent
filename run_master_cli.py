#!/usr/bin/env python
"""CLI entry to run the Master Agent tool-calling loop."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


def _prepare_env() -> None:
    root = Path(__file__).resolve().parent
    for env_path in (root / ".env", root / "app" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    src_dir = root / "app" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main(argv: list[str]) -> int:
    _prepare_env()
    from app.master.agent import run_master  # pylint: disable=import-error

    if len(argv) < 2:
        print("Usage: python run_master_cli.py \"<question>\"")
        return 1
    question = argv[1]
    answer = run_master(question)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
