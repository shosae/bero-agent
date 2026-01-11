from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3] # RLN/Bero


def project_root() -> Path:
    return _PROJECT_ROOT


def artifacts_dir() -> Path:
    return _PROJECT_ROOT / "artifacts"


def auth_dir() -> Path:
    return _PROJECT_ROOT / "auth"


def grpc_dir() -> Path:
    return _PROJECT_ROOT / "grpc"


def robot_resources_dir() -> Path:
    return _PROJECT_ROOT / "src" / "app" / "modules" / "robot" / "resources"
