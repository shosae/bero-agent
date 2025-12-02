"""Alias package to expose robot services under app.robot.services.*.

This keeps existing app.services.* modules as the single source of truth,
while allowing imports like `app.robot.services.executor_service`.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_MODULES = [
    "classify_intent_service",
    "conversation_service",
    "executor_service",
    "llm_service",
    "mission_summary_service",
    "navigation_service",
    "planner_service",
    "rag_service",
    "robot_grpc_client",
    "validator_llm_service",
    "validator_rule_service",
    "validator_service",
    "vectorstore_utils",
]

pkg_prefix = "app.services"
alias_prefix = "app.robot.services"

# Re-export modules under app.robot.services.*
for mod in _MODULES:
    target_name = f"{pkg_prefix}.{mod}"
    alias_name = f"{alias_prefix}.{mod}"
    target_mod: ModuleType = importlib.import_module(target_name)
    sys.modules[alias_name] = target_mod

# Also allow `import app.robot.services as ...`
sys.modules[alias_prefix] = sys.modules[__name__]

__all__ = _MODULES
