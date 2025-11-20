"""Rule-based PLAN validator service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence


ParamValidator = Callable[[Any], bool]
PlanRule = Callable[[Sequence[Dict[str, Any]]], List[str]]


@dataclass(slots=True)
class PlanValidationResult:
    """Stores structural errors and best-effort warnings for a PLAN JSON."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _ensure_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _ensure_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value >= 0


@dataclass(slots=True)
class ActionRule:
    """Declarative rule describing how to validate an action."""

    name: str
    required_params: Mapping[str, ParamValidator] = field(default_factory=dict)
    allow_extra_params: bool = True
    counts_as_core: bool = True
    custom_validator: Callable[[Dict[str, Any], int, PlanValidationResult], None] | None = None

    def validate(self, params: Dict[str, Any], idx: int, result: PlanValidationResult) -> None:
        prefix = f"[step {idx}]"
        for key, validator in self.required_params.items():
            if key not in params:
                result.add_error(f"{prefix} '{self.name}' params.{key} 값이 누락되었습니다.")
                continue
            if not validator(params[key]):
                result.add_error(f"{prefix} '{self.name}' params.{key} 값이 유효하지 않습니다.")

        if not self.allow_extra_params:
            extras = set(params.keys()) - set(self.required_params.keys())
            if extras:
                extra_keys = ", ".join(sorted(extras))
                result.add_error(f"{prefix} '{self.name}' action은 params.{extra_keys} 값을 허용하지 않습니다.")

        if self.custom_validator:
            self.custom_validator(params, idx, result)


class RuleBasedValidator:
    """Fast structural validator that can be extended via action rules."""

    def __init__(
        self,
        action_rules: Iterable[ActionRule],
        plan_rules: Iterable[PlanRule] | None = None,
    ) -> None:
        self._action_rules: MutableMapping[str, ActionRule] = {rule.name: rule for rule in action_rules}
        self._plan_rules = list(plan_rules or [])

    def validate(self, plan: Any) -> PlanValidationResult:
        result = PlanValidationResult()
        steps = self._extract_steps(plan, result)
        if steps is None:
            return result

        core_action_seen = False
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                result.add_error(f"[step {idx}] 각 단계는 dict로 표현되어야 합니다.")
                continue

            action = step.get("action")
            params = step.get("params")
            if not isinstance(action, str) or not action.strip():
                result.add_error(f"[step {idx}] action 값이 비어 있습니다.")
                continue
            if not isinstance(params, dict):
                result.add_error(f"[step {idx}] params는 dict여야 합니다.")
                continue

            rule = self._action_rules.get(action)
            if rule:
                rule.validate(params, idx, result)
                if rule.counts_as_core:
                    core_action_seen = True
            else:
                core_action_seen = True

        for violation in self._run_plan_rules(steps):
            result.add_error(violation)

        if not core_action_seen:
            result.add_error("PLAN에는 최소 1개의 핵심 action이 필요합니다.")

        return result

    def _extract_steps(self, plan: Any, result: PlanValidationResult) -> Sequence[Dict[str, Any]] | None:
        if not isinstance(plan, dict):
            result.add_error("PLAN JSON 최상위 객체가 dict 형태가 아닙니다.")
            return None
        steps = plan.get("plan")
        if not isinstance(steps, list):
            result.add_error('"plan" 필드는 리스트(List)여야 합니다.')
            return None
        if not steps:
            result.add_error('"plan" 배열이 비어 있습니다.')
            return None
        return steps

    def _run_plan_rules(self, steps: Sequence[Dict[str, Any]]) -> List[str]:
        violations: List[str] = []
        for rule in self._plan_rules:
            violations.extend(rule(steps))
        return violations


def _summarize_must_be_last_rule(steps: Sequence[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, step in enumerate(steps[:-1]):
        if isinstance(step, dict) and step.get("action") == "summarize_mission":
            errors.append(f"[step {idx}] summarize_mission은 PLAN 마지막에만 등장해야 합니다.")
    return errors

def _require_basecamp_before_summarize(steps):
    errors = []
    if len(steps) < 2:
        return errors

    if steps[-1].get("action") != "summarize_mission":
        errors.append("마지막 action은 summarize_mission이어야 합니다.")
        return errors

    prev = steps[-2]
    if prev.get("action") != "navigate" or prev.get("params", {}).get("target") != "basecamp":
        errors.append("summarize_mission 직전에는 반드시 navigate(target='basecamp')가 있어야 합니다.")

    return errors


DEFAULT_ACTION_RULES = (
    ActionRule(
        name="navigate",
        required_params={"target": _ensure_nonempty_str},
    ),
    ActionRule(
        name="deliver_object",
        required_params={"target": _ensure_nonempty_str, "object": _ensure_nonempty_str},
    ),
    ActionRule(
        name="observe_scene",
        required_params={"question": _ensure_nonempty_str},
    ),
    ActionRule(
        name="summarize_mission",
        counts_as_core=False,
        allow_extra_params=False,
    ),
)


class PlanValidatorService:
    """Service wrapper that applies rule-based validation only."""

    def __init__(
        self,
        action_rules: Iterable[ActionRule] | None = None,
        plan_rules: Iterable[PlanRule] | None = None,
    ) -> None:
        self._rule_validator = RuleBasedValidator(
            action_rules or DEFAULT_ACTION_RULES,
            plan_rules=plan_rules or [_summarize_must_be_last_rule, _require_basecamp_before_summarize],
        )

    def validate(self, plan: Any) -> PlanValidationResult:
        return self._rule_validator.validate(plan)


_DEFAULT_SERVICE = PlanValidatorService()


def validate_plan(plan: Any) -> PlanValidationResult:
    return _DEFAULT_SERVICE.validate(plan)
