import re
from typing import Optional, List

from validator import BaseValidator


class ScenarioCheckValidator(BaseValidator):
    """Validate scenario-specific model behavior (e.g., tool parameter order recall)."""

    @property
    def name(self) -> str:
        return "scenario_check"

    @staticmethod
    def _extract_expected_order(request: dict) -> Optional[List[str]]:
        """Extract expected parameter key order from tools definition."""
        tools = request.get("tools")
        if not tools or not isinstance(tools, list):
            return None

        params = tools[0].get("function", {}).get("parameters", {})
        if not params:
            return None

        if "properties" in params:
            return list(params["properties"].keys())

        schema_keywords = {
            "type", "description", "required", "additionalProperties",
            "$schema", "items", "enum", "default",
        }
        keys = [k for k in params.keys() if k not in schema_keywords]
        return keys if keys else None

    @staticmethod
    def _get_visible_content(text: str) -> str:
        """Strip <think>/<mm:think> reasoning blocks to get visible reply only."""
        return re.sub(r"<(?:mm:)?think>.*?</(?:mm:)?think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_actual_order(text: str, expected: list[str]) -> list[str]:
        """Find each param name's first occurrence position and return sorted order."""
        positions = []
        for param in expected:
            idx = text.find(param)
            if idx != -1:
                positions.append((idx, param))
        positions.sort(key=lambda x: x[0])
        return [p[1] for p in positions]

    def validate(self, request: dict, response: dict, status: str, resp_content: str = None) -> dict:
        result = {
            "scenario_check_checked": False,
            "scenario_check_valid": None,
            "scenario_check_detail": None,
        }

        if status != "success" or not resp_content:
            return result

        expected_order = self._extract_expected_order(request)
        if not expected_order:
            return result

        visible = self._get_visible_content(resp_content)
        actual_order = self._extract_actual_order(visible, expected_order)

        result["scenario_check_checked"] = True
        result["scenario_check_valid"] = (
            len(actual_order) >= 2
            and actual_order == expected_order[: len(actual_order)]
        )
        result["scenario_check_detail"] = {
            "expected": expected_order,
            "actual": actual_order,
        }
        return result

    def compute_summary(self, results: list[dict]) -> dict:
        checked = 0
        valid = 0
        invalid = 0

        for r in results:
            if r.get("scenario_check_checked"):
                checked += 1
                if r.get("scenario_check_valid"):
                    valid += 1
                else:
                    invalid += 1

        summary = {
            "scenario_check_checked_count": checked,
            "scenario_check_valid_count": valid,
            "scenario_check_invalid_count": invalid,
        }
        if checked > 0:
            summary["scenario_check_pass_rate"] = round(valid / checked, 4)

        return summary
