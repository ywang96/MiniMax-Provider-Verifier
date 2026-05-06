import json

from jsonschema import validate, ValidationError

from validator import BaseValidator

# Common command prefixes that indicate a merged argument when followed by a space
_COMMON_COMMANDS = [
    "ls ", "cat ", "git ", "npm ", "npx ", "cd ", "cp ", "mv ", "rm ",
    "mkdir ", "chmod ", "chown ", "find ", "grep ", "curl ", "wget ",
    "pip ",
]


def _is_shell_c_invocation(cmd: list) -> bool:
    """Check if cmd is a shell -c invocation (e.g. bash -lc '...', sh -c '...')."""
    if not cmd or len(cmd) < 3:
        return False
    shell = cmd[0]
    if shell not in ("bash", "sh", "zsh", "/bin/bash", "/bin/sh", "/bin/zsh",
                     "/usr/bin/bash", "/usr/bin/sh", "/usr/bin/zsh"):
        return False
    # Match patterns: ["bash", "-c", "..."], ["bash", "-lc", "..."],
    # ["bash", "-l", "-c", "..."], ["bash", "--login", "-c", "..."]
    for i, arg in enumerate(cmd[1:], start=1):
        if arg in ("-c", "-lc"):
            return True
        if arg in ("-l", "--login"):
            continue
        break
    return False


def is_valid_array_command(cmd) -> bool:
    """Validate that an array command has properly separated arguments.

    Returns False when the model merges multiple arguments into a single
    string (e.g. ["ls -la /workspace"]) which would fail at execvp() time.
    Shell -c invocations are exempted because the script string after -c
    is expected to contain spaces.
    """
    if not isinstance(cmd, list) or len(cmd) == 0:
        return False

    if _is_shell_c_invocation(cmd):
        return True

    for elem in cmd:
        if not isinstance(elem, str):
            return False
        if " " in elem:
            # Check if it starts with a common command prefix
            for prefix in _COMMON_COMMANDS:
                if elem.startswith(prefix):
                    return False

    # Single-element array with spaces is suspicious
    if len(cmd) == 1 and " " in cmd[0]:
        return False

    return True


class ToolCallsValidator(BaseValidator):
    """Validator for tool calls schema validation."""

    @property
    def name(self) -> str:
        return "tool_calls"

    def validate(self, request: dict, response: dict, status: str, resp_content: str = None) -> dict:
        """Validate tool calls in the response."""
        result = {
            "tool_calls_finish_reason": None,
            "tool_calls_valid": None,
            "tool_calls_count": 0,
        }

        if status != "success" or not response or "choices" not in response:
            return result

        choice = response["choices"][0] if response["choices"] else {}
        finish_reason = choice.get("finish_reason")
        result["tool_calls_finish_reason"] = finish_reason

        if finish_reason == "tool_calls":
            tools = request.get("tools", [])
            tool_calls = choice.get("message", {}).get("tool_calls", [])
            result["tool_calls_count"] = len(tool_calls)

            if tool_calls:
                result["tool_calls_valid"] = all(
                    self.validate_tool_call(tc, tools) for tc in tool_calls
                )
            else:
                result["tool_calls_valid"] = False

        return result

    def compute_summary(self, results: list[dict]) -> dict:
        """Compute summary statistics for tool calls validation."""
        summary = {
            # Confusion matrix: expected vs actual
            "tool_calls_finish_tool_calls": 0,  # expected tool_call, actual tool_call (TP)
            "tool_calls_finish_stop": 0,         # expected tool_call, actual stop (FN)
            "stop_finish_tool_calls": 0,         # expected stop, actual tool_call (FP)
            "stop_finish_stop": 0,               # expected stop, actual stop (TN)
            "tool_calls_finish_others": 0,
            "tool_calls_finish_others_detail": {},
            # Expected tool call count (denominator for match rate)
            "expected_tool_call_total_count": 0,  # Total labeled count (True + False)
            # Schema validation stats
            "tool_calls_schema_validation_error_count": 0,
            "tool_calls_successful_count": 0,
            "tool_calls_total_count": 0,
            "tool_calls_count_distribution": {}, 
        }

        for r in results:
            finish_reason = r.get("tool_calls_finish_reason")
            tool_calls_valid = r.get("tool_calls_valid")
            tool_calls_count = r.get("tool_calls_count", 0)
            expected_tool_call = r.get("expected_tool_call")

            summary["tool_calls_total_count"] += tool_calls_count

            # Count expected_tool_call labels
            if expected_tool_call is True or expected_tool_call is False:
                summary["expected_tool_call_total_count"] += 1

            # Classify into confusion matrix based on expected and actual
            actual_tool_call = (finish_reason == "tool_calls")
            actual_stop = (finish_reason == "stop")

            if expected_tool_call is True:
                if actual_tool_call:
                    summary["tool_calls_finish_tool_calls"] += 1
                    if tool_calls_valid:
                        summary["tool_calls_successful_count"] += 1
                    else:
                        summary["tool_calls_schema_validation_error_count"] += 1
                    count_key = str(tool_calls_count)
                    summary["tool_calls_count_distribution"].setdefault(count_key, 0)
                    summary["tool_calls_count_distribution"][count_key] += 1
                elif actual_stop:
                    summary["tool_calls_finish_stop"] += 1
                elif finish_reason:
                    summary["tool_calls_finish_others"] += 1
                    summary["tool_calls_finish_others_detail"].setdefault(finish_reason, 0)
                    summary["tool_calls_finish_others_detail"][finish_reason] += 1
            elif expected_tool_call is False:
                if actual_tool_call:
                    summary["stop_finish_tool_calls"] += 1
                elif actual_stop:
                    summary["stop_finish_stop"] += 1
                elif finish_reason:
                    summary["tool_calls_finish_others"] += 1
                    summary["tool_calls_finish_others_detail"].setdefault(finish_reason, 0)
                    summary["tool_calls_finish_others_detail"][finish_reason] += 1
            else:
                # expected_tool_call is None — skip, not counted in confusion matrix
                pass

        return summary

    def validate_tool_call(self, tool_call: dict, tools: list[dict]) -> bool:
        """Validate tool call arguments against schema."""
        try:
            tool_name = tool_call["function"]["name"]
            schema = next(
                (
                    t["function"]["parameters"]
                    for t in tools
                    if t["function"]["name"] == tool_name
                ),
                None,
            )
            if not schema:
                print(f"No schema for tool {tool_name}")
                return False
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            validate(instance=args, schema=schema)

            # Additional: validate array command format
            for param_name, param_schema in schema.get("properties", {}).items():
                if (param_name == "command"
                        and param_schema.get("type") == "array"
                        and param_schema.get("items", {}).get("type") == "string"):
                    cmd_value = args.get(param_name)
                    if cmd_value is not None and not is_valid_array_command(cmd_value):
                        print(f"Array command format validation failed for {tool_name}.{param_name}: {cmd_value}")
                        return False

            return True
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Schema validation failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected validation error: {e}")
            return False
