from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_FORMAT = "mini-swe-agent-1.1"


class TrajectoryError(ValueError):
    pass


@dataclass(frozen=True)
class TrajectoryAction:
    step: int
    command: str
    recorded_stdout: str | None = None
    recorded_returncode: int | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ExecutionSettings:
    cwd: str = "/testbed"
    interpreter: tuple[str, ...] = ("bash", "-c")
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 300


@dataclass(frozen=True)
class ParsedTrajectory:
    format: str
    actions: tuple[TrajectoryAction, ...]
    settings: ExecutionSettings
    recorded_image: str | None = None


def load_trajectory(path: str | Path) -> ParsedTrajectory:
    trajectory_path = Path(path)
    try:
        data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrajectoryError(f"Trajectory does not exist: {trajectory_path}") from exc
    except json.JSONDecodeError as exc:
        raise TrajectoryError(f"Trajectory is not valid JSON: {exc}") from exc

    return parse_trajectory(data)


def parse_trajectory(data: Any) -> ParsedTrajectory:
    if not isinstance(data, dict):
        raise TrajectoryError("Trajectory root must be a JSON object")
    trajectory_format = data.get("trajectory_format")
    if trajectory_format != SUPPORTED_FORMAT:
        raise TrajectoryError(
            f"Unsupported trajectory format {trajectory_format!r}; expected {SUPPORTED_FORMAT!r}"
        )
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise TrajectoryError("Trajectory must contain a 'messages' array")

    settings = _parse_execution_settings(data)
    actions: list[TrajectoryAction] = []

    for message_index, message in enumerate(messages):
        raw_actions = _message_actions(message)
        if not raw_actions:
            continue

        segment_end = _next_action_message(messages, message_index + 1)
        observations = _observation_candidates(messages[message_index + 1 : segment_end])
        used_observations: set[int] = set()

        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                raise TrajectoryError(f"Action in message {message_index} must be an object")
            command = raw_action.get("command")
            if not isinstance(command, str) or not command.strip():
                raise TrajectoryError(
                    f"Action in message {message_index} must contain a non-empty string 'command'"
                )

            tool_call_id = raw_action.get("tool_call_id")
            if tool_call_id is not None and not isinstance(tool_call_id, str):
                raise TrajectoryError(f"Action tool_call_id in message {message_index} must be a string")

            observation_index = _match_observation(
                observations, used_observations, tool_call_id
            )
            observation = observations[observation_index] if observation_index is not None else None
            if observation_index is not None:
                used_observations.add(observation_index)

            stdout, returncode = _recorded_outcome(observation)
            actions.append(
                TrajectoryAction(
                    step=len(actions) + 1,
                    command=command,
                    recorded_stdout=stdout,
                    recorded_returncode=returncode,
                    tool_call_id=tool_call_id,
                )
            )

    if not actions:
        raise TrajectoryError("Trajectory contains no executable shell actions")
    info = data.get("info")
    config = info.get("config") if isinstance(info, dict) else None
    environment = config.get("environment") if isinstance(config, dict) else None
    recorded_image = environment.get("image") if isinstance(environment, dict) else None
    if not isinstance(recorded_image, str) or not recorded_image:
        recorded_image = None
    return ParsedTrajectory(trajectory_format, tuple(actions), settings, recorded_image)


def _message_actions(message: Any) -> list[Any]:
    if not isinstance(message, dict):
        return []
    extra = message.get("extra")
    if isinstance(extra, dict):
        actions = extra.get("actions")
        if isinstance(actions, list) and actions:
            return actions

    tool_calls = message.get("tool_calls")
    if message.get("role") != "assistant" or not isinstance(tool_calls, list):
        return []
    actions = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise TrajectoryError("Structured tool call must be an object")
        function = call.get("function")
        function_name = function.get("name") if isinstance(function, dict) else function
        if function_name != "bash":
            raise TrajectoryError(f"Unsupported tool call function {function_name!r}; expected 'bash'")
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise TrajectoryError(f"Invalid JSON tool call arguments: {exc}") from exc
        if not isinstance(arguments, dict):
            raise TrajectoryError("Structured bash tool call arguments must be an object")
        actions.append(
            {"command": arguments.get("command"), "tool_call_id": call.get("id") or call.get("call_id")}
        )
    return actions


def _next_action_message(messages: list[Any], start: int) -> int:
    for index in range(start, len(messages)):
        if _message_actions(messages[index]):
            return index
    return len(messages)


def _observation_candidates(messages: list[Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        extra = message.get("extra")
        if not isinstance(extra, dict):
            continue
        if any(key in extra for key in ("raw_output", "returncode", "exception_info")):
            candidates.append(message)
    return candidates


def _match_observation(
    observations: list[dict[str, Any]], used: set[int], tool_call_id: str | None
) -> int | None:
    if tool_call_id is not None:
        has_observation_ids = False
        for index, observation in enumerate(observations):
            if index in used:
                continue
            observed_id = observation.get("tool_call_id") or observation.get("call_id")
            has_observation_ids = has_observation_ids or observed_id is not None
            if observed_id == tool_call_id:
                return index
        if has_observation_ids:
            return None
    for index in range(len(observations)):
        if index not in used:
            return index
    return None


def _recorded_outcome(observation: dict[str, Any] | None) -> tuple[str | None, int | None]:
    if observation is None:
        return None, None
    extra = observation.get("extra")
    if not isinstance(extra, dict):
        return None, None
    stdout: str | None
    if "raw_output" not in extra:
        stdout = None
    else:
        value = extra.get("raw_output")
        stdout = "" if value is None else str(value)
    raw_returncode = extra.get("returncode")
    returncode = raw_returncode if isinstance(raw_returncode, int) and not isinstance(raw_returncode, bool) else None
    return stdout, returncode


def _parse_execution_settings(data: dict[str, Any]) -> ExecutionSettings:
    info = data.get("info")
    config = info.get("config") if isinstance(info, dict) else None
    environment = config.get("environment") if isinstance(config, dict) else None
    if not isinstance(environment, dict):
        environment = {}

    cwd = environment.get("cwd", "/testbed")
    if not isinstance(cwd, str) or not cwd:
        cwd = "/testbed"

    interpreter_raw = environment.get("interpreter", ["bash", "-c"])
    if (
        not isinstance(interpreter_raw, list)
        or not interpreter_raw
        or not all(isinstance(item, str) and item for item in interpreter_raw)
    ):
        interpreter = ("bash", "-c")
    else:
        interpreter = tuple(interpreter_raw)

    env_raw = environment.get("env", {})
    env = (
        {str(key): str(value) for key, value in env_raw.items()}
        if isinstance(env_raw, dict)
        else {}
    )

    timeout_raw = environment.get("timeout", 300)
    timeout = (
        timeout_raw
        if isinstance(timeout_raw, int) and not isinstance(timeout_raw, bool) and timeout_raw > 0
        else 300
    )
    return ExecutionSettings(cwd=cwd, interpreter=interpreter, env=env, timeout=timeout)
