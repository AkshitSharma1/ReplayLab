from __future__ import annotations

import unittest

from replaylab.trajectory import TrajectoryError, parse_trajectory


def trajectory(messages, **environment):
    return {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {"config": {"environment": environment}},
        "messages": messages,
    }


class TrajectoryParserTests(unittest.TestCase):
    def test_pairs_multiple_tool_calls_by_id(self):
        parsed = parse_trajectory(
            trajectory(
                [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "echo first", "tool_call_id": "one"},
                                {"command": "echo second", "tool_call_id": "two"},
                            ]
                        },
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "two",
                        "extra": {"raw_output": "second\n", "returncode": 0},
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "one",
                        "extra": {"raw_output": "first\n", "returncode": 0},
                    },
                ],
                cwd="/testbed",
                interpreter=["bash", "-c"],
                timeout=45,
                env={"PAGER": "cat"},
            )
        )
        self.assertEqual([a.command for a in parsed.actions], ["echo first", "echo second"])
        self.assertEqual([a.recorded_stdout for a in parsed.actions], ["first\n", "second\n"])
        self.assertEqual(parsed.settings.timeout, 45)
        self.assertEqual(parsed.settings.env, {"PAGER": "cat"})

    def test_preserves_recorded_image_for_replay_identity_check(self):
        parsed = parse_trajectory(
            trajectory(
                [{"role": "assistant", "extra": {"actions": [{"command": "pwd"}]}}],
                image="docker.io/swebench/example:latest",
            )
        )
        self.assertEqual(parsed.recorded_image, "docker.io/swebench/example:latest")

    def test_pairs_text_action_observation_positionally(self):
        parsed = parse_trajectory(
            trajectory(
                [
                    {"role": "assistant", "extra": {"actions": [{"command": "pwd"}]}},
                    {"role": "user", "extra": {"raw_output": "/testbed\n", "returncode": 0}},
                ]
            )
        )
        self.assertEqual(parsed.actions[0].recorded_stdout, "/testbed\n")
        self.assertIsNone(parsed.actions[0].tool_call_id)

    def test_unmatched_tool_id_does_not_pair_unrelated_observation(self):
        parsed = parse_trajectory(
            trajectory([
                {"role": "assistant", "extra": {"actions": [
                    {"command": "pwd", "tool_call_id": "wanted"}
                ]}},
                {"role": "tool", "tool_call_id": "other", "extra": {
                    "raw_output": "wrong output", "returncode": 0
                }},
            ])
        )
        self.assertIsNone(parsed.actions[0].recorded_stdout)

    def test_missing_observation_is_replayable(self):
        parsed = parse_trajectory(
            trajectory([{"role": "assistant", "extra": {"actions": [{"command": "true"}]}}])
        )
        self.assertIsNone(parsed.actions[0].recorded_stdout)
        self.assertIsNone(parsed.actions[0].recorded_returncode)

    def test_structured_tool_call_export_without_extra_metadata(self):
        parsed = parse_trajectory(
            trajectory(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call_a", "function": "bash", "arguments": {"command": "ls -la"}},
                            {"id": "call_b", "function": "bash", "arguments": '{"command":"pwd"}'},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_a", "content": "rendered output"},
                    {"role": "tool", "tool_call_id": "call_b", "content": "rendered output"},
                ]
            )
        )
        self.assertEqual([action.command for action in parsed.actions], ["ls -la", "pwd"])
        self.assertTrue(all(action.recorded_stdout is None for action in parsed.actions))

    def test_rejects_non_bash_structured_tool_call(self):
        with self.assertRaisesRegex(TrajectoryError, "Unsupported tool call"):
            parse_trajectory(
                trajectory(
                    [{"role": "assistant", "tool_calls": [
                        {"function": "python", "arguments": {"command": "print(1)"}}
                    ]}]
                )
            )

    def test_rejects_unsupported_format(self):
        with self.assertRaisesRegex(TrajectoryError, "Unsupported trajectory format"):
            parse_trajectory({"trajectory_format": "mini-swe-agent-1", "messages": []})

    def test_rejects_malformed_command(self):
        with self.assertRaisesRegex(TrajectoryError, "non-empty"):
            parse_trajectory(
                trajectory([{"role": "assistant", "extra": {"actions": [{"command": ""}]}}])
            )

    def test_rejects_empty_trajectory(self):
        with self.assertRaisesRegex(TrajectoryError, "no executable"):
            parse_trajectory(trajectory([{"role": "system", "content": "hello"}]))


if __name__ == "__main__":
    unittest.main()
