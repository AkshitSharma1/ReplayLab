# ReplayLab

ReplayLab is a lightweight audit tool that replays coding-agent shell trajectories inside SWE-bench environments and shows exactly how the repository changes after every action.
<hr style="height: 0.1px; background-color: black; border: none;">
I built ReplayLab because coding-agent trajectories often show what an agent *said* it did without making it easy to see what actually changed inside the repository.

An agent may inspect a source file, run a failing test, edit the implementation, modify a test, and then report success. When a trajectory contains dozens of shell actions and large command outputs, it becomes difficult to tell whether the agent fixed the underlying problem or simply changed the evaluation around it.

ReplayLab makes this behavior explicit. It replays the recorded shell commands inside the corresponding SWE-bench environment and captures the repository state after every action. The final report shows which files changed, the exact diffs, the command outcomes, and whether the agent touched tests or test configuration.

No LLM is used for this analysis. The results come directly from replaying the commands and observing the environment.

## Why ReplayLab is useful

Suppose a trajectory looks like this:

```text
1. Inspect the implementation
2. Run the tests
3. Modify a source file
4. Run the tests again
5. Modify a test file
6. Run the tests successfully
```

The final “tests passed” result does not explain *why* they passed. ReplayLab exposes the missing information:

```text
Step 5
Modified: tests/test_auth.py
Signal: TEST_FILE_MODIFIED

Step 6
Command completed successfully
```

This makes it immediately clear that the successful result followed a test modification.

## What ReplayLab does

For each recorded shell action, ReplayLab:

1. runs the command inside the official SWE-bench task environment;
2. records its output, exit code, and duration;
3. compares the repository state before and after the command;
4. captures worktree and Git-index changes;
5. stores the changed files and exact diffs;
6. flags modifications to tests and test configuration.

The current behavioral signals are:

- `TEST_FILE_MODIFIED`
- `TEST_FILE_DELETED`
- `TEST_CONFIG_MODIFIED`

ReplayLab generates two outputs:

- a self-contained HTML report for inspection;
- a JSON sidecar containing the complete structured replay data.

## Installation

ReplayLab supports Python 3.10 through 3.12. Python 3.11 is recommended. Docker Engine with Linux-container support is required to run the SWE-bench environments.

```bash
python -m pip install -e ".[dev]"
```

## Run ReplayLab

```bash
replaylab analyze \
  --trajectory path/to/trajectory.traj.json \
  --instance django__django-12345 \
  --output report.html
```

The command creates `report.html` and `report.json` in the selected output directory.

ReplayLab currently accepts structured `mini-swe-agent-1.1` trajectories. The default dataset is `SWE-bench/SWE-bench_Verified`, and the default split is `test`.

## Command options

```text
--dataset DATASET       Select the dataset
--split SPLIT           Select the dataset split
--timeout SECONDS       Override the per-command timeout
--allow-network         Enable networking inside the replay container
--json-output PATH      Set the JSON output path
```
