# Direct weight editing through MCP

English · [한국어](STANDALONE_MCP.ko.md)

This workflow runs without loop-engineering or another MCP server. Connect ForwardEdit MCP to a workspace containing a prepared model and development data. A fresh Codex or Claude session analyzes weights, chooses explicit edits, saves candidates, runs inference and revises candidates from feedback. Gradient and optimizer training are not provided. Any performance improvement requires separate independent evaluation.

## Install and register a client

The repository distributes a Python MCP server and skills in source form. It does not provide a marketplace plugin installation command. Follow the [README installation steps](../README.md#install) and install the source in editable mode into your chosen inference environment.

```bash
git clone https://github.com/AnamKwon/ForwardEdit.git
cd ForwardEdit
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[lm,mcp]'
```

This combination uses the Transformers 4.x baseline environment. Create a separate Qwen research environment with `.[qwen]`; do not install both dependency groups into the same environment. The executable in the registration below must belong to the selected environment.

For Claude Code, replace the placeholders with actual absolute paths and create the workspace directory first:

```bash
claude mcp add --transport stdio --scope user forwardedit -- \
  /absolute/path/to/ForwardEdit/.venv/bin/forwardedit-mcp \
  --workspace /absolute/path/to/experiment
claude mcp get forwardedit
```

For Codex, use the [Codex registration guide](CODEX_MCP_CLIENT.md). Clients configured through files can use the JSON example below. After registration, start a new session and call `edit_api_schema` and `read_guidance(path="skills/weight-space-mcp-only/SKILL.md")` to check the connection. These two tools work with an empty workspace and require no model inference. Prepare the model, data and configuration below before calling `experiment_info` or model tools.

## Server and workspace

Start the server with the following command. After package installation, `forwardedit-mcp --workspace /absolute/workspace` uses the same entry point. `weight-space-mcp` remains a compatibility alias.

```bash
/path/to/model-environment/python -m weight_space_lab.standalone --workspace /absolute/workspace
```

The Python environment must include this project, MCP and PyTorch/Transformers versions that support the target model. Keep Qwen3.5 separate from the Transformers 4.x small-model environment. If using vLLM, pin its separate executable in the workspace's `research.json`. GPU access follows the server's runtime environment, not the MCP client's environment.

Example Claude MCP configuration:

```json
{
  "mcpServers": {
    "forwardedit": {
      "command": "/path/to/model-environment/python",
      "args": ["-m", "weight_space_lab.standalone", "--workspace", "/absolute/workspace"],
      "env": {"CUDA_VISIBLE_DEVICES": "0"},
      "timeout": 7200000
    }
  }
}
```

A research workspace needs the original `model/`, `discovery`/`validation` inputs in `practice.json`, and a `research.json` that pins source, data, precision, inference policy and budgets with hashes. An optional `protocol.json` specifies validation IDs, edit limits and selection criteria. Keep the independent test outside the editor's MCP workspace. This workflow does not download models or prepare datasets automatically.

Example session request:

> Use only ForwardEdit MCP. Read `skills/weight-space-mcp-only/SKILL.md` through `read_guidance`, then attempt to improve this workspace model's mathematics performance. Analyze actual weights and forward observations, choose targets, coordinates and changes yourself, and record the rationale and falsification conditions. Do not use gradients, optimizers, finite differences, automatic fitting, or coordinate/strength sweeps. Check losses of originally correct answers. Do not access the independent test before freezing the candidate.

Clients without a shell can use `array_compute`, `prepare_edit_from_arrays`, `create_candidate_from_arrays`, `record_observation` and `forward_and_wait` for arithmetic, initial analysis, editing, recording and inference. These tools call the project's candidate writer and research inference; they do not choose edit values automatically. Explicit numeric plans, hashes, calls, errors and actual inference results remain in the workspace. MCP path restrictions apply only to registered tools. To verify an MCP-only run, disable the client's shell/file tools and other MCP servers separately.

Registering the server does not automatically inject skill instructions. Install the local skills in your client or actually read them through `read_guidance`, as above. `freeze_candidate` records the selection and hashes; it does not approve a performance claim. Report improvement only after independent evaluation passes accuracy, preservation of originally correct answers and uncertainty criteria.

MCP SDK 1.x and 2.x expose the same project API. Install the Qwen model/analysis environment from source with `pip install -e '.[qwen]'`. Install and pin a GPU-driver-compatible PyTorch first. vLLM has its own PyTorch/Transformers constraints and should use a separate environment. Keep the source directory available because skills are read from it. Verify vLLM's CUDA/driver compatibility and record the actual version that passed the original-model inference control through MCP.

Before writing an edit, call `edit_api_schema` for required JSON fields and formulas. The tool catalog also includes the plan structure. `scale.strength` is the fraction removed: `W_new=(1-strength)*W`. Check array-reference plans with `preview_edit_from_arrays`. Error responses identify required fields, argument paths and the underlying cause.

To resume the same interrupted experiment, use `workspace_status` and `list_workspace_artifacts` to locate existing results and notes. Do not rerun successful inference merely to select a different outcome. Preserve `research.json`, the ledger and snapshots to retain cumulative budgets and provenance. SDK-internal turns, actual model response IDs and tool calls are distinct counts; measure each separately.
