# Using ForwardEdit MCP with Codex

English · [한국어](CODEX_MCP_CLIENT.ko.md)

ForwardEdit provides a stdio MCP server that runs without loop-engineering. A fresh Codex session reads the project skills and chooses explicit edit values. The server executes the specified arithmetic and returns actual inference results. A weight change alone does not establish a performance improvement.

You need a workspace with a prepared model, development data and inference policy, plus a Python environment containing this project. See [workspace preparation](STANDALONE_MCP.md). Keep the independent test outside this workspace.

## Register through the CLI

Complete the [source installation](../README.md#install), create the workspace directory and replace both paths with actual absolute paths for your environment:

```bash
codex mcp add forwardedit --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  /absolute/path/to/ForwardEdit/.venv/bin/forwardedit-mcp \
  --workspace /absolute/path/to/experiment
codex mcp get forwardedit
```

`mcp get` checks saved registration only. In a new Codex session, actually call `edit_api_schema` and `read_guidance(path="skills/weight-space-mcp-only/SKILL.md")` to verify the connection and read the instructions. Both work in an empty workspace without model inference. Call `experiment_info` after preparing the model, development data and `research.json`.

This repository supplies a local stdio MCP server and skills, without a separate Codex marketplace plugin bundle. Reading skills through MCP does not require copying them into a local skills directory.

## Configure through a file

Use the following example if editing the configuration directly. Do not also add a duplicate server entry through the CLI.

```toml
[mcp_servers.forwardedit]
command = "/absolute/model-environment/bin/python"
args = ["-B", "-m", "weight_space_lab.standalone", "--workspace", "/absolute/workspace"]
startup_timeout_sec = 60
tool_timeout_sec = 7200
required = true
tool_input_schema_max_bytes = 50000

# Use only if the user has already authorized reads, direct edits and
# inference in this workspace. This setting applies to this server only.
default_tools_approval_mode = "approve"

[mcp_servers.forwardedit.env]
CUDA_VISIBLE_DEVICES = "0"
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
PYTHONDONTWRITEBYTECODE = "1"
```

Install this project into the Python environment above. If using a module search path instead of source installation, add `PYTHONPATH = "/absolute/forwardedit"` to the same `env` table. When using vLLM, match its environment, driver and sampler settings to the workspace's frozen policy.

With noninteractive `approval_policy = "never"`, even read tools can be denied depending on the server's default approval policy. The server-specific `default_tools_approval_mode` above scopes an authorization the user has already given. There is no need to disable global sandbox, authentication or administrator policies. See the [official Codex MCP configuration](https://developers.openai.com/codex/mcp) for per-server approval settings.

Example request for a fresh session without other MCP servers or prior conversation:

> Use only forwardedit MCP and the project skills to experiment with this model's mathematics performance. Read `skills/weight-space-mcp-only/SKILL.md` with `read_guidance`, then check `edit_api_schema`, `experiment_info` and `workspace_status`. Choose targets and changes from actual parameters, computation structure and forward observations. Use GPU only for inference. Do not use gradients, optimizers, finite differences, automatic fitting, or random/coordinate/strength sweeps. Record a proposal and falsification conditions before editing. Verify actual values loaded on GPU, accuracy on the same inputs, losses of originally correct answers and reference NLL. After any feedback-based revision, freeze one candidate. Evaluate the independent test only after freezing. Do not use shell tools, file editing or another MCP server.

If Codex exposes `functions.exec` as a tool-call wrapper, locate `forwardedit` tools in `ALL_TOOLS` and call only that MCP. Disabling the code-mode host can also disable registered MCP calls in this client. Audit use of the wrapper separately from the actual MCP tools invoked inside it.

To verify independent execution, compare actual client MCP events against requests/responses in the workspace's `mcp-audit`. Check that only one server was registered, no other MCP/shell/file-editing tools were invoked, and actual edits match the LLM-submitted plan. This verifies the fresh session and tool path; it does not establish OS-container isolation.

Count provider `response_id` values, input/cached/output tokens, user turns, tool calls and inference batches separately. Codex's saved-session `token_usage_record` provides tokens per actual response. Do not add reasoning tokens again when they are already included in output tokens. Distinguish an API-equivalent cost estimate from the subscription account's actual bill.

## Manage evaluation time

Diagnose a small number of failed and normal cases first. Evaluate promising candidates within the full fixed validation budget. If originally correct answers are lost beyond the preservation limit, analyze and revise or end the run as a failure. Normally, run a large independent test only after a candidate has passed development accuracy and preservation criteria and has been frozen. A `diagnostic_only` freeze does not automatically authorize a large evaluation. A short generation diagnostic does not establish final performance. Before evaluation, state the token limit, item count × model count, actual generated-token count when available, and estimated runtime.
