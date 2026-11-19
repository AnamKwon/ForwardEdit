# ForwardEdit

**Inference-only tools for inspecting, directly editing, and evaluating neural network weights.**

[한국어](README.ko.md) · [Case studies and logs](experiments/README.md) · [Architecture](DESIGN.md) · [Detailed usage](docs/USAGE.md)

ForwardEdit lets an external LLM inspect real weights and forward observations, propose a failure mechanism, and specify explicit parameter changes through MCP or a CLI. Edits are computed on CPU; GPU use is inference only. The package does not train a model or automatically fit edits to a loss.

It is inspired by [Weight-Space Programming (WSP)](https://rikiyat.github.io/blog/weight-space-programming/). The contribution here is a reusable tool interface, editing workflow and inspectable evidence. A new editing algorithm or superiority over WSP has not been established.

## What is included

- Actual parameter values, module ownership/axes, loaded forward source and selected token activations.
- Explicit value replacement, local scaling, projection and rank-one edits with source checks and change budgets.
- Parent/child revisions, value restoration and comparisons against the original model.
- Forward-only development evaluation, frozen selections and separate independent confirmation.
- Model-editor skills and MCP interfaces usable from external clients such as Codex or Claude.
- Historical tool receipts, edit plans, failed hypotheses and both positive and negative score comparisons.

The LLM chooses what to change and why. Tools calculate and record the specified operations. A plausible explanation, successful file write or passing software test does not establish a causal mechanism or a better model.

## Read an actual experiment

These are retrospective publications of completed local studies, not new runs. Every bundled study failed at least one declared acceptance criterion; none is presented as an accepted model improvement.

| Case | Recorded outcome | Why read it |
|---|---|---|
| [Direct values 001](experiments/direct-values-inference-only-001/README.md) | Two candidates, six values changed each; both rejected | Exact old/new numbers, MCP calls and CPU/CUDA checks |
| [TinyStories 004](experiments/skill-edit-experiment-004/README.md) | Independent repetition 7.279% → 7.081%; interval includes zero | Development improvement did not establish an independent benefit |
| [TinyStories 005](experiments/failure-repair-005/README.md) | Independent repetition 6.060% → 6.512%; rejected | Feedback revisions, an opposing control and normal-case damage |
| [Source-informed Qwen 006](experiments/extended-discovery-006/README.md) | Independent answers 18 → 103 / 192; preservation failed | A large score gain can still fail the declared acceptance gate |
| [MCP-only Qwen 009–014](experiments/independent-mcp-014/README.md) | Development 5 → 10 / 64; independent 25 → 22 / 192 | Fresh client sessions, failures, one feedback revision and negative transfer |

The Qwen 006 edit used source-paper-informed host guidance; it is not blind rediscovery. Its gain is against the fixed original under the recorded decoding configuration, with no demonstrated advantage over the strongest unedited configuration. The studies use different populations and protocols and are not a controlled head-to-head comparison. See each case's **Unaddressed criticisms / 미대응 비판**.

The [evidence index](experiments/README.md) explains selection, redactions and omissions. Check the published bytes without loading any model:

```bash
python3 scripts/verify_public_evidence.py
```

This checks artifact integrity and JSON parseability. It does not reproduce inference, authenticate unavailable originals or certify performance.

## Install from this checkout

Python 3.11 or newer is required. Use an editable source install: MCP guidance reads the repository's `skills/` and contract documents.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[lm,mcp,test]'
.venv/bin/forwardedit --help
.venv/bin/forwardedit-mcp --help
```

This installs the Transformers 4.x small-model path. The separate Qwen/research environment uses `pip install -e '.[qwen]'`; choose a PyTorch build compatible with the inference hardware. Optional vLLM generation requires its own compatible, explicitly pinned executable. Do not combine the `lm` and `qwen` Transformers constraints in one environment. See [runtime setup](docs/STANDALONE_MCP.md).

The distribution is `forwardedit`; the Python import/module path remains `weight_space_lab`. Existing `weight-space` and `weight-space-mcp` commands are compatibility aliases.

## Connect an LLM client

Prepare a workspace containing your own local model, development inputs and the appropriate configuration. No checkpoint or dataset is downloaded automatically.

```bash
.venv/bin/forwardedit-mcp --workspace /absolute/path/to/experiment
```

Portable configuration templates are provided for [Claude-style MCP clients](examples/mcp/claude.json) and [Codex](examples/mcp/codex.toml). Replace all absolute-path placeholders. For setup details, see [standalone MCP](docs/STANDALONE_MCP.md) and [Codex client notes](docs/CODEX_MCP_CLIENT.md).

The editor starts by reading `skills/weight-space-mcp-only/SKILL.md` through `read_guidance`, then inspects `edit_api_schema` and `experiment_info`. Choose targets and coefficients from actual observations, record predictions before intervention, and freeze selection before independent evaluation. Historical cases are for human review; supplying their recipes to an editor changes a blind experiment into a source-informed one.

## CLI and model support

```bash
.venv/bin/forwardedit --workspace /absolute/path/to/experiment inspect model/model.safetensors
.venv/bin/forwardedit --workspace /absolute/path/to/experiment preview plan.json
.venv/bin/forwardedit --workspace /absolute/path/to/experiment apply plan.json candidate-01
```

`plan.json` is an explicit caller-authored edit, not a request for automatic optimization. See [numeric examples and evaluation APIs](docs/USAGE.md).

| Path | Scope |
|---|---|
| Basic model/CLI path | Local `config.json` plus one `model.safetensors`; CPU/CUDA FP32 observations; CPU FP32 paired NLL confirmation |
| Research MCP path | Hash-bound model/data/decoding policy, bounded logical jobs and configured BF16 inference; optional separate vLLM generation runtime |

Quantized/sharded checkpoints, remote model code and automatic model downloads are unsupported. Tied/shared parameter edits are restricted. A saved candidate contains replacement tensors and a manifest, not a complete Hugging Face checkpoint. Read the [architecture](DESIGN.md) before assuming support for another model or precision.

## Execution rules

The [inference-only contract](docs/INFERENCE_ONLY_CONTRACT.md) excludes backward/autograd, gradient collection, numerical gradients, optimizer/LoRA training, automatic loss fitting and random/coordinate/strength sweeps. Explicit CPU arithmetic and bounded hypothesis-driven controls are allowed. A one-shot multi-error update must have its examples, layers, coefficients and formula chosen explicitly by the LLM; least-squares/pseudoinverse fitting remains prohibited.

Models run frozen in evaluation/inference mode. The runtime and tests include training API guards. These are software checks, not an OS sandbox or a guarantee about arbitrary external code. Correct writes, behavior changes and independent task gains are separate claims.

## Repository and agents

| Path | Purpose |
|---|---|
| `weight_space_lab/`, `tests/` | Implementation and software checks |
| `skills/` | Product guidance for a weight-editing LLM |
| `AGENTS.md`, `DESIGN.md` | Contributor instructions and current implementation architecture |
| `docs/` | Usage, contracts, review checklist and historical reports |
| `experiments/` | Selected public historical text evidence |
| `examples/mcp/` | Shareable client configuration templates |
| `archive/gradient_experiments/` | Non-executable records of earlier disallowed approaches, excluded from current success evidence |

Personal agent configuration, credentials, virtual environments, logs, checkpoints and full `runs/` workspaces are ignored. The public cases are deliberate copies, including records from local workspaces that were stored elsewhere through symlinks. Model weights, large activations and full datasets are not bundled. Published test examples are now exposed and cannot serve as fresh held-out data.

## Development checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider
python3 scripts/verify_public_evidence.py
```

Tests use untrained numerical fixtures and inference-only paths. Report skipped checks separately. Source/evidence maintenance requires no new model experiment.

## Attribution

The original WSP work and its [public experiment logs](https://github.com/RikiyaT/lm_understand-logs) provide the methodological reference. [Our source investigation](docs/WEIGHT_SPACE_SKILL_DESIGN.md) distinguishes that work from this implementation. External model and dataset terms remain applicable; publishing these text records does not distribute or relicense their checkpoints or full datasets.
