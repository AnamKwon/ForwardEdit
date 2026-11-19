# ForwardEdit architecture

ForwardEdit implements a reusable inspection, direct-edit and evaluation workflow inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Its contribution is tooling and recorded workflow. A new editing algorithm, superior performance over WSP and generally improved models have not been established.

The LLM chooses hypotheses, tensors, coordinates, directions and coefficients. Tools execute explicit CPU arithmetic, persist candidates and return forward observations. No provider call or automatic weight-fitting loop selects edits inside the package. The [inference-only contract](docs/INFERENCE_ONLY_CONTRACT.md) applies to implementation, tests and examples.

## Components

| Component | Responsibility |
|---|---|
| `core.py` | Workspace paths, source hashes, explicit numeric operations, dtype-rounded change budgets and candidate manifests |
| `lineage.py` | Source/parent/child comparisons and cumulative revisions with current-value checks |
| `computation.py`, `analysis.py` | Actual module/forward inspection, signed token-position observations, matrix analysis and initial proposals |
| `inference.py`, `diagnostics.py` | FP32 loading, frozen inference, token predictions and activations |
| `evaluation.py`, `experiment.py`, `confirmation.py` | Baseline/candidate metrics, declared candidate comparisons, frozen choices and independent confirmation |
| `research.py`, `research_worker.py`, `research_analysis.py` | Hash-bound, budgeted research jobs, BF16 observations and proposals against retained activations |
| `vllm_generation.py` | Optional generation backend in a separately pinned runtime |
| `server.py`, `standalone.py`, `mcp_compat.py` | MCP adapters, SDK compatibility, restricted research workflow and audit records |
| `cli.py` | Explicit local CLI operations |
| `skills/` | Procedural guidance for an external model editor |

## Runtime paths

The baseline FP32 path reads local `config.json` and a single `model.safetensors`. It supports CPU/CUDA inference; final paired NLL evaluation uses CPU FP32. The research path has its own source/data/decoding bindings and supports configured BF16 inference. Optional vLLM generation uses an explicit compatible executable and environment. These paths have distinct precision, model-support and evaluation scope.

The package does not automatically download models, execute remote model code, load sharded/quantized checkpoints, or export a complete ready-to-distribute Hugging Face model. Shared/tied weight edits are restricted. Original WSP task restrictions and decoding conditions must be checked separately for any reproduction claim.

## Edit lifecycle

1. Inspect actual parameter values, module ownership, axes and forward code.
2. Observe failure and reference contexts through inference. Retain token positions and artifact identities.
3. State a mechanism hypothesis, explicit value derivation, alternative explanation, prediction and falsification condition.
4. Prepare a proposal, then write an explicit candidate. A prepared proposal is not a causal or performance certificate.
5. Reobserve the candidate. Revise from actual parent feedback or restore values, preserving lineage.
6. Select on development/validation and freeze source, candidate, dataset and decoding hashes.
7. Confirm the fixed choice on independent data and record every failed preservation/improvement gate.

`run_experiment` compares a finite set of already authored plans under declared rules. `scripts/research_schedule.py` executes an already authored forward schedule and persists job IDs before polling. Neither creates or fits new weight values. Bounded controls need stated hypotheses; this project does not provide coordinate/strength search recipes.

## Numeric and evidence boundaries

`set_values` checks current expected values and writes explicit replacements. `scale` attenuates a selected block by `(1-strength)`. `project_out` subtracts a stated directional component; coefficients above one can reverse it. `rank_one` adds a stated outer product. Limits apply to actual rounded changes, and revisions retain parent edits unless explicitly restored.

For `Linear`, weights have layout `[out,in]`; GPT-2 `Conv1D` uses `[in,out]`. A large activation, weight or singular direction does not identify a semantic role. `trace_token_flow` records observed tensor identities and signed values; functional operations and unobserved paths prevent it from being a complete graph.

Keep three conclusions separate: a value was written correctly; a predicted behavior changed; an independent task improved while required behavior was preserved. Tests establish covered software behavior. Pass counts, valid manifests and plausible explanations do not establish model improvement.

## Agents and publication

[AGENTS.md](AGENTS.md) governs repository maintenance. Product `skills/` guide an external weight editor. Personal coding-agent settings are ignored; portable MCP examples live in `examples/mcp/`. The package requires an external client and does not require a particular provider.

Host review and historical cases stay outside model-editor guidance. `read_guidance` exposes shipped skills and the two approved contract documents, not historical results. This tool-level restriction does not isolate a client with unrestricted filesystem/shell access. A blind experiment must configure that boundary explicitly.

`runs/` remains local. `experiments/` contains selected text evidence, plans, tool receipts and results with source and published checksums. Weights, activation arrays, private client configuration and full datasets are omitted. Integrity checks require no model, agent or GPU; full replay requires separately retained source assets and the historical runtime.

Historical records are snapshots of earlier implementations and may contain superseded names or claims. Read final decisions, omissions and [review limitations](docs/REVIEW_GUIDE.md) together. The [case index](experiments/README.md) retains positive score comparisons, negative results and failed acceptance gates.
