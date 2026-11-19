# API and CLI usage

English · [한국어](USAGE.ko.md)

[Project overview](../README.md) · [Execution contract](INFERENCE_ONLY_CONTRACT.md) · [MCP setup](STANDALONE_MCP.md)

The baseline CLI/evaluation workflow below uses FP32. BF16 long-form inference uses the separate [research workflow](../skills/weight-space-engineer/references/research-tools.md). Derive coordinates and values from actual observations when applying these examples.

## Tools for the editing LLM

| Tool | Observation or action |
|---|---|
| inspect_checkpoint / read_tensor_slice | Read actual names, shapes, hashes and small weight slices |
| describe_parameter | Read actual values, module class, structural role and input/output axes, including candidate values |
| probe_model | Observe next-token top-k and greedy continuation; optional EOS stopping and KV cache; return length, stop reason and text |
| describe_model_structure / inspect_module_computation | Inspect actual module hierarchy, shared weights, forward source and formulas |
| read_parameter_entries / analyze_matrix_geometry | Read arbitrary-rank coordinates; compute SVD and matrix products for selected blocks |
| build_repetition_contexts | Build discovery contexts from actual repetition segments and matched-position normal references |
| trace_token_flow | Record signed per-token inputs/outputs across modules and observed tensor-identity links |
| revise_candidate | Apply the LLM's explicit revision to a parent while retaining values, rationale and hashes |
| compare_candidate_parameters | Compare source, parent and child values; count retained, revised, restored and newly changed entries and check hashes |
| confirm_generation | Generate from the frozen candidate and source on test data; check initially sealed data hashes and frozen source/config/patch/decoding |
| diagnose_model | Observe actual forward input/output mean and RMS, plus weight norms, for source/candidate; no gradients |
| analyze_parameter_contexts | Save signed activations, weight contributions, context contrasts and local effects of specified new values at selected token positions |
| prepare_edit_proposal | Connect analysis to numeric derivation, alternatives, falsification and preservation criteria; distinguish deferral from readiness for a causal probe |
| write_parameter_values | Check LLM-specified old/new values and write exact coordinates |
| preview_edit / create_candidate | Validate or save explicit set_values, scale, project_out and rank_one plans |
| score_candidate | Run discovery/validation inference during editing; return per-example NLL and accuracy |
| evaluate_baseline / evaluate_candidate | Evaluate the source or perform paired confirmation for a frozen candidate |
| run_experiment | Compare already authored, bounded hypothesis plans, freeze selection and confirm independently; never generate new edit values automatically |

The workflow is **read → actual inference → role/mechanism hypothesis → LLM specifies new values → edit → inference feedback → revisit the hypothesis**. Distinguish source/candidate values, numeric differences, actual logit/generation changes and performance improvement. A structural role such as MLP output does not establish a semantic hypothesis such as “a repetition-causing neuron.” A large norm or activation alone does not justify removal.

For initial analysis, follow [parameter analysis before editing](PARAMETER_ANALYSIS.md) to connect actual input signs with selected weight contributions. Compare specified reference/failure positions. Check whether the new values produce the intended local direction and how much they affect reference cases. Missing or contradictory evidence means **defer**; adequate explanation and partial local checks mean **ready_for_causal_probe**. Causality and performance remain unverified until bounded interventions and independent confirmation. This procedure does not block the low-level write API for every use case.

Linear weights use `[out,in]`; GPT-2 Conv1D uses `[in,out]`. `describe_parameter` inspects the actual module and returns axes and the forward formula. Semantic roles require matched contexts and bounded falsification interventions.

## Specify values directly

`write_parameter_values` accepts checkpoint, source_sha256, tensor, rows, columns, expected_values, values, hypothesis, limits and output. `expected_values` contains values actually read from the source; `values` contains new values chosen and justified by the LLM. `create_candidate` can also take the plan below. Names and numbers here illustrate the format; replace them with actual observations.

```json
{
  "checkpoint":"model/model.safetensors",
  "source_sha256":"ACTUAL_SOURCE_SHA256",
  "hypothesis":"A falsifiable hypothesis based on the observed failure and this coordinate's structural contribution",
  "limits":{"max_changed_fraction":0.002,"max_relative_l2":0.05},
  "operations":[{
    "kind":"set_values","tensor":"ACTUAL_TENSOR_NAME",
    "rows":[0],"columns":[1,2],
    "expected_values":[[0.12,-0.03]],"values":[[0.11,-0.025]]
  }]
}
```

Specify up to 4,096 values in rows × columns order. A mismatched hash or old value, nonfinite new value, shape mismatch or exceeded edit budget is rejected. The writer checks actual changed-entry counts and relative L2 after rounding to the storage dtype. The source remains unchanged. A candidate's `replacements.safetensors` contains full replacements for changed tensors, not a complete Hugging Face checkpoint. Keep only this file and `manifest.json` in the candidate directory; store reports outside it.

`scale` multiplies a selected block by `(1-strength)`. `project_out` removes projection onto an explicit direction. `rank_one` adds `strength*outer(left,right)`. Scale strength must be in [0,1]; project_out/rank_one accept finite nonnegative coefficients. A plan allows one operation per tensor and at most 32 operations. A `revise_candidate` sequence combines explicit follow-up operations on the same tensor, rounding to storage dtype after each step. `set_values` needs neither axis nor strength. `create_candidate` branches from the source; `revise_candidate` preserves the parent's changes and combines the new explicit operations.

## CLI

```bash
.venv/bin/forwardedit --workspace /absolute/experiment inspect model/model.safetensors
.venv/bin/forwardedit --workspace /absolute/experiment describe model TENSOR --rows 0 1 --columns 0 1
.venv/bin/forwardedit --workspace /absolute/experiment probe model development.json --device cuda --max-examples 2 --max-new-tokens 16
.venv/bin/forwardedit --workspace /absolute/experiment diagnose model development.json TENSOR --device cuda
.venv/bin/forwardedit --workspace /absolute/experiment analyze-parameter analysis-spec.json analysis.json
.venv/bin/forwardedit --workspace /absolute/experiment prepare-proposal analysis.json basis-request.json proposal.json
.venv/bin/forwardedit --workspace /absolute/experiment preview plan.json
.venv/bin/forwardedit --workspace /absolute/experiment apply plan.json candidate-01
.venv/bin/forwardedit --workspace /absolute/experiment describe model TENSOR --rows 0 1 --columns 0 1 --candidate-dir candidate-01
.venv/bin/forwardedit --workspace /absolute/experiment score model candidate-01 development.json --device cuda
.venv/bin/forwardedit --workspace /absolute/experiment probe model development.json --candidate-dir candidate-01 --device cuda
```

CUDA is used only for inference in probe/diagnose/score. Final paired gate evaluation uses CPU FP32. Loaded models are frozen and in eval mode; forward calls use `inference_mode`. The `include-gradients` CLI/MCP option was removed. Legacy Python gradient requests and `gradient_assisted` mode are also rejected.

## Data and independent confirmation

```json
{"name":"local-probe","version":"1","splits":{
 "discovery":[{"id":"d0","tokens":[1,2,3]}],
 "validation":[{"id":"v0","tokens":[2,3,4]}],
 "test":[{"id":"t0","tokens":[3,4,5]}]}}
```

Prepare token IDs with the same model/tokenizer. IDs must be unique across all splits; each sequence needs at least two tokens. During editing, expose only development data to the worker and restrict score/probe to discovery/validation. Declare candidate counts, forward/token budgets, minimum improvement, allowed regressions and generation settings beforehand. Automatic loss fitting and strength/coordinate sweeps are forbidden. Only bounded interventions, controls and dose-response checks designed to falsify a mechanism hypothesis are allowed.

Use `evaluate_candidate` after freezing. It evaluates all splits and must not be used for exploration. The gate option `min_nll_gain` must be positive. Selection/held-out gain, held-out confidence interval, accuracy, preservation-split and leakage checks must pass. Do not reselect a runner-up on the same test. Lower NLL alone does not establish better mathematics or free-generation ability.

`run_experiment` configuration accepts model_dir, dataset_path, plans (at most 12), min_nll_gain, selection_split, heldout_split, preserve_splits, max_nll_regression, max_accuracy_drop, seed, bootstrap, max_examples and max_tokens. Only `direct_edit` is allowed as design_mode. It saves baseline, identity, selection evidence, frozen choice and confirmation, without replacing the source automatically. The host manages file-access sealing and total experiment budgets; this MCP is not an OS sandbox.

For repetition evaluation, use `probe_model(stop_at_eos=true,use_cache=true)` and `summarize_generations`/`compare_generations` on recorded results. Exclude tokens after EOS and retain reference NLL/accuracy, output length and short-output rate. After freezing validation selection, use `confirm_generation` to evaluate only the source and selected candidate on the initially sealed test data under identical decoding. This generation criterion differs from `evaluate_candidate`'s NLL-improvement gate.
