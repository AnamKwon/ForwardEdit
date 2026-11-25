# Data separation for LLM weight editing

[한국어](EDIT_EVALUATION_PROTOCOL.ko.md)

Weight editing needs the same separation between data used to construct a change
and data used to evaluate transfer as fine-tuning. An improved score on examples
used to derive an edit is a development result. It is not validation generalization.
The [inference-only contract](INFERENCE_ONLY_CONTRACT.md) still applies: the LLM
specifies explicit CPU arithmetic; this protocol does not authorize training.

## Roles fixed before editing

| Cohort | Use | Exposure to the weight editor |
|---|---|---|
| Edit/train (`discovery`) | Error analysis, labels, activations, edit design and feedback revisions | Allowed |
| Selection validation (`validation`) | Rank/select explicitly authored candidates using fixed metrics | Selection metrics; no case-derived weight updates |
| Final test | Compare the frozen choice and declared original baselines | Open after freeze; never feed back into selection or editing |

Three roles prevent repeated validation selection from being mistaken for final
generalization. A two-way edit/train–holdout experiment is also valid when one
candidate is frozen before the holdout is opened and that holdout never selects
or changes the candidate. State which design is used; do not infer it from a
filename or an API split label.

1. Use appropriate official splits, or partition a fixed source pool before model
   error inspection. Save source/version hashes, IDs, counts, assignment rule and
   seed. Declare exclusions and known prior exposure before outcomes are seen.
2. Deduplicate and group related variants at the intended independence unit. For
   example, put all questions from one image or paraphrases of one problem in one
   group when measuring transfer across those sources. Declare whether generated
   tests share templates with edit data. Never choose favorable failure subsets
   for the validation/test cohort after observing model outputs.
3. Construct/revise edits from edit/train data only. Keep its metrics separate
   from validation metrics. Use validation to select a candidate under a fixed
   rule. If validation cases become inputs to edit construction, disclose that
   they have become development data; their subsequent score is not independent.
4. Freeze the chosen patch, source/parent, data manifests, prompts/tokenizer,
   decoder, precision, token cap, scorer, baselines and decision rules before
   final evaluation. Compare the same items under matched conditions.
5. Report final target accuracy, originally correct answers lost, cutoffs and
   unrelated-task preservation separately. A net gain and a failed retention
   rule can coexist. Do not retune or switch candidates on final results.

Do not impose a universal ratio without considering available examples and
source groups. Report actual counts and uncertainty; multiple completions of one
problem do not increase the number of independent problems. Previously exposed
cohorts stay exposed when copied or renamed. No post-hoc repartition can undo that.

## MCP implementation

Read [portable editor guidance](../skills/weight-space-engineer/references/data-separation.md)
before the first task observation. Research `discovery` maps to edit/train and
`validation` to selection by default. Both can be read by development tools, so
the host must manage final-data access separately. The scientific role and all
actual exposure must be recorded even when an API uses a different wire name.

The research loader rejects cross-split duplicate normalized prompts, identical
token inputs, and repeated declared `group_id` values. Image identity uses byte
hashes together with the text/tokens. Example metadata:

```json
{"id":"problem-17-paraphrase-2","group_id":"problem-17","prompt":"…","answer":"…"}
```

Variants with that `group_id` belong to one split. This preflight covers the loaded
dataset, not semantic paraphrase discovery, unrecorded prior exposure, all files
in other workspaces, or OS-level isolation. Separate development and final
workspaces also need host review of their manifests for overlap and grouped
sources. The existing FP32 confirmation path checks an initially sealed final
file; research BF16 final evaluation requires a host-controlled frozen follow-up
workspace and aggregate study budgets. Do not claim a new server-enforced test
gate from the duplicate check.

## Required report layout

| Data role | Items/groups | Used for edit construction? | Used for selection? | Original | Fixed edit | Rescues/losses | Cutoffs |
|---|---|---|---|---|---|---|---|
| Edit/train | Declare | Yes | Declare | Measure | Measure | Measure | Measure |
| Selection validation | Declare | No | Yes | Measure | Measure | Measure | Measure |
| Final test | Declare | No | No | Measure, or unmeasured | Measure, or unmeasured | Measure | Measure |

Include source/split/patch hashes, controls, decoder/precision, item-level paired
outcomes, exposure limitations, and **Unaddressed criticisms / 미대응 비판** in case
reports. A result cannot move to another row merely because its filename changed.
This is a prospective protocol; it does not retrofit historical experiments or
create new model measurements.

## Comparison tiers (adopted 2026-11-24, prospective)

A weight edit changes the model; an inference-time method (prompting, few-shot,
CoT, decoding settings, budget forcing) changes how the model is called. Reports
keep these apart and state each tier separately:

1. **Model level (primary).** Original vs edited model under the same deployment
   settings. "Default settings" means the model vendor's recommended decoding for
   the mode being used (Qwen3.5 thinking, general tasks: T 1.0, top-p 0.95, top-k 20,
   presence penalty 1.5, up to 32,768 output tokens; corrected 2026-11-25 from an
   earlier T 0.6 mix-up), not greedy decoding chosen to make the original look weak. Also
   compare against other weight-modification methods on the same data (LoRA,
   MEMIT/ROME; host-only comparators) and check every supported usage mode
   (e.g. thinking on and off) for losses.
2. **Combination.** Edited model + the strongest inference-time method vs the
   original model + the same method, at every declared setting of that method.
   A positive combination result is the strongest evidence that the edit adds
   something inference-time methods cannot.
3. **Reference.** Inference-time-only scores, reported for context. An edit that
   matches but does not exceed them is described as moving that behaviour into
   the weights (no serving change needed), not as a capability gain.

Verdicts already pre-registered under earlier rules are not re-judged; papers may
re-read old results through these tiers but must quote the original verdict.
