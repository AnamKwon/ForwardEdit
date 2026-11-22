# Manuscript evidence map

[English manuscript](manuscript.md) · [한국어 원고](manuscript.ko.md) · [Extracted values and source-byte hashes](evidence.json)

This supplement maps claims to retained public artifacts. JSON paths below describe fields to inspect; they are not instructions to execute historical logs. Source snapshot: `32f62aef193fb6cac4dc84db6128657ecfe2be4f`. New descriptive counts are calculated from retained rows only. Historical uncertainty intervals are copied without re-estimation.

## E1. Exact writes 001

- Sources: [host-audit.json](../experiments/direct-values-inference-only-001/host-audit.json), [decision.json](../experiments/direct-values-inference-only-001/decision.json), [case](../experiments/direct-values-inference-only-001/README.md).
- Audit fields: `baseline.validation.nll`; `candidates[*].changed_parameters`; `manual_patch_exact`, `actual_parameter_exact`, `unselected_parameters_unchanged`; `cpu_metrics.validation.nll`; `heldout_evaluated`.
- Recorded values: baseline 2.9254278648124963; candidate 01 2.926467628073225; candidate 02 2.938719861461947; six changed values per candidate; no held-out evaluation.
- Decision: `selected_candidate = null`. The audit's `passed = true` concerns application/verification, not task improvement.
- Claim supported: specified changes were reported as exactly applied and loaded in these checked paths. Semantic interpretation, independent benefit and superiority over an unedited alternative are unestablished.

## E2. Repetition 004

- Sources: [heldout-result.json](../experiments/skill-edit-experiment-004/heldout-result.json), [protocol.json](../experiments/skill-edit-experiment-004/protocol.json), [case](../experiments/skill-edit-experiment-004/README.md).
- Fields: `baseline.mean_repeat_fraction`, `candidate.mean_repeat_fraction`, `comparison.mean_repeat_gain`, `comparison.gain_ci95`, `comparison.reference_nll_increase`, `validation_eligible`, `adoption_passed`.
- Independent n = 64; repetition 7.2791335762% → 7.0811964502%; reduction 0.1979371261 pp; 95% interval [−2.9103522841, +3.2061684080] pp.
- Reference NLL increase 0.0063885260. Cutoff 71.8750% → 64.0625%. The repetition definition is repeated generated content 8-grams, excluding EOS/post-EOS tokens.
- Protocol: greedy generation, 128 tokens, EOS stopping, cache; paired bootstrap 2,000 draws, seed 17; minimum reduction 0.02 in fraction units (2 pp), not 0.02 pp.
- The case README supplies the final 256-value count and revision summary; these are narrative-record claims, separate from the generated numeric figure.
- Decision: development eligible, independent gate failed; no accepted improvement.

## E3. Repetition 005

- Sources: [heldout-result.json](../experiments/failure-repair-005/heldout-result.json), [protocol.json](../experiments/failure-repair-005/protocol.json), [case](../experiments/failure-repair-005/README.md), [final-audit.json](../experiments/failure-repair-005/final-audit.json).
- Same metric fields as E2. Independent n = 128; repetition 6.0596829132% → 6.5117995378%; reduction −0.4521166246 pp; 95% interval [−2.5610529834, +1.6950436686] pp.
- Reference NLL increase 0.0026216425. Cutoff 68.7500% → 66.40625%. The case records worse reference NLL for 91/128 stories.
- Decision: development ineligible; frozen diagnostic confirmation; `adoption_passed = false`.
- Control interpretation: the case records development improvements in both directions for some examples. This weakens directional specificity; it is not a proof of no causal effect under every setting.

## E4. Source-informed Qwen 006

- Sources: [study-outcome.json](../experiments/extended-discovery-006/study-outcome.json), [confirmation-report.json](../experiments/extended-discovery-006/host-confirmation/confirmation-report.json), [protocol.json](../experiments/extended-discovery-006/artifacts/protocol.json), [case](../experiments/extended-discovery-006/README.md).
- Accuracy: `independent.original_correct = 18`, `edited_correct = 103`; report `statistics.items = 192`, `gain = 0.4427083333`, `paired_bootstrap_interval = [0.359375, 0.5260416667]`, `confidence = 0.975`.
- Paired transitions: 86 wrong-to-right, one right-to-wrong. The protocol's independent conditional preservation threshold is `max_previously_correct_to_wrong_fraction = 0.05`. One of 18 originally correct answers is 5.5556%; one of all 192 items is 0.5208%. Only the first denominator answers this preservation question.
- Independent subsets: 128 GSM8K (13 → 76 correct), 64 MATH-500 (5 → 27). These are selected subsets, not full benchmark scores.
- Parsing: `statistics.original_unparsed = 174`, `edited_unparsed = 73`; the 192 retained rows reproduce these counts from null grades. All 86 rescued items have a null original grade and a true edited grade; no parsed-wrong rescue is reported.
- Termination: counting `rows[*].original_finish` / `edited_finish == "max_new_tokens"` gives 175 → 72. This is a descriptive count newly extracted from existing rows, not new model inference. Cutoff and parsing-failure counts must not be interchanged.
- Parameters: `parameter_changes.stored_changed_values = 124928`, `actual_gpu_bf16_changed_values = 123452`, `total_text_parameters = 752393024`, `layers = [15,19]`; storage fraction approximately 0.0166%.
- Accounting: 50 successful development logical batches, 51 accepted attempts including one environment failure, two independent logical paired jobs. These are not 50 edits, epochs or measured engine-internal forward invocations.
- Decision: final host review not approved, `study_complete = false`, preservation criterion failed. Do not promote an interim positive review to final acceptance.
- Limits: source-informed intervention, no strongest-unedited comparison, rescue concentrated in previously unparsed outputs, missing replay binaries, unresolved lost-answer mechanism.

## E5. MCP-only Qwen 009–014

- Sources: [study-report.json](../experiments/independent-mcp-014/study-report.json), [feedback-flow-audit.json](../experiments/independent-mcp-014/feedback-flow-audit.json), [protocol.json](../experiments/independent-mcp-014/protocol.json), [session timeline](../experiments/README.md#same-study-session-timeline-009014), [case](../experiments/independent-mcp-014/README.md).
- Study unit: six sessions in one continuation. Claude's parent candidate and observations were inherited by Codex; this is not six independent runs.
- Feedback: `verified_feedback_dependent_revisions = 1`. Completed result/generation reads precede the actual edit request. The revision restored three coordinates while retaining the parent's other changes.
- Development: `validation[*]` records source 5/64, parent 7/64, child 10/64; original-correct losses remain 3/5 for both candidates. Gain 7.8125 pp is below the required 10 pp. Conditional loss 60% exceeds 5%. Aggregate reference-text preservation passed separately.
- Stored/GPU changes: final candidate `changed_parameters = 3069`; final development `actual_changed_GPU = 3067`.
- Independent: `independent_confirmation` records original 25/192, edited 22/192, gain −0.015625, interval [−0.078125, 0.046875], confidence 0.975, wrong-to-right 13, right-to-wrong 16, conditional loss 0.64.
- Accounting: eight successful development logical jobs, two independent paired jobs, one earlier aborted accepted attempt. Baselines differ from study 006; do not compare their raw accuracy as a treatment effect.
- Decision: `diagnostic_only`, `all_predeclared_acceptance_gates_passed = false`. Actual feedback use is supported; autonomous independent improvement is not.

## E6. Mechanism and publication boundaries

- Source-informed controls: [interim analysis](../experiments/extended-discovery-006/causal-and-transfer-interim-analysis.json). Its `limitations` record that an opposing edit solved 2/6, short controls used 8,192 tokens, and some original generations differed. This is pre-confirmation analysis; references there to a pending independent evaluation reflect its historical timestamp.
- Contributor review: [review guide](../docs/REVIEW_GUIDE.md), [execution contract](../docs/INFERENCE_ONLY_CONTRACT.md), [parameter analysis](../docs/PARAMETER_ANALYSIS.md), [architecture](../DESIGN.md).
- Publication: [case index](../experiments/README.md), [artifact manifest](../experiments/MANIFEST.json), [byte verifier](../scripts/verify_public_evidence.py).
- The supplement extracts 13 numeric/protocol/audit JSON sources and retains their current **published-byte** hashes in `evidence.json`. This is a subset of the full 435 extracted public artifacts. Editorial case READMEs, literature and implementation documents are additional interpretive sources, outside those 13 numeric inputs.
- Original embedded hashes are not rewritten. Checksums do not authenticate omitted originals. Reports retain historical implementation semantics; today's safeguards are not retroactively attributed to every run.
- Excluded from empirical results: separate unpublished local studies, gradient-assisted archives, video mock calls, generated conceptual artwork, proposed follow-up studies and assertions of superhuman interpretation or recursive self-improvement.

## Unaddressed criticisms / 미대응 비판

The manuscript does not provide a strongest-unedited/steering comparison, controlled editor/provider comparison, independent blind rediscovery, broad preservation set, complete circuit identification or fully replayable model/data/runtime bundle. The proposal for future work is explicitly unexecuted. All bundled studies retain failed adoption gates.

## Rebuild and check

```bash
python3 scripts/verify_public_evidence.py
python3 paper/build_evidence.py --check
# Optional regeneration of manuscript data and SVG/PDF figures; requires matplotlib:
python3 paper/build_evidence.py --figures
```

These commands read retained data and calculate descriptive arithmetic. They do not load a model, run historical agents, fit an edit or open a new test.
