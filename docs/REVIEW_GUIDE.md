# Host and reviewer checklist

Audience: experiment hosts, maintainers and reviewers. Do not inject historical solutions or model-specific findings into a blinded model editor's prompt. This portable checklist carries procedural requirements from the earlier local critique notes, without their task-specific solution hints. It is outside the `read_guidance` allowlist.

Before a new protocol, report or improvement claim, check:

1. **Unedited baselines.** Compare against the strongest measured applicable configuration, including decoding/prompt changes, non-LLM recipes and runtime steering where relevant. If these controls were not measured, disclose the gap and limit claims to the actual fixed-baseline comparison.
2. **Truncation and budgets.** Report accuracy separately from length/cutoff rate. Count attempts, successful logical jobs, generated tokens, actual edits and editor responses separately. A logical batch is not an autoregressive forward or an epoch.
3. **Preservation.** Include a target-unrelated preservation set and the fraction of originally correct answers lost. Aggregate reference NLL alone does not establish answer preservation.
4. **Prior knowledge and leakage.** Record what the host/editor knew about the model, failure mechanism, source-paper recipes and previous runs. Disclose source-informed interventions and session continuations. Do not call them independent discoveries.
5. **Failure type and controls.** State the proposed failure type in advance; retain identity/opposing or other bounded negative controls. Keep declared criteria unchanged when controls or final results disagree.
6. **Predictions and feedback.** Record predictions before observing results. Establish that parent feedback was actually read before a claimed revision. Preset alternatives are not feedback-dependent revisions.
7. **Claim scope.** Distinguish synthetic recovery, pretrained-model behavior changes, independent task improvements and generalization. Do not infer new capabilities, superhuman interpretation or recursive self-improvement from exact writes or narrow benchmark changes.
8. **Data roles.** Apply [data separation](EDIT_EVALUATION_PROTOCOL.md) before observing errors. Separate edit/train construction, validation selection and selection-unseen final test. Report counts/groups, hashes, exposure and scores for each. A validation case used to derive an edit becomes development evidence. Two cohorts suffice only when the second is opened after one candidate is frozen and is never used for selection/revision. API split labels, new IDs or renamed files do not establish independence.

Freeze selection before independent confirmation. Exposed test items cannot be reused as fresh evidence for the same selection problem. Do not relax thresholds, reset budgets or omit failed sessions to obtain a passing story.

Every new case report must identify **Unaddressed criticisms / 미대응 비판** that remain. In a retrospective publication, name missing controls and incomplete assets without inventing a new preregistration or claiming a new experiment.

## Publishing historical records

Keep source artifacts unchanged locally. Publish a deliberate file selection with original-source and published-byte SHA-256 digests. Document redactions and omitted assets; historical hash fields continue to identify historical originals. A checksum checker validates bundle consistency, not authenticity of unavailable originals, causal conclusions or model performance.

Do not distribute personal agent state, credentials, caches, checkpoints or hidden evaluation material automatically. Published results are now exposed examples, suitable for inspection and audit rather than reuse as a fresh test.
