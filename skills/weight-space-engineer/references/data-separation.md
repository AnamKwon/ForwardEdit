# Separate edit construction, selection, and final evaluation

Apply this before observing task errors or proposing a parameter edit. Data used
to construct edits serves the role of fine-tuning training data, even though the
inference-only contract forbids training and optimization.

| Scientific role | Permitted use | What its score means |
|---|---|---|
| Edit/train (`discovery`) | Read labels, failures, activations and references; derive explicit edits and feedback-dependent revisions | Performance on data used to construct the edit |
| Selection validation (`validation`) | Compare already specified candidates against matched originals under declared criteria | Candidate-selection performance, not independent final evidence |
| Final test | Evaluate the frozen selected patch and frozen controls after editing stops | Performance on selection-unseen data, subject to the declared scope |

Use an official dataset split when suitable. Otherwise, have the host partition
and hash a fixed source pool before error inspection. Record source revision,
item IDs, split membership, seed if used, counts and prior exposure. Never select
the validation/test cohort by observed model failures or favorable edit effects.
Do not retrospectively split an already inspected pool and call it unseen.

Keep related records together by the scientific independence unit: original
problem and its paraphrases, document, image, patient, or generated template as
appropriate. State whether the goal is new instances within known templates or
transfer to unseen templates. Exact input deduplication alone does not establish
semantic independence. The host must review relatedness and known prior exposure.

The editor's numerical derivations, directions, coefficients, coordinate choices,
case bank and feedback revisions use edit/train examples only. Do not use a
validation failure's label, activation or error vector to construct a subsequent
edit and continue describing that example as unused validation. If that happens,
record the exposure and treat it as development evidence. A genuinely untouched
final cohort is still required for an independent claim.

Select candidates with validation under criteria fixed before its results. Keep
validation use limited to selection metrics; do not request case-level traces for
weight construction. Validation can guide selection, so even a positive result
there is not an unbiased final estimate. With only two cohorts, the second may
serve as a final holdout only if it is opened after one choice is frozen and is
never used to select or revise candidates. Repeated selection needs a third cohort.

Before final access, freeze source/parent/patch hashes, prompt and tokenizer,
decoder, precision, token cap, scoring code, baselines and decision criteria. Run
matched original/edited evaluation; report each split separately with item counts,
rescues, regressions, cutoffs and unrelated preservation. A failed final result
does not authorize selecting a runner-up or changing the scorer, prompt or bias
on that same cohort. Keep failed gates and corrections visible. A material scoring
bug requires an audit and disclosed correction, not silent favorable reselection.

## Current tools and their boundary

The research MCP accepts only `discovery` and `validation` wire split names. Both
are readable through development tools. The name `validation` is not a seal and
does not prove it was unused. Record the scientific role and exposure separately.

The research dataset loader rejects identical normalized prompt text or token
inputs across those splits. Images are part of the input identity and are compared
by byte hashes, so renaming an image cannot create a new input. Optional nonempty
`group_id` values must remain in one split. This preflight does not detect every
paraphrase, reconstruct source groups, or inspect exposure in other workspaces.
It also does not guarantee that the supplied prompt agrees with supplied tokens.

For the existing small FP32 path, use the frozen `confirm_generation` workflow
with its initially sealed data. For research BF16, the host must keep final items
outside the editing workspace/client context and run fixed follow-up evaluation
after freeze. If a separate immutable research workspace uses the wire name
`validation` for that final cohort, explicitly record its final-test role, parent
freeze, data hashes and cumulative budgets. Do not give that workspace back to
an editor for revisions. These are workflow controls, not an OS sandbox or a
universal server-enforced train/validation/test state machine.

Reports must put edit/train, selection validation, and final test in separate
rows. Missing final evaluation stays unmeasured. Do not rename historical
development or exposed confirmation scores to make them appear independent.
