# Controls for long sampled inference

Keep the question set, order, batching policy, decoder, limits and actual runtime precision visible in every comparison. Changing a cohort can change generation even with per-item seeds. Compare each candidate with its proper matched baseline; do not attribute a difference from another cohort to the weight edit.

When interpreting a small sampled gain, use a zero-edit candidate under a representative task, length and cohort. Verify actual loaded GPU weights as well as stored files. A short successful identity control covers its own conditions; it does not establish long-generation reproducibility. If a zero-edit pair diverges, retain the result and uncertainty. It does not alone identify the cause as sampling, kernels, batch scheduling or weight loading. Check file/loaded-value equality and fixed-input observations to distinguish them.

Same-input teacher observations can isolate an edited computation path even when sampled continuations vary. Compare upstream states, selected affine inputs, predicted local deltas and actual downstream changes. BF16 residual subtraction includes rounding; a residual-minus-CPU-matmul estimate is not an exact measured attention output. Keep a demonstrated numeric influence separate from a claimed semantic circuit or mathematical improvement.

If generation variability matters, use a finite declared replication or opposing control. Report per-problem outcomes and sampling uncertainty. Multiple completions of one problem are not independent new problems; use the problem as the statistical unit when aggregating. Do not repeatedly reroll until a favorable score appears. Final candidate/data/decoder remain frozen before independent confirmation.

For correct-answer preservation, report both `lost / all_inputs` and `lost / originally_correct_inputs`, plus counts. An aggregate bound can hide losing every originally correct answer when baseline accuracy is low. Declare the intended denominator and threshold before selection; an empty original-correct group leaves that preservation question unassessed. Reference token NLL/accuracy addresses a different behavior and cannot replace task accuracy.

These controls support scoped conclusions. Null variability is not evidence that direct weight editing cannot work; a real parameter delta or one improved completion is not evidence that model performance improved.
