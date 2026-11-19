# Notes — direct-edit repetition experiment (inference only)

No backward/autograd/optimizer/numerical gradients/sweeps. Every model run went through experiment_client.py (MCP). CPU numpy was used only to read receipts and to build explicit JSON values.

## Observation (flow-01, source)
- Positions: 16 induction-eligible "copy" positions (the current bigram appeared earlier, and the next token matches the earlier continuation) and 16 nearby fresh-token positions, in 4 discovery reference stories (08537, 08570, 09488, 13123). These same ids loop in baseline greedy generation.
- Per-head attention output norms (out_proj inputs, head h = columns 4h..4h+3):
  - L4H14: mean .149 vs .079, but only because of 2 outliers.
  - L5H5: .166 vs .100, a consistent shift.
  - L7H15: .152 vs .091, a consistent shift.
- Copy positions already had high target log-probability (median about -0.05), so copying is often correct in natural text. This is the main alternative explanation: damaging copy signal could hurt reference NLL.

## Iteration 1 — candidate-01 (from source)
- Proposal: analysis-01 and proposal-01 (ready_for_causal_probe).
- Edit: L7 out_proj columns 60-63 ×0.7. The coefficient is bounded by the rel-L2 budget, since the head holds 40% of the tensor norm.
- Local prediction: projected delta .0077 at copy positions vs .0043 at fresh positions.
- Result: validation repeat .0693 → .1027 (worse); NLL +.0038.
- Teacher-forced log-probability changes (flow-02): copy -.044, fresh -.039, so not selective. **Disconfirmed.**

## Iteration 2 — candidate-02 (child of 01)
- Opposing challenge: the same block set to 1.3× source.
- Result: repeat .0831 (still worse than baseline); NLL +.0028. Per-prompt outcomes flipped both ways.
- Conclusion: L7H15 is not a selective loop driver. Effects look like chaotic changes to the greedy trajectory.

## Iteration 3 — candidate-03 (child of 02)
- Rollback of the L7 block to the exact source values, plus L5 out_proj columns 20-23 (head 5) ×0.5 (rel L2 .142).
- Result: repeat .0464 (gain .0228); NLL +.0098; accuracy -.0014; short-rate 0; median length 128. All development gates passed, narrowly.
- Teacher-forced (flow-04 vs flow-01): copy -.041 vs fresh -.015. This is copy-selective and consistent with the hypothesis, but it does not prove the head's role.
- Fragility: most of the gain comes from tsval-00888 (.562 → 0). Two new loops appeared (05226 → .20, 08278 → .22). When candidate-03 was frozen, no matched control had been run yet. See the post-freeze control below.

## Freeze
candidate-03 is frozen for host-only confirmation (decision.json). Heldout was never accessed. Validation was used for selection, so it is development evidence only.

## Post-freeze control — candidate-04 (from source)
- Edit: L5 out_proj columns 8-11 (head 2) ×0.5. Head 2 has a copy/fresh norm ratio of .99 and norm share .254.
- Result: repeat .072 (gain -.003); NLL +.0185; gate failed.
- Teacher-forced: copy +.016, fresh -.072, so not copy-selective.
- Reading: this is consistent with candidate-03's effect being specific to head 5, but it is a single control. It was run after decision.json was frozen, and the choice was not changed.
