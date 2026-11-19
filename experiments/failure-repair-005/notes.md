# failure-repair-005 notes (Claude worker)

## Iteration 1 (written before candidate-01)
Observation (flow-01, source, CUDA trace of h5/h6/h7 mlp.c_proj inputs, all 256 hidden units, pairs tsval-12237 pos46-49 and tsval-19392 pos67-70):
h7 hidden units 113 and 157 had failure>reference post-GELU activation at 8/8 matched positions (mean fail 0.196/0.240, ref -0.026/-0.003). No h5/h6 unit was consistent at 8/8.
analysis-01 (4 pairs incl. two NOT used for selection: tsval-05900, tsval-01945): in tsval-05900 both units are NEGATIVE in failure (-0.08..-0.16), and tsval-01945 is mixed. So the support is limited to the selecting pairs (selection bias) -> hypothesis weak.
Hypothesis: h7 MLP writer columns 113,157 add a loop-continuation component to the final residual in some loops.
Value derivation: attenuate those two columns of transformer.h.7.mlp.c_proj.weight (rows 0..63) by fraction 0.5 (new=0.5*old). Fixed-input local check: mean failure projected delta 0.01175 vs reference 0.00022 along -mean(failure contribution); max reference delta norm 0.029 (bound declared 0.05 AFTER seeing analysis; disclosed).
Prediction: small validation repeat decrease, NLL change <0.01. Given delta norm ~0.04, expect possibly no measurable behavior change.
Disconfirming: validation repeat gain <=0 or NLL increase >0.01; post-edit target log-prob for loop tokens unchanged.

## Iteration 1 feedback (candidate-01)
Validation: repeat gain +0.0007, CI [-0.028,0.029], LOO -0.004, reference NLL +0.0205 (FAIL >0.01), acc -0.0045. Many prompts regressed (e.g. tsval-13579 +0.40, tsval-20845 +0.30). Post-edit analysis-02: loop target logprob fell at 12237/49 (-0.87) and 19392/69 (-0.18) but rose elsewhere; reference 19392/67 fell -0.43. Hypothesis (general loop writer) disconfirmed for the pair of columns.

## Iteration 2 (written before candidate-02)
Observation: col113 failure activation is negative in held-out tsval-05900 and 0.352 on reference 12237/49; it is the less selective unit. col157 is the 8/8 consistent one with larger effect on 19392.
Hypothesis: NLL damage is mainly from col113; keep only col157 attenuation.
Value: restore col113 rows0-63 to original (expected = parent 0.5*old FP32), keep col157 at 0.5*old.
Prediction: NLL increase falls well below 0.0205; repeat gain small.
Disconfirming: NLL increase stays >0.01 or repeat gain <=0.

## Iteration 2 feedback (candidate-02)
Validation: repeat gain -0.0048, CI [-0.037,0.025], LOO -0.0099, NLL +0.0080 (pass), acc -0.0012; 23 prompts regressed. analysis-03: loop target logprob dropped at 12237/49 (-0.23) and 19392/69 (-0.17) but reference 19392/67 dropped -0.26. Even 64-weight halving costs 0.008 NLL: col157 is used for normal prediction too. Loop-selective-writer hypothesis disconfirmed at the generation level.

## Iteration 3 (written before candidate-03)
Observation: no repeat benefit from col157 attenuation; NLL cost 0.008 (close to bound).
Hypothesis: no supported repetition mechanism remains in this edit; the only feedback-justified change is to reduce collateral damage (col157 attenuation 0.5 -> 0.25 of removal, i.e. new=0.75*old) to keep a low-harm diagnostic candidate for the frozen confirmation slot.
Prediction: NLL increase < 0.008; repeat gain ~0 (not expected to pass gates).
Disconfirming: NLL increase not reduced, or larger generation regressions.
No opposing control will be run (time budget); disclosed as unmet.

## Iteration 3 feedback (candidate-03) and decision
Validation: gain +0.0145, CI [-0.006,0.035], LOO +0.0100, NLL +0.0022, acc -0.0007, 13 regressed. Gate failed (min gain, CI). Non-monotonic dose ordering suggests greedy-trajectory noise. decision.json: candidate-03 frozen as no_gain_diagnostic. No control run (unmet).

## Iteration 4 — opposing norm-matched CONTROL (written before candidate-04; second/last invocation)
Purpose: this is not a primary repair attempt. It tests whether the col157 direction is sign-specific. Host review flagged it as an open finding.
Target: the same final selected writer as candidate-03, `transformer.h.7.mlp.c_proj.weight` (Linear, layout out,in), rows 0..63, column 157 only. Column 113 stays at its source value.
Value: candidate-03 has delta = -0.25*old on col157. The control is a source branch made with create_candidate/set_values: new = 1.25*old, so delta = +0.25*old. It has the same L2 norm and the same 64 coordinates, with the opposite sign. Old values are the source values recorded in analysis-01 original_values[:,1] (= candidate-01 expected_values). No other values were tried, and there is no sweep.
Local arithmetic: at fixed inputs the module-output delta is exactly the negation of candidate-03's. That is linear by construction, so it is not evidence. The post-edit analysis checks the downstream target log-prob.
Candidate-03 observed target log-prob deltas (analysis-04 vs analysis-01): fail 12237/49 -0.107, 19392/69 -0.067, 19392/70 +0.045, 12237/48 +0.035. Ref 19392/67 -0.128, 12237/46-47 -0.046.
Competing predictions:
- H_loop_writer (attenuation was the right sign; col157 writes loop continuation): candidate-04 should roughly mirror candidate-03 locally. Fail 12237/49 should go to about +0.1 and 19392/69 to about +0.07. Validation repeat should get worse (gain <= -0.01).
- H_wrong_sign (amplifying col157 is the loop-reducing direction): candidate-04 repeat gain should be larger than candidate-03's (+0.0145), with fewer regressed prompts.
- H_nonspecific (col157 is a general writer; generation changes are greedy-trajectory noise): candidate-04 gain should be near 0 with a CI spanning 0, and the regressed-prompt count should be comparable (about 13-28). The reference NLL should RISE again (both directions damage, i.e. a second-order effect near the pretrained point) rather than fall by about 0.002 (a first-order mirror).
Both directions damaging normal prediction is explicitly allowed for.
Predeclared decision rule: the protocol gates are unchanged. Candidate-04 is chosen only if it passes ALL validation gates. Otherwise candidate-03 stays the frozen diagnostic, because a control is not reselected as a repair.
Predeclared qualitative inspection: for candidate-03 and candidate-04, take the validation prompts with (a) the largest gain, (b) the largest regression, and (c) one prompt whose metric is unchanged. Compare the decoded continuations with the baseline.

## Iteration 4 feedback (candidate-04 control): predicted vs observed
Write: create_candidate on the source branch, col157 rows0-63 = 1.25*old. 64 params changed, relative_l2 0.014984 (candidate-03: 0.014984, so norm-matched). patch 655877e1…
Local post-edit (analysis-05, compared with analysis-01): an almost exact first-order mirror of candidate-03. Fail 12237/49 +0.093 (H_loop_writer predicted about +0.1), 19392/69 +0.036 (predicted about +0.07; partial), 19392/70 -0.056, ref 19392/67 +0.127. Teacher-forced local effects are sign-symmetric, as linearity implies, and small (|dlogp| <= 0.13).
Full 64-prompt validation, same protocol and gates: repeat gain +0.0231, CI95 [-0.0080, 0.0651] (FAIL), leave-largest-out +0.0081, reference NLL +0.0015 (pass), acc +0.0003, short 0.0, median 128, regressed 16 / improved 27. Decision gate_failed (CI lower bound <= 0).
For comparison, candidate-03: gain +0.0145, CI [-0.0060, 0.0350], leave-largest-out +0.0100, NLL +0.0022, regressed 13 / improved 23.
Interpretation (a hypothesis, not proof):
- H_loop_writer is contradicted at the generation level. Its local logit prediction held, but repetition did NOT rise; the mean gain is positive in both directions.
- H_wrong_sign is not supported. The higher mean comes mostly from one prompt (tsval-02052: baseline 0.967 -> 0.000 is about 0.015 of the 0.023 mean). Leave-largest-out (+0.008) is LOWER than candidate-03's, the CI spans 0, and more prompts regress.
- H_nonspecific is best supported. Opposite perturbations give per-prompt gains with the same sign on 20 prompts and opposite signs on only 6 (correlation 0.17). Reference NLL rises in BOTH directions (+0.0022 / +0.0015), i.e. second-order damage around the pretrained point and not a first-order mirror. A small change to col157 either way shifts greedy trajectories; it does not target a loop mechanism.
Qualitative inspection (predeclared picks, decoded saved continuations, first ~90 tokens):
- c03 largest gain, tsval-14655 (0.322->0.025): base loops "She was so excited to see all the things she could see." x2; cand diverges after that point ("She ran and jumped, and around, until she was tired…" — note the ungrammatical "and around").
- c03 largest regression, tsval-17667 (0.081->0.444): cand skips a story segment and enters the baseline's own "He ate the apple and ate it all up. He was so happy." loop earlier, then ends early ("The end."). The SAME loop arrives earlier; no new loop.
- c03 unchanged, tsval-10907: tokens identical to baseline, which still loops ("Can I have some candy?" … x2).
- c04 largest gain, tsval-02052 (0.967->0.000): the degenerate "and her cat, and her cat, …" loop is replaced by a coherent-looking party story (still with internal repetition such as "have a party for you"/"party").
- c04 largest regression, tsval-00009 (0.000->0.260): store/necklace story diverges into a NEW loop: "She climbed up the tree and climbed up the tree. She was very happy." x3. One loop is swapped for another.
- c04 unchanged, tsval-20845: tokens identical to baseline.
Conclusion: across both directions, the generation changes look like trajectory divergence that can remove, delay or create loops, not like selective loop suppression. The 8-gram metric is not a quality judgment; no coherence claim.
Decision (applying the predeclared rule): no candidate passes the gates. Candidate-04 is a control and is not reselected. Candidate-03 remains the frozen diagnostic (validation_eligible=false). Model work ends here.
