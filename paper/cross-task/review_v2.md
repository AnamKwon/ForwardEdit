# Review of `manuscript_v2.md` (2026-11-24)

Reviewer role: independent, skeptical (strict ML-venue reviewer plus research-integrity audit). No GPU was used. The manuscript was not edited. All raw-evidence checks below were done read-only against host JSON/log files under `D:` = the host data root (`qcf15_4b`, `qcf16_4b`, `qcf17_4b`, `qcf20_9b`, `qcap_08b`, `bench/regress_*.json`, `defect_audit/scan_4b_{off,on8k}.json`, `quant/*.json`) and the scripts that produced them (`bench/baseline_ladder_cf.py`, `bench/gen_eval_cf.py`, `bench/lora_compare_4b.py`, `defect_audit/scan.py`). I did not use the manuscript's ledger (`ledger_1124.md`) for any spot check.

---

## Summary verdict

**Weak reject in its current form (would be "major revision" at a journal).** It is honest in tone and unusually thorough about failed gates, and almost every number I checked matches the raw files. However:

1. **The headline mechanistic claim is weaker than the abstract and title say.** "Gated sparse memory writes" rests on two validation-only ladders. Those ladders come from 3-role, recipe-informed arms, and the analysis was added after their sealed results were known. The logit-bias and steering grids start at strength 1 and S falls monotonically from there, so the "best" baseline is a grid edge that does worse than no edit at all. The "always-on steering" arm uses the **mean of 40 near-orthogonal per-fact writes** (pairwise cosine about −0.02), which is not "the edit's own write vector." The gate-activity percentages depend on a per-arm relative threshold that the paper does not disclose. Section 6.1 then makes a causal claim ("because of *when* they fire, not because of the direction they write") that no intervention tested. The oracle arm, which uses the same direction with a gate, also loses to the edit, so the paper's own data argue against that sentence.
2. **There is a data-role blur the paper does not mention.** In every CounterFact arm, all 40 sealed ES prompts appear verbatim in discovery: they are the edit prompts. Sealed S therefore averages one edit-construction component with two held-out components. Separately, 2–11 sealed neighbourhood prompts per arm also occur in discovery, validation or confirmation. In qcf17 one sealed neighbour ("Dennis Rodman plays" → basketball) is the edit prompt of another case (→ hockey).
3. **The criterion-1 passes that support "task improvement" are within noise.** The margins are .012 (qcap_on), .013 (qcf20) and .026 (rep1). The unpaired SE of S at these n is about .02. The base model's S also moves by up to .006 between two evaluation runs of the same item set. On the strongest route itself, qcap_on's edit *lowers* S (all_x2: .8339 vs .8364). With a 2/5 replicate pass rate, every single-run 4B/9B statement needs an explicit "one draw from a distribution with roughly 40% pass rate" qualifier.
4. **Several framing errors.** qcf20 is called "the first freeze under the four-role split", but qcap_on and three other qcap runs froze under four roles on 11-23. The no-context generation comparison pools edits designed for context (qcf15/16) with the one no-context edit (qcf17). "Near-official settings" is used for greedy decoding, which the Qwen model card advises against in thinking mode.

The comparator section (Sections 4.3 and 6.2) is the paper's strongest and most useful contribution. Its main message is that fitted editors beat LLM-arithmetic values without context, and that the editor seems to choose a good location but sets weak values. That message is well supported and honestly reported. A version re-centred on that message, plus the protocol and its failure modes, with the mechanism claim downgraded and the controls in the "experiments" list added, could be a solid workshop paper. It is not yet a main-track paper.

---

## 1. Numerical spot checks (raw JSON, not the ledger)

| # | Location | Manuscript | Raw value (file) | Status |
|---|---|---|---|---|
| 1 | Abstract, Table 1 | edit validation S .903 / .860 | .9025 / .8599 (`qcf15_4b`, `qcf16_4b` `baseline_ladder.json` arms.E.S) | OK |
| 2 | Abstract, Table 1 | best LB .536 / .423 | .5363 / .423 (LB(1.0)) | OK as numbers, but LB(1.0) is the smallest grid value; see MF-3 |
| 3 | Abstract, Table 1 | best ST .684 / .674 | .6836 / .6741 (ST(1.0)) | OK as numbers, but grid edge; ST = mean of 40 writes; see MF-3 |
| 4 | Abstract, Table 1 | gate active on generic text 0.0% / 0.68% | 0.0 / 0.0068 (`gate_activity`; thresholds 2.245 and 0.544 = 0.1 × median subject-write norm) | OK; threshold undisclosed (MF-4) |
| 5 | Table 1 | non-subject active 0.63% / 7.91% | .0063 / .0791 | OK |
| 6 | Table 1 | Oracle steering .759 / .729 | .7585 / .7291 | OK |
| 7 | Table 1 | "S" | Validation S = harmonic mean of (PS, NS, plain-NS), **no ES**, e.g. HM(.9375, .925, .85) = .9025 | **Definition mismatch** with Section 3.2 (MF-5) |
| 8 | Table 2 | qcf15 ST PS .663/.650/.675/.775; MMLU .735/.495/.215/.270 | identical | OK |
| 9 | Table 2 | qcf16 ST PS .638/.600/.625/.650; MMLU .700/.690/.195/.255 | identical | OK |
| 10 | Abstract, Table 3 | edit no-ctx sealed S .682 | .6823 (`qcf17_4b/sealed_eval.json`) | OK |
| 11 | Abstract, Table 3 | strongest prompt .763 | all_x2 .7629 | OK |
| 12 | Abstract, Table 3 | ROME .806; MMLU .7015; ARC .9087; all-ctx .721 | .806 / .7015 / .9087 / .7209 (`memit_sealed_ROME.json`) | OK |
| 13 | Abstract, Table 3 | MEMIT .897; all-ctx .920; MMLU .7215; ARC .9044; cov. tokens 124,537 | .8969 / .9202 / .7215 / .9044 / 124537 (`memit_compare.log` line 24 and line 4) | OK (log only, as disclosed) |
| 14 | Abstract, Table 3 | LoRA .898 (1.00 / .7875 / .935), all-ctx .927; MMLU −0.25 ± 0.47; ARC 0.0 ± 0.34 | .8984, identical components, .9271, identical | OK |
| 15 | Table 3 | LoRA chat rows .518 (.875/.4125/.45), all .675, MMLU −0.1 ± 0.25 | .5182, identical, .6746, identical | OK |
| 16 | Table 3 | N-edit .884, all .929; N-orig .668, all .890; MMLU .7285 / .719 | .8836, .9286, .6682, .8895, .7285, .719 | OK |
| 17 | Table 3 | **Original MMLU-2000 .7255; ARC-C ≈ .913** | Paired regression original (`bench/regress_qcf17_4b.json`): **MMLU .7245, ARC .9121**. .7255 is the comparator harness's `general.O`; .913 is the ROME script's `general_O` ARC, which equals the *edit's* ARC in regress | **WRONG / inconsistent** (MF-11) |
| 18 | Table 3 | edit regression MMLU +0.15 ± 0.19; ARC +0.09 ± 0.19; GSM +1.0 ± 1.41 | identical (`regress_qcf17_4b.json`) | OK |
| 19 | Abstract, Table 4 | edit no-ctx generation 1.25–3.75% | qcf16 .0125, qcf17 .025, qcf15 .0375 (`gen_eval.json`) | Numbers OK; **pooling is not comparable** (MF-6) |
| 20 | Abstract, Table 4 | MEMIT .2625, ROME .2875, ROME neighbours-not-new .9125 | identical | OK |
| 21 | Table 4 | stored ctx .2375→.425, .1625→.40; neighbour not-new .975/.975; neighbour old .3375/.4125 | identical | OK |
| 22 | Table 5 | qcap_on S .848, PS +13.75 ± 4.24 (12/1), NS 0.0 ± 1.53, margin .012 | .8484, identical, .8484 − .8364 = .0120 | OK |
| 23 | Table 5 | qcap_off .813, +6.25 ± 3.23 (6/1), NS −2.5 ± 1.23 (0/4); ARC −1.28 ± .52 | identical (`sealed_eval_qcap_off_claude.json`, `compare_reps.md`) | OK |
| 24 | Table 5 | rep1 .863, +17.5 ± 4.6 (15/1), NS +0.62 ± 1.08 | .8627, identical | OK |
| 25 | Section 6.4 | **rep1 margin .027** | .8627 − .8364 = **.0263 → .026** | **WRONG (rounding)** |
| 26 | Table 5 | rep2 .784, 0.0 ± 3.06 (3/3), NS −1.25 ± 1.25 | .7837, identical | OK |
| 27 | Section 4.4 | largest frozen change .0163%; no candidate > 2% | .0001633; max candidate .004878 (0.49%) (`compare_reps.md`) | OK |
| 28 | Table 8 | qcf20 sealed S all .784→.836; all_x2 .823/.852; none .037/.059; oracle .747/.787; override .754/.851 | identical (`qcf20_9b/sealed_eval.json`) | OK |
| 29 | Table 8 | PS +16.25 ± 4.12 (13/0); NS 0.0 ± 2.93 (11/11); all_x2 NS −3.75 ± 2.32; override NS +5.62 ± 3.09; oracle ES −5.0 ± 3.45 | identical | OK |
| 30 | Abstract, Section 4.5 | ARC −0.85 ± 0.38 (5/15; 2.2 SE); MMLU −0.75 ± 0.46 (34/49); GSM −2.0 ± 3.08; originals .7585 / .9386 / .745 | identical (`bench/regress_qcf20_9b.json`) | OK |
| 31 | Footnote bar | pre-launch all_x2 .815 | .8149 (`qcf20_9b/baseline.json`) | OK |
| 32 | Section 4.1 | qcf15/16 sealed S .922 / .894; best no-edit .760 / .785; LoRA .989 / .987; qcf16 LoRA MMLU −1.55 ± 0.77 | .9217 / .8943; .7601 / .7852; .9891 / .9874; −1.55 ± 0.77 | OK |
| 33 | Abstract, Table 10 | 162 of 201 truncations are loops | truncated ∩ loop = 162 of 201 (`scan_4b_on8k.json` records) | OK |
| 34 | Table 10 | overall .842 / .777; finished .903 (916/1,014) / .949 (853/899); thinking-off loops 2/86 | identical | OK |
| 35 | Table 10 | per-task acc/truncated/finished (IFEval, MATH, MMLU, GSM, ARC) | all identical (e.g. MATH off 53/56, on 40/42 = .952) | OK |
| 36 | Section 4.6 | 117 truncated IFEval+MATH, 1 closed `</think>`; 46 wrong finished: arith 12, distractor 10, loop 6, unlabeled 19 | identical | OK |
| 37 | Section 4.6 | "A loop is a 16-token repeated prefix" | rule `window16_run64`: a 16-token window that repeats at ≥64 consecutive start positions (`scan.py:detect_loop`) | **Imprecise** (SF) |
| 38 | Table 7 | bf16 +35.0 (.675→.906); int8 +33.75 (.679→.885); nf4 +36.25 (.664→.890); keep-bf16 +33.75 (.670→.886) | PS .5875→.9375 etc.; S identical (`quant_edit_check_4b.json`) | OK |
| 39 | Section 4.5 | delta cos ≥ .9845 nf4, ≥ .996 int8; qcf18 dense cos .751, norm ×1.326, rel. .14; int8 .9945 | .9845 / .9962 / .751 / 1.3261 / .1402 / .9945 | OK |
| 40 | Section 4.5 | overlay smoke .708→.958 on 24 | .7083→.9583 | OK |
| 41 | Table 9 | 9B 8.04 / 15.04 GB; 27B nf4 19.11 GB, 33.1 s; offload 21.24 GB, 61.3 s, 22/46; NLL gap .002–.277; MoVA 8.62 GB, 84/288, 26.8B non-4-bit, 74.5 s, .116/.141/8.49 | identical (`quant_9b_smoke.json`, `offload_27b_smoke.json`, `mova_smoke.json`) | OK |
| 42 | Section 6.2 | "stayed within the 2 SE regression bounds … in 2 of 5 runs at 0.8B" | Regression was computed for 3 runs (on, off, rep1), and 2 were clean. rep2 was not computed; rep3 did not freeze | **Misleading denominator** (MF-12) |
| 43 | Sections 1, 4.5, Table 13 | "first freeze under the four-role split" (qcf20) | `qcap_08b/dev.json` has a `confirmation` split (430 rows). qcap_on, qcap_off and rep1/rep2 froze on 11-23; qcf20 launched 11-24 | **WRONG** (MF-7) |

Also noticed:

- The LoRA validation-selection metric in `qcf17_4b/lora_compare.json` shows S = 1.3776 > 1. This comes from a since-fixed bug (`3/Σ` used with 2 sources; see the comment at `lora_compare_4b.py:60`). It rescales HM by a constant 1.5, so the argmax is unchanged, but it should be disclosed.
- No-edit sealed S differs between two evaluation passes of the same items: qcf17 all_x2 is .7629 in `sealed_eval.json` and .7598 in the LoRA run; qcf16 all is .7221 vs .7285. Validation bf16 edit S is .9025 in the ladder and .9064 in the quant check. So the harness has roughly ±.003–.006 S of run-to-run non-determinism, which matters at margins of .012.

**Net:** 37 of 43 checks are exact. The errors are one wrong cell (Table 3 "Original"), one rounding error (.027), one wrong "first" claim, one misleading denominator, one undisclosed metric-definition change, and one imprecise loop definition. Values marked ‡ (Sparks, gradient-audit counts, rep3, qcf18/19) could not be checked from the listed raw sources.

---

## 2. MUST-FIX items

### MF-1. Title overstates the evidence

> "Gated Sparse Memory Writes, Not New Capabilities: A Pre-Registered Four-Role Study of Inference-Only LLM Weight Editing"

Problems with the current title:

- The "gated sparse memory write" evidence comes from the 3-role arms qcf15/16. Those arms had recipe notes, and their ladder was a host analysis run after the sealed results (the ladder script is dated 11-23; neither qcf15 nor qcf16 has a ladder pre-registration).
- Only qcap (0.8B) and qcf20 (9B) used four roles, and neither has a ladder.

**Replace with**, for example: "Inference-Only LLM Weight Edits for Counterfactual Facts: Sparse, Subject-Selective, and Outperformed by Fitted Editors." Alternative: "Where LLM-Written Weight Edits Stand: Pre-Registered Comparisons with Prompting, Steering and Fitted Editors."

### MF-2. Abstract: the mechanism claim needs its scope and provenance

> "On 4B CounterFact rewrites with facts in context, frozen edits behave as gated sparse memory writes."

**Replace with:**

> "On two 4B CounterFact arms run with recipe notes from earlier sessions, and with facts in context, a post-hoc host analysis on the validation split (on which the edits were also selected) found that the frozen edits change the residual stream at the edited layer almost only at subject tokens. In this analysis they outscored every tested strength of a global logit bias and of an always-on steering vector. None of those tested strengths improved on the unedited model."

### MF-3. The baseline ladder grids are truncated, and "always-on steering" is not the edit's write vector

> "Best global logit bias, LB(1)" / "Best always-on steering, ST(1)" / "the edit's write vector added at s× strength at every token" / "Always-on steering with the edit's own write vector."

What the raw files and script show:

- `baseline_ladder_cf.py` tries only b ∈ {1, 2, 4, 8} and s ∈ {1, 4, 16, 40}.
- S decreases monotonically from b = 1 (qcf15: .536 → .355 → .306 → .190).
- Both LB(1) arms score **below no edit** (.536 < .675 and .423 < .680), and ST(1) is within ±.01 of no edit. The true optimum of each family on this grid is therefore at or near strength 0, which is the unedited model.
- `ST(s) = s · mean_i w_i` (script line 127). This is the **average** of 40 per-fact writes with mean pairwise cosine −0.023/−0.014. At s = 1 each fact's own component is about 1/40 of its write. So ST tests "one global vector for 40 different facts", which no critic proposed and which cannot work by construction.

**Required changes:**

- Rename the rows to "Logit bias, smallest tested b = 1 (S falls monotonically with b; no tested b beats no edit)" and "Steering with the mean of the 40 per-fact writes, smallest tested s = 1."
- In Section 3.5, state the grids and the selection rule (MMLU-200 within 2 points).
- Either add b, s ∈ {0.125, 0.25, 0.5} or state that the best of each family is indistinguishable from no edit.
- Replace "the edit's own write vector" with "the mean, over the 40 facts, of the edit's per-fact write at the subject token."

### MF-4. Gate activity needs its threshold and sampling disclosed

> "The edits fire on 0.0% and 0.68% of generic-text tokens."

The script counts a token as active when the edited layer's output changes by more than 0.1 × the median subject-write norm. The thresholds are 2.245 (qcf15) and 0.544 (qcf16). The generic text is 100 MMLU questions truncated to 256 tokens. "Non-subject" masks only the last 4 subject tokens.

**Add to Section 3.5 and the Table 1 caption:** "A token is counted active when the edit changes the layer-8 block output by more than 10% of the median subject-token write norm (2.245 for qcf15, 0.544 for qcf16). Generic text is 100 MMLU questions (≤ 256 tokens each). The subject mask covers only the last 4 subject tokens." Also report the active fraction at 1%, 5% and 20% thresholds (cheap; see E-3).

### MF-5. The validation S is a different metric from the S defined in Section 3.2

Section 3.2 defines S = HM(ES, PS, NS). The ladder, LoRA/MEMIT/ROME selection and quantization tables use HM(PS, NS, plain-NS) on validation (no ES), or HM(PS, NS) for qcf17. **Add:**

> "On the validation split, S is the harmonic mean of the available sources: PS, NS and plain-NS for qcf15/16, and PS and NS for qcf17. It is not comparable in level with sealed S, which includes ES."

### MF-6. The generation-based comparison pools non-comparable arms

> "In generation-based scoring, the edit emits the new object for 1.25–3.75% of prompts, against 26–29% for MEMIT and ROME."

MEMIT and ROME were run only on qcf17. qcf15/16 are all-context designs evaluated out of design at no context. In addition, the metric is `startswith` after normalization (`gen_eval_cf.py:31`). Under that metric the **original model starts with the true object on only 6–9% of no-context prompts** (`rewrite_old_rate` .0625–.0875). The metric is close to its floor and largely measures answer format. One edit sample, " 1623 in France.", contains the new object but scores 0.

**Replace with:**

> "In greedy 8-token generation on the validation split, with no context, the no-context edit (qcf17) began with the new object on 2.5% of prompts, against 26% for MEMIT and 29% for ROME. Under this prefix-match metric, the unedited model begins with its own true object on only 6–9% of prompts. A containment-based metric was not computed."

Add a containment score (CPU-only, from the stored samples if the full outputs were saved; otherwise see E-6).

### MF-7. "First freeze under the four-role split" is false

The qcap arms (0.8B) used discovery/validation/confirmation/sealed and froze on 11-23 (4 of 5 runs). qcf20 ran on 11-24.

- Section 1, contribution 5: replace "the first freeze under the four-role split passed sealed criteria 1–3" with "the first 9B freeze, under the four-role split, passed sealed criteria 1–3."
- Section 4.5 heading: replace "qcf20_9b: first freeze under the four-role split" with "qcf20_9b: first freeze at 9B."
- Abstract: replace "the first four-role freeze" with "the first 9B freeze."

### MF-8. The sealed ES component is edit-construction data, and some sealed neighbours leak

All 40 sealed ES prompts are verbatim discovery `rewrite_nll`/`rewrite` prompts in qcf15, qcf17, qcf20 and qcap. Sealed neighbourhood prompts also overlap dev splits:

| Arm | Sealed NS prompts overlapping dev | Duplicates within sealed |
|---|---|---|
| qcf15 | 7/200 | 3 |
| qcf16 | 0/200 | 1 |
| qcf17 | 2/200 | 1 |
| qcf20 | 2/160 (one in confirmation) | 2 |
| qcap | 11/160 (discovery 8, validation 5, confirmation 5 hits) | 4 |

In qcf17, the sealed neighbour "Dennis Rodman plays" (expected: basketball) is the discovery **edit prompt** of another case that rewrites Dennis Rodman to hockey. The neighbour set was not filtered against edited subjects, so this item cannot be satisfied by a working edit.

This conflicts with `EDIT_EVALUATION_PROTOCOL.md` items 2–3 and the report layout ("item-level … exposure limitations").

**Add to Section 3.2 or 3.3:**

> "ES uses the requested rewrite prompt, which is also an edit-construction (discovery) row. Only PS and NS are unseen wordings, and all sealed items concern the same 40 edited facts. 0–11 sealed neighbourhood prompts per arm also appear in a development split, and one qcf17 neighbour targets an edited subject. Section X reports the PRIMARY criteria with these items removed."

The re-scoring is a sensitivity analysis on the frozen candidate, not a reselection. Pre-register it and label it as such (E-1).

### MF-9. Section 6.1 makes a causal "when, not what" claim without the intervention

> "…and beat logit bias and constant steering because of *when* they fire, not because of the direction they write."

The oracle arm writes the same per-fact direction at the subject token and still loses (.759 vs .903). ST used a different direction (the mean of 40). So direction and gating were never separated. Section 4.2 itself says "Two things separate the edit from this oracle: when it fires, and the size and shape of its write."

Also note that `add_oracle` injects only at the subject in the query, never at subject tokens inside the context. The edit fires there too, so a context-side effect is a live alternative explanation for the large in-context NS gains (sealed all-ctx NS +27 / +22 points).

**Replace with:**

> "They outscored subject-agnostic logit bias and a single always-on vector. They also outscored the same per-fact write injected only at the queried subject. The edit's advantage over that oracle may come from firing at subject mentions inside the context, or from the size and shape of its write. We have not separated these."

### MF-10. Criterion-1 margins have no uncertainty, and qcap_on does not improve its own strongest route

> "qcap_on cleared the bar by .012" / "Criterion 1 passes (.836 > .823, margin .013)" / Section 6.4 "only qcf15, qcf16, qcap_on and rep1 achieved *task improvement* …"

A delta-method approximation gives an unpaired SE of each sealed S of about .020–.023 at n = 40/80/160, so the margins are about 0.4–0.9 SE of the difference. Evaluation non-determinism alone moves S by up to .006. Also, the edit applied **on the all_x2 route** gives qcap_on .8339 < .8364, and it costs NS −5.0 ± 1.94 there (2.6 SE; qcf20 all_x2 NS −3.75 ± 2.32; rep2 all_x2 NS −13.75 ± 2.72). The pre-registered verdicts must stand.

**Add after Table 5 and Table 8:**

> "Criterion 1 compares point estimates without uncertainty. The margins (.012, .013, .026) are below one standard error of the S difference (about .03, unpaired) and only about two to four times the evaluation's own run-to-run variation (≤ .006). For qcap_on, applying the edit together with the strongest prompt route lowers S (.834 vs .836) and costs neighbours (NS −5.0 ± 1.9). The pre-registered verdicts are unchanged, but these passes should not be read as reliable improvements."

In Section 6.4, replace "achieved task improvement over the strongest measured no-edit baseline on a sealed set with all criteria met" with "met all pre-registered criteria on a sealed set (two of the four by margins below one SE)."

### MF-11. The Table 3 "Original" row mixes harnesses

> "Original | … | MMLU-2000 .7255; ARC-C ≈ .913"

The paired regression harness gives the original model .7245 / .9121. The comparator harness gives .7255 / .9121 (LoRA, neuron) and .7255 / .913 (ROME script). The ROME/MEMIT/N-arm deltas (−2.4, −0.4, unpaired) are computed against .7255. **Replace with:** "MMLU-2000 .7245 (paired harness) / .7255 (comparator harness); ARC-C .9121." Add a footnote: "Unpaired comparator deltas use the comparator harness's original value. The two harnesses differ by 0.1 MMLU points on the same model, which is a quarter of the MEMIT difference."

### MF-12. Section 6.2 regression denominator

> "stayed within the 2 SE regression bounds at 4B (qcf17) and in 2 of 5 runs at 0.8B."

**Replace with:** "stayed within the 2 SE regression bounds at 4B (qcf15, qcf16, qcf17) and in 2 of the 3 0.8B runs whose regression was computed (rep2's was not computed; rep3 did not freeze)."

### MF-13. "Near-official settings" describes a setting the model card advises against

> "At near-official settings, the dominant defect of the 4B model is thinking-mode repetition looping under greedy decoding." (contribution 6, Section 4.6, Table 10 header, Section 5 #19)

Qwen recommends sampling (T 0.6, top-p 0.95, top-k 20) for thinking mode and warns against greedy decoding. The manuscript itself says so in Section 4.6. **Replace "near-official" everywhere with** "thinking mode with greedy decoding and an 8,192-token cap (the model card recommends sampling instead)." Downgrade contribution 6 to "We confirm the model card's warning: under greedy thinking-mode decoding, …".

### MF-14. Five "same-condition" runs: disclose that the pooling was decided after the results

The qcap PREREG defines qcap_on and qcap_off as **different conditions** (cap 0.02 vs 1.0, visible to the editor through `protocol.json` and tool errors). The reclassification as "effectively two runs of the same condition" appears in the outcome section, written after the results. **Add to Section 4.4:**

> "Strictly, the capped condition has four runs (qcap_on, rep1–3), of which 2 passed. Counting qcap_off as a fifth run of the same condition is a post-hoc decision, justified because the cap never bound."

Keep "2/5" only alongside "2/4 strictly."

### MF-15. Multiple-comparison structure of the PRIMARY rule

Criterion 4 runs three tests at 2 SE each. For an edit with no true effect, the chance of a false FAIL is about 1 − 0.977³ ≈ 7% one-sided per benchmark set (about 13% two-sided). qcf20's ARC at 2.24 SE would pass under Holm/Bonferroni (2.39 SE for 3 tests). Criterion 3 ("NS loss not significant") is a non-significance test that passes trivially at low power. The pre-registered verdict must not change. **Add one paragraph to Section 3.2 or Section 7:**

> "Criterion 4 applies three uncorrected 2 SE tests. Criterion 3 is a non-significance condition with limited power (NS SE 1.0–3.0 points). We keep the pre-registered verdicts. Under a Holm correction, qcf20 would pass criterion 4; we report this only to show how sensitive the verdict is to the rule."

### MF-16. Recipe provenance belongs in the abstract

The 4B PASS results (qcf15/16) were recipe transfer from qcf12–14, which passed PRIMARY 3/3 at 0.8B with the recipe. Blind runs at 0.8B passed 2/5, and blind runs at 4B gave 0/2 sealed results. The abstract states none of this. **Add one sentence to the abstract:**

> "The two 4B edits that passed were made with recipe notes from earlier sessions. Blind runs produced no sealed result at 4B (0/2 froze) and passed 2 of 5 at 0.8B."

### MF-17. Section 4.3: "The edit remains strongest on the all-ctx route" is unsupported

> "The edit remains strongest on the all-ctx route (.939, against .920–.927 for MEMIT and LoRA)."

The differences of .012–.019 have no SE, the comparators have one seed each, and N-edit reaches .929. **Replace with:** "On the all-ctx route the edit (.939) is comparable to MEMIT (.920), LoRA (.927) and N-edit (.929); the differences are within the noise of these single runs."

### MF-18. Hygiene: untracked files contain machine-specific paths

`paper/cross-task/ledger_1124.md` (lines 5, 189, 240, 280) contains absolute host paths, a symlink target into a personal workspace root, a GPU index and a different checkout path (`wsl-v2-dev`). `paper/figures/make_figs_1124.py:23` hard-codes `ROOT = '/data/…'`. These files are currently untracked (`git status`: `??`).

- Before any commit, read the root from an environment variable (e.g. `WSL_EVIDENCE_ROOT`) with no default.
- Keep the ledger host-only, or redact its paths to logical prefixes.

No tokens, keys or credentials were found in `paper/`. The bib keys used in the manuscript all resolve.

---

## 3. SHOULD-FIX items

1. **"Hypothesis (stated before the ladder ran)."** Cite the timestamped record (file and time). If only the script docstring exists, write "stated in the analysis script before it was run (11-23); not pre-registered."
2. **"Edit type" contribution (Section 1.2).** The text says "The baseline strengths were tuned on the same split, which favours the baselines." The edits were also selected on that split (blind validation), so both sides are optimistically biased. Replace with "Both the edit (blind validation) and the baseline strengths were selected on this split; no sealed ladder exists."
3. **Loop definition.** Replace "A loop is a 16-token repeated prefix" with "A reply is labelled a loop when a 16-token window repeats at 64 or more consecutive positions." Also report that 51 of 213 loops were in finished replies (IFEval: 80 loops, 59 truncated).
4. **Background table (Section 4.1, † values).** It carries headline-adjacent claims ("6/6", "95%", "LoRA +84.5") that were not re-traced. Either re-trace them or move the table to an appendix and drop it from the narrative of the Introduction.
5. **qcf20 confirmation numbers.** They come from the PREREG outcome section (a host summary), not a JSON. Mark them ‡, or cite the confirmation job JSON.
6. **LoRA selection bug disclosure.** Add to the Table 3 footnote: "qcf17 LoRA selection used S = 3/Σ(1/x) over two sources (values > 1). This is a constant multiple of the harmonic mean and does not change the selected setting."
7. **Gradient audit provenance.** The code search was run on a different checkout (`wsl-v2-dev`) than this repository. State the commit hash or tree hash of the tool code used by each arm, or say that it is unrecorded. "Guard tests 5/5; 269 passed" was not rerun. Rerun the CPU test suite as given in `AGENTS.md` and report passes and skips separately.
8. **Tool-level priors in "blind" arms.** The blind prompts removed recipe notes, but the tools (`errors` returning E_new − E_true, neuron-allocation helpers, low_rank) were built during recipe development. Add to Section 8 that "blind" means no recipe text, not no recipe-shaped tooling.
9. **Patch and split hashes.** `EDIT_EVALUATION_PROTOCOL.md` asks for source/split/patch hashes per role. The `patch_sha256` values exist in each `sealed_eval*.json`; list them in an appendix table along with sealed-file SHA-256s.
10. **Predictions vs outcomes.** REVIEW_GUIDE item 6: qcf15 pre-registered "PRIMARY pass probability about 40%", and qcap pre-registered several predictions. Report these next to the outcomes. This strengthens the paper.
11. **Table 13 "complete with respect to the evidence ledger".** Add the sealed-ES/NS overlap (MF-8), the off-route NS losses (qcap_on all_x2 −5.0 ± 1.94; qcf17 oracle NS −4.0 ± 2.1), and the truncated-grid issue (MF-3).
12. **Section 6.2 "Possibly for footprint and audit."** Auditability is asserted, not measured. Add: "We did not measure whether this record helps anyone audit or revert an edit."
13. **Cut or demote.** Section 4.8 (Sparks small-network reproduction) and most of Section 4.6 (defect audit, Stage 2 pending) are off the paper's main question. Move them to appendices or a separate note; contributions 6 and 8 are not ready for a submission. Table 9's MoVA retry history can shrink to one line.
14. **Section 2.1 and the abstract framing of critique (A).** Barry's steering critique concerns a single-behaviour loop edit. For 40 different facts, a single global vector is not a serious alternative; the oracle arm is the relevant comparison. Say explicitly that the knowledge-edit ladder does not test critique (A) as originally posed.
15. **ES/route reporting.** For each PASS, report the edit applied to the strongest no-edit route as well (it is in the JSON). This is the deployment-relevant comparison.
16. **Tokenizer warning.** Every 4B log prints a transformers warning about an "incorrect regex pattern … incorrect tokenization." It is likely a false positive for Qwen, but state that tokenization was checked.

---

## 4. Missing baselines and controls (ranked; cost on one 24 GB GPU)

| Rank | Control | Why a reviewer will demand it | Cost |
|---|---|---|---|
| 1 | **Specificity controls on the frozen qcf15/16 (4B) and qcap_on/rep1 (0.8B) patches:** (a) sign-flipped values; (b) random value directions at matched norm on the same neurons and keys; (c) key/value permutation across facts (fact i's value at fact j's key, a label-permutation control in the sense of Ahmed). Validation only. | Without them, "subject-gated memory" cannot be told apart from "any sparse write at L8 with this norm." Required by REVIEW_GUIDE item 5. | **Cheap:** CPU arithmetic plus forward-only validation scoring; minutes per arm at 4B bf16 (≈10 GB) |
| 2 | **Per-item sealed re-score plus paired bootstrap CI** of the criterion-1 margin, with sealed NS items that overlap dev or target edited subjects removed. Pre-registered as a sensitivity analysis on frozen candidates, not a reselection. | MF-8, MF-10. Settles whether the .012/.013/.026 passes survive. | **Cheap:** one forward pass over 320–360 items per route per arm |
| 3 | **Corrected ladder:** b, s ∈ {0.125, 0.25, 0.5}; per-fact steering at matched norm without a gate (sum, not mean); oracle injection at *all* subject mentions including the context; and the edit with its effect zeroed at context positions (keep only query positions). | Separates direction, gating and context-side firing (MF-3, MF-9). | **Cheap** (hooks only; same script) |
| 4 | **Sealed ladder** with strengths frozen from validation, run once. | The ladder is the paper's headline and exists only on the selection split. The sealed set is already opened, so the baselines must be frozen first; this is a fixed-baseline comparison, not a selection. | **Cheap** |
| 5 | **Gate-threshold curve** (active fraction at 1/5/10/20% of median write norm; larger generic corpus, e.g. 1k WikiText passages). | MF-4 | **Cheap** |
| 6 | **Generation evaluation with a containment metric and the chat template**, ideally also on sealed (disclosed as a second use) and for qcf20. | MF-6. Mirage-style evaluation is part of the stated contribution. | Cheap at 4B; 9B nf4 is feasible (~8 GB) |
| 7 | **Fitted comparators for the four-role arms:** LoRA/MEMIT/ROME on qcap facts (0.8B) and QLoRA on qcf20 (9B nf4). | Currently "fitted editors win" rests on one 4B arm (qcf17). QLoRA is also needed for the resource argument (Section 6.3, Section 8 #11). | Moderate (0.8B trivial; 9B QLoRA fits in 24 GB) |
| 8 | **Comparator seeds** (3 seeds each for LoRA; ROME/MEMIT hyperparameters tuned for Qwen rather than GPT-J defaults; MEMIT re-run with ~200k covariance tokens and JSON output). | Single-seed comparators; MEMIT log only. | Moderate (GPU hours, no LLM tokens) |
| 9 | **More blind editor replicates at 4B under four roles (≥ 3).** | Every 4B claim is a single, recipe-informed draw. | Expensive in LLM tokens; GPU-feasible. Run one arm at a time, as pre-registered |
| 10 | **Finish defect Stage 2** with the model-card sampling at a 32k cap. | Without it, Section 4.6 has no conclusion. | Moderate (thinking-mode generation is slow; 4B fits) |

---

## 5. Clarity and structure

- **Title and abstract** promise a mechanism result and a pre-registered four-role study. The four-role arms (0.8B, 9B) have no mechanism evidence, and the mechanism arms (4B) are not four-role. Restructure around three questions: (i) how LLM-written edits compare with prompting and fitted editors (Sections 4.3, 4.4, 4.5, the strongest material); (ii) what the protocol learned (failures to freeze, a confirmation pass followed by a sealed null, the fixed-bound problem); (iii) a bounded, validation-only description of the edits' locality.
- **Section 1, eight contributions** is too many for the evidence. Merge them to four: protocol; comparison; variance; locality analysis (validation only). Move scale feasibility, the defect audit and Sparks to appendices.
- **Section 5** largely duplicates the Discussion and Limitations. Keep Table 13, but drop the "Earlier summaries corrected" list from the main text (move it to the appendix or ledger).
- **Section 4.2 Table 1** mixes S (no ES) with the Section 3.2 S. Put the S definition in the caption.
- **Section 3.6** should add columns for "sealed ES = edit prompt?" and "ladder run?" so readers can see at a glance which arms support which claim.
- The **†/‡ convention** is good, but the abstract currently contains no †/‡ items, which is correct. Keep it that way.

---

## 6. Public-repository hygiene

- `manuscript_v2.md`: no absolute paths, hostnames, credentials or GPU indices. Clean.
- `ledger_1124.md`, `make_figs_1124.py`: absolute host paths, a symlink into a personal workspace root, GPU index, and a second checkout path. Fix before commit (MF-18).
- Figure CSVs: values only. Clean, and they match the raw files that I checked (`fig_replicates.csv`, `fig_qcf20.csv`, `fig_loops.csv`, `fig_steer_dose.csv`, `fig_ladder.csv`).
- No secrets matched patterns `hf_…`, `sk-…` or `ghp_…` under `paper/`.
- Section 9 correctly says the artifacts are not public. Keep "should not be cited as verified evidence" until a hashed subset is published.

---

## 7. Prioritized experiments to run next

1. **E-1 / E-2 (cheap, highest value):** per-item sealed re-score of qcf15, qcf16, qcf17, qcf20 and qcap_on/off/rep1/rep2 with overlapping or contradictory neighbour items removed; paired bootstrap CIs for criterion-1 margins and for S everywhere. Pre-register it as a sensitivity analysis. No reselection.
2. **E-3 (cheap):** specificity controls (sign flip, random direction at matched norm, cross-fact key/value permutation) for qcf15/16 and qcap_on/rep1, on validation.
3. **E-4 (cheap):** corrected ladder (sub-unit strengths, matched-norm summed per-fact steering, oracle at all subject mentions, edit masked at context positions) plus a gate-threshold curve.
4. **E-5 (cheap):** a one-shot sealed ladder with validation-frozen strengths for qcf15/16.
5. **E-6 (cheap–moderate):** generation evaluation with containment scoring and chat template; add qcf20.
6. **E-7 (moderate):** LoRA/MEMIT/ROME on qcap facts (0.8B) and QLoRA on qcf20 (9B).
7. **E-8 (moderate):** comparator seeds and a properly configured MEMIT run with JSON output.
8. **E-9 (expensive in tokens):** at least 3 blind four-role editor replicates at 4B, one at a time.
9. **E-10 (moderate):** defect Stage 2 with recommended sampling at a 32k cap; only then decide whether looping is an edit target.
10. **Process:** for future arms, use SE-scaled preserve bounds, confirmation preserve sets sized for a 1-point MMLU/ARC loss (n ≥ 1,000), automatic dev/sealed overlap checks (including neighbour-vs-edited-subject conflicts), and per-item JSON for every sealed evaluation.

---

## Author response / change log (2026-11-24)

The manuscript was revised in place (`manuscript_v2.md`, now v2.1); the pre-review text is kept as `manuscript_v2_pre_review.md`. No GPU was used and no experiment was run. Every disputed or changed number was re-checked read-only against the raw JSON and scripts on the host (not the ledger). A new marker ◊ flags results pending a planned control; the controls are listed in the new Section 3.8, "Planned controls (pre-declared before running)", none of which has been run. `ledger_1124.md` was not edited and stays host-only.

### MUST-FIX

| Item | Disposition | Notes |
|---|---|---|
| MF-1 Title | **Done** | Reviewer's second alternative adopted: "Where LLM-Written Weight Edits Stand: Pre-Registered Comparisons with Prompting, Steering and Fitted Editors". Chosen over the first alternative because "subject-selective" is itself a validation-only, post-hoc claim. |
| MF-2 Abstract scope/provenance | **Done** | Reviewer's text used, with the gate threshold, the mean-of-40 steering definition, the grid truncation and "holds only under the tested grid◊" added. |
| MF-3 Truncated grids; ST = mean of 40 writes | **Done** | Verified in `baseline_ladder_cf.py` (grids {1,2,4,8}/{1,4,16,40}; `u = ws.mean(0)`; pairwise cos −0.023/−0.014) and `baseline_ladder.json`. Rows renamed; grids, selection rule and ST definition in Section 3.5; "best of each family is indistinguishable from no edit" stated; sub-unit grid declared in Section 3.8 (item 1) and the ladder claim marked pending (◊). |
| MF-4 Gate threshold/sampling | **Done** | Threshold (10% of median subject-write norm; 2.245 / 0.544), 100 MMLU questions ≤ 256 tokens, 4-token subject mask added to Section 3.5 and Table 1 caption. Threshold curve declared (Section 3.8, item 4), not run. |
| MF-5 Validation S definition | **Done** | Reviewer's text added to Section 3.2; S definition also in the Table 1 caption. |
| MF-6 Generation comparison pooling | **Done** | Like-for-like (qcf17 only vs MEMIT/ROME) in abstract and Section 4.3; prefix-match metric and 6–9% old-object floor disclosed; " 1623 in France." example verified in `qcf17_4b/gen_eval.json` (edit, none route). Containment score **not computed** (the samples are stored, so it is CPU-feasible; deferred to stay within a text-only revision). |
| MF-7 "First four-role freeze" | **Done** | Verified: qcap sealed evals written 11-23 14:51–16:45; qcf20 sealed eval 11-24 02:50; `qcap_08b/dev.json` has a confirmation split (430 rows). Changed to "first 9B freeze" in abstract, contribution 5 and Section 4.5 heading. |
| MF-8 Sealed ES/NS overlap | **Done** | Recomputed on CPU from `sealed.json`/`dev.json`: ES 40/40 in discovery in all five arms (qcf16 too, which the review's list omitted); NS overlaps 7/0/2/2/11 and within-sealed duplicates 3/1/1/2/4 match the review; the qcf17 "Dennis Rodman plays" conflict confirmed. New Section 3.7 table; disclosure in Section 3.2; affected tables marked ◊; re-scoring declared as a sensitivity analysis (Section 3.8, item 5). (Minor: my per-split NS hit counts for qcap are discovery 5, validation 3, confirmation 3, versus the review's 8/5/5; the union, 11, agrees. Likely a counting difference with duplicates; the manuscript reports only the union.) |
| MF-9 Causal "when, not what" | **Done** | Section 6.1 replaced with the reviewer's text; "when it fires" is now a hypothesis pending Section 3.8 controls. Verified that `add_oracle` injects only at the subject in the queried prompt, not in the context; this is now stated in Section 3.5 and Section 4.2. Section 2.3 also softened. |
| MF-10 Criterion-1 margins within noise | **Done** | Paragraph added after Table 5 and to the qcf20 verdict; harness drift (.7629/.7598, .7221/.7285, .9025/.9064) disclosed in Section 3.2; qcap_on on all_x2 (.8339 < .8364; NS −5.0 ± 1.94) verified and reported; Section 6.4 sentence replaced. The SE ≈ .02 per S is the review's delta-method approximation; no bootstrap has been computed (declared, Section 3.8 item 5). Pre-registered verdicts unchanged. |
| MF-11 Table 3 "Original" | **Done** | Verified `regress_qcf17_4b.json`: orig MMLU .7245, ARC .9121. Row now ".7245 (paired) / .7255 (comparator); ARC-C .9121" with the reviewer's footnote. |
| MF-12 Regression denominator | **Done** | Verified `regress_qcf15_4b.json` and `regress_qcf16_4b.json` are clean. Reviewer's text used, with "and failed them at 9B" added. |
| MF-13 "Near-official" | **Done** | Replaced everywhere (contribution 6, Section 4.6 ×3, Table 13 #19) with greedy thinking-mode decoding at 8,192 tokens and the model card's sampling recommendation; contribution 6 downgraded to "We confirm the model card's warning". |
| MF-14 qcap_off pooling post hoc | **Done** | Verified in `qcap_08b/PREREG.md` (two arms "identical except for the cap"; "effectively two runs of the same condition" appears only in the outcome section). Reviewer's text added to Section 4.4; "2/4 strictly" now accompanies every "2/5" (abstract, contribution 4, Table 5 caption, Figure 3 caption, summary, Table 13, Sections 6.4 and 7). The figure's own title still reads "one condition"; the caption flags this. |
| MF-15 Multiple comparisons | **Done** | Paragraph in Section 3.2, sensitivity note in Section 4.5, limitation in Section 7. Threshold check: ARC z = 0.85/0.38 = 2.24; Holm's first step at the family level implied by a 2 SE test (α ≈ .046 two-sided, /3) needs ≈ 2.4 SE, so qcf20 would pass. Verdict stays FAIL as pre-registered. |
| MF-16 Recipe provenance in abstract | **Done** | Reviewer's sentence added, plus "2 of 4 strictly". |
| MF-17 "Strongest on all-ctx" | **Done** | Reviewer's text used. |
| MF-18 Machine paths | **Done** | Manuscript re-checked: no absolute paths, hostnames, GPU indices or checkout names. `paper/figures/make_figs_1124.py` now reads the root from `WSL_DATA_ROOT` (the review suggested `WSL_EVIDENCE_ROOT`; the name was set by the author) with no default, and exits with a message if it is unset. Verified: unset → exit 1; set → all five CSVs and PNGs byte-identical to the previous outputs, and SVGs identical apart from matplotlib's random element ids. `ledger_1124.md` intentionally untouched (host-only; never commit). |

### SHOULD-FIX

| Item | Disposition |
|---|---|
| 1 Hypothesis provenance | Done ("stated in the analysis script before it was run, 2026-11-23; not pre-registered"). |
| 2 Both sides selected on validation | Done (contribution 2, Section 3.5). |
| 3 Loop definition; 51 finished loops | Done; verified 213 loops = 162 truncated + 51 finished; IFEval 80 loops, 59 truncated. |
| 4 Background † table | Partly: stated that no conclusion rests on the † values; table not moved. |
| 5 qcf20 confirmation ‡ | Done. |
| 6 LoRA selection bug | Done (Table 3 footnote). |
| 7 Gradient-audit provenance | Partly: stated that the search ran on a separate checkout and tool hashes were not recorded; test suite **not rerun** (text-only revision). |
| 8 Tool-level priors | Done (Section 3.6 and Section 8 #4). |
| 9 Patch and split hashes | Partly: new Appendix B lists the recorded `patch_sha256` and sealed-file SHA-256 per arm; development split hashes left in the host ledger. |
| 10 Predictions vs outcomes | Done (Appendix A; qcf15/16/17 and qcap; qcf20's PREREG has no explicit prediction). |
| 11 Table 13 completeness | Done (rows 22–25). |
| 12 Auditability not measured | Done (Section 6.2, abstract). |
| 13 Cut/demote Sections 4.6, 4.8 | Not done (structural; deferred to the next revision). |
| 14 Critique (A) as posed | Done (Sections 2.1, 6.1, 8 #1). |
| 15 Edit on strongest no-edit route | Done for qcap_on/rep1/rep2 (Section 4.4), qcf20 (Section 4.5), qcf15/16 (Section 4.1); qcf17 is in Table 3/13. |
| 16 Tokenizer warning | Done as a limitation (not separately verified). |

### Not resolved in this revision

- All controls in Section 3.8 (extended ladder, specificity controls, sealed ladder, gate-threshold curve, sealed re-scoring with bootstrap CIs) are declared but not run; every ◊ result stays pending.
- Containment-based generation metric not computed (MF-6).
- Restructuring to four contributions, and moving Sections 4.6/4.8 to appendices (Section 5 of the review; SF-13), not done.
- Figure 1 and Figure 3 image text still say "Best" and "one condition"; captions qualify them. Regenerating them with new labels would change the figure script's behaviour, which this revision kept identical.
