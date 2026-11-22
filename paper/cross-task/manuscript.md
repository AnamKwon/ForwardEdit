# Beyond Loop Repair: When Do LLM-Written Weight Edits Help? A Pre-Registered Cross-Task Study

[한국어](manuscript.ko.md) · [Draft index](README.md) · [Study summary](../../docs/CROSS_TASK_STUDY_1122.md)

**Status.** Working draft, 2026-11-23. The numbers come from host evaluation files of the 2026-11-08 to 11-23 series. **Those run artifacts are not bundled in this repository**; see *Artifact availability*.

## Abstract

Weight-Space Programming (WSP) showed that an LLM can read and directly write the weights of another network. The published demonstrations focus on mathematics, and its best-known gains come from repairing thinking loops. We ask a broader question: **on which kinds of task does an LLM-written, inference-only weight edit produce a real improvement, and why?**

An LLM editor (Claude Opus) edited Qwen3.5 models (0.8B, 2B, 4B) and Qwen2.5-0.5B-Instruct. The GPU was used for inference only: no backward pass, optimizer, solver or automatic search. We tested more than 15 tasks across five categories: arithmetic, letter and word skills, knowledge insertion, knowledge rewriting (CounterFact), format and persona control, and official BIG-Bench Hard tasks. Each test was pre-registered, evaluated once on a sealed set, and followed by a general-ability regression check (GSM8K, MMLU, ARC-C). Every edit was compared with the strongest unedited baseline we could measure, and with a host-only LoRA trained on the same data.

Four findings emerge:

1. **Skill-like edits pass sealed tests but lose to prompting.** In 6 of 6 sealed passes, a thinking or brief chain-of-thought (CoT) prompt beats the edit. Arithmetic edits also hurt step-by-step answers (−8 to −11 points).
2. **Knowledge edits are the robust case.** At 0.8B, inserted past tenses beat 3-shot prompting and stack with it (74.1%). At 4B, CounterFact rewrites passed all pre-registered criteria in 2 of 2 runs (S .922 / .894). They beat every prompting route, including an explicit "these facts override your knowledge" instruction, with no general-ability change.
3. **The kind of gap predicts success: a stored association or a direct readout is editable; a missing computation is not.** This explains why skill edits do not scale from 0.8B to 4B while knowledge rewrites do. A cheap forward-only "writer-credit" prescreen predicts transfer before any editor run. It matched the outcome on 6 tasks post hoc and was correct on several prospective tasks. Multiple-choice letter formats are a known failure mode.
4. **The editor logs make the black box concrete.** Successful edits reuse one shared component: an attention head for arithmetic, about 40 near-dead MLP neurons as subject-keyed gated memories for facts. Failed edits are low-rank writes at the last layers that only reweight output tokens.

A LoRA trained on the same rows inserts knowledge more strongly. A third, pre-registered 4B run made no-context persistence the PRIMARY. The edit raised no-context S from .092 to .682 but missed the bar set by prompting (.763), and a LoRA trained on the same no-context rows reached .898. **The edit's remaining advantages are footprint (38 neurons in one layer), auditability and no training compute, not effect size.** A same-footprint comparator locates the shortfall. Training only the editor's 38 neurons from its values reaches .884; the same neurons trained from their original values reach .668. **The LLM supplies a structure that gradient training cannot rediscover in that budget, while its hand-set values realize about 70% of it.**

## 1. Introduction

WSP reframed model editing: instead of training, an LLM inspects weights and activations and writes parameter values itself [Takehi et al., 2026]. Follow-up reports gave large gains on small models, mostly on mathematics and instruction following (WSP-Scope, Hu et al.). Public critics asked whether these gains reflect new mathematical ability or the removal of a pathology, such as repetition loops under thinking mode.

We checked this directly on WSP-Scope's own IFEval test split. **Turning thinking off, with no edit, recovers about 95% of the reported +36.7-point gain** (56.1% vs 58.1%). On MATH-500 it recovers about 92% (35.4% vs 37.8%). This supports the criticism: loop repair is real, but it is not a new capability.

This raises the question this paper studies. **If loop repair is the main thing WSP-style edits deliver on math, is there anything else they can do?** We move beyond math and ask, task by task, whether an edit beats the strongest unedited configuration, and what distinguishes the tasks where it does.

**Contributions.**

1. **A cross-task map** (Section 4). It covers 15+ tasks in five categories, each with a sealed result, a regression check, the strongest no-edit baseline and, where available, a matched LoRA reference. Negative results are kept.
2. **A gap-type hypothesis** (Section 5.1), with a forward-only prescreen that operationalizes it and was tested prospectively.
3. **Baseline erasure** (Section 5.2). A sealed PASS is not an improvement until it beats prompting, and skill edits fail this test.
4. **Knowledge rewriting as the robust niche** (Section 5.3). It scales to 4B and beats strong prompt baselines, with a candid LoRA comparison.
5. **Mechanisms read from editor logs** (Section 5.4), explaining both successes and failures.

## 2. Related work

- **Weight-Space Programming.** WSP [Takehi et al., 2026] introduced LLMs as native readers and writers of weights. WSP-Scope (Hu et al., *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations*) reported held-out gains on seven benchmarks for Qwen3.5-0.8B/9B. Its gains were measured with thinking enabled, and its appendix attributes much of the effect to fewer truncations. A POSTECH study (arXiv 2611.22055) has an LLM build lookup-table models from scratch. **We differ from all three.** We edit a pretrained model to change one behavior while preserving others, derive edits from the model's own forward observations, and require each edit to beat the strongest unedited baseline.
- **Direct knowledge editing.** ROME and MEMIT locate factual associations in mid-layer MLPs and write them with closed-form updates [Meng et al., 2022; 2023]. Our knowledge edits likewise end up in mid-layer MLPs. The values, however, are designed by the LLM without a solver, and the edits repurpose near-dead neurons rather than applying a rank-one update. **We have not run ROME or MEMIT as baselines** (Section 7).
- **Fine-tuning.** LoRA [Hu et al., 2021] is our reference for what gradient training achieves on the same rows. It is a host-only comparator, never an edit.
- **Benchmarks.** CounterFact [Meng et al., 2022], BIG-Bench Hard [Suzgun et al., 2022], IFEval [Zhou et al., 2023], GSM8K [Cobbe et al., 2021], MMLU [Hendrycks et al., 2021], ARC [Clark et al., 2018].

## 3. Method

### 3.1 Editing contract

The editor accesses the model only through an MCP server.

- **Reading:** weights and structure, stored activations, attribution of residual writers to the output (`writers`), output-embedding error vectors (`errors`) and causal tracing.
- **Writing:** explicit edit operations (scale, project_out, replace_block, low_rank).
- **Budget:** 20–24 forward passes per study, plus a soft tool-call budget.
- **Prohibited:** the GPU runs inference only. No backward pass, autograd, optimizer, least-squares or pinv solve, iterative fitting or random search. A test tripwire enforces this.
- **Allowed exception:** an LLM-designed, one-shot multi-error update built from observed error vectors and key activations. The LLM chooses the layers, examples and coefficients.

### 3.2 Evaluation protocol

- **Data roles.** Discovery rows build the edit. A validation split confirms it blindly: aggregates only, at most 2 candidates. The tool enforces a freeze rule: the pooled gain must exceed 2 SE, with no significant preserve loss and at least 100 GSM8K preserve items measured.
- **Sealed test.** A pre-registered sealed set is evaluated once on the frozen candidate, paired against the original.
- **Regression.** Every frozen edit is checked on MMLU (2,000), ARC-C and GSM8K (200).
- **Baselines.** The strongest unedited baseline across direct, few-shot, brief CoT, thinking and, for CounterFact, context routes. The best of these is the bar for any improvement claim.
- **LoRA reference.** A host-only LoRA trained on the same discovery rows and selected on validation, where feasible.

### 3.3 Writer-credit prescreen

Before an editor run, the host runs one forward pass over wrong development rows. For each residual writer (attention or MLP output at each layer), it computes the share of wrong rows on which that writer pushes the logit toward the gold answer.

- **Gate:** best share ≥ .80, with at least 2 writers ≥ .70, computed on the failure mode the evaluation wordings exhibit.
- **Cost:** about one minute of inference, against 1.5–2 hours of editor tokens.

## 4. Results: the cross-task map

Δ is the sealed paired change versus the original model, in points (± SE where reported).

| Category | Task (model) | Original | LLM edit (sealed) | General ability | Strongest no-edit / LoRA |
|---|---|---:|---|---|---|
| Arithmetic | 2–3 digit add/sub (0.8B) | 48.5–74% | +4.8 to +6.8; 2 of 2 blind runs found L22 head 12 | no loss | brief CoT 95.5% vs edit 67.75%; edit −8.25 / −11.0 under CoT; LoRA +37.5 with GSM −32 |
| Fault repair | Thinking loops, IFEval (0.8B) | 20.8% | +44.8 | no loss | thinking-budget forcing 70.1% |
| Letter skills | Last letter (0.8B, 2B, Qwen2.5-0.5B) | 6.75–72% | +4.0 to +9.75 | small ARC-C cost in 3 of 4 runs at 0.8B | thinking beats the edit |
| | Letter before the first letter (0.8B) | 7–11% | +14.5 to +24.25 | clean after a neighbour fix | thinking/CoT beat the edit; stacking with CoT −29.75 |
| | Third letter (0.8B) | 11.75% | +12.25 ± 2.28 | ARC-C −2.3 ± 0.92 (fail) | — |
| Word routing | 2nd word, first letter (0.8B) | 40.5% | +26.75 | no loss | thinking beats the edit |
| | 3rd word, first letter (0.8B / 2B) | 13% / 32% | +11.0 ± 2.11 / +10.5 ± 2.26 (PRIMARY PASS) | borderline / no loss | thinking 32.25% vs 24.0%; 2B 96.5% vs 42.5% |
| Knowledge insertion | 58 irregular past tenses (0.8B) | 10.3% | +46.6, unseen wordings (54/0) | no loss | 3-shot 33.6%; edit + 3-shot 74.1%; LoRA +84.5 |
| | 120 facts, 2 relations (0.8B) | 15.4% | +55.0 (132/0) | ARC-C −0.85 (fail) | LoRA +80.4 (capital −6.5, GSM −5) |
| Knowledge rewriting | CounterFact, 40 facts (4B), run 1 | best no-edit S .760 | S .922; PS +17.5 ± 4.3, NS +27.0 ± 3.3 | MMLU/ARC/GSM n.s. | override prompt below; LoRA S .989 |
| | CounterFact, 40 new facts (4B), run 2 | best no-edit S .785 | S .894 | n.s. | LoRA S .987 |
| | CounterFact, no-context design (4B), run 3 | best no-edit S .763 | no-context S .682 (PRIMARY 3/4, FAIL); all-ctx .939 | n.s. | LoRA no-context S .898 |
| Format / persona | lowercase (0.8B) | 0% | 88.7% (133/0) | other formats −18.9 (fail) | — |
| | no comma (0.8B) | 83% | +90.7 (r2) / +56.7 (r3) | other formats −10.0 / GSM −16.0 (fail) | — |
| Domain | Medical MC (0.8B) | 48.6% | +3.0 / +1.9 (n.s.) | no loss | LoRA +4.4, equal off-domain (letter prior); GSM −37 |
| BBH (official) | dyck_languages (2B) | 5% | +8.0 ± 3.37 official wording; −3.0 unseen wording | GSM −9.0 (fail) | CoT 7.6% |
| | salient translation (4B) | 56.4% | no freeze (letter-preference shift) | — | CoT 65.2% |
| No freeze | unseen irregular verbs, comparatives, spelling-change past, "Have a nice day" persona | — | — | — | — |

## 5. Hypotheses and evidence

### 5.1 H1: the kind of gap decides editability

**H1.** An inference-only edit closes a gap when the needed information is already linearly present in some residual writer, so the edit only has to re-route or amplify it, or when a stored association is missing and can be written into a dedicated slot. It cannot close a gap that needs a missing computation.

**Evidence.**

- **The prescreen measures "already present".**
  - Transfer tasks: best writer share .82–.86, with at least 2 writers ≥ .70.
  - Failed tasks (unseen irregular verbs, comparatives, spelling past): .62–.66, with no writer ≥ .70.
  - Prospective predictions: qadv (.729, predicted FAIL) and qprev (.885, predicted PASS; sealed +24.25) were both correct, as were the refined gate on third letter and the 3rd-word task on 2B.
  - Failures of the gate: a digit-answer subset that existed in only one wording (qlonger), and multiple-choice letter formats, where letter-decision writers dominate (Qwen2.5 addition, BBH salient). The gate predicts transfer, not safety.
- **Causal tracing locates the comparatives failure.** The adjective reaches the answer position at L14–18, so the failure lies in answer formation, not in binding.
- **Scale behaves as H1 predicts.**
  - The irregular-past gap (0.8B 11.2% direct) is 80.2% direct at 4B: the knowledge is simply present.
  - At 4B, 3-shot CoT reaches ≥ .95 on 12 of 27 BBH tasks.
  - The tasks left (dyck .516, word_sorting .632) fail the writer gate: they need computation.
  - At 4B, the only gap an edit closed was a stored association: CounterFact.

### 5.2 H2: baseline erasure, and skill edits trade CoT for direct answers

**H2.** A skill edit amplifies a direct readout. Its value disappears once the model may reason, and it can damage the reasoned answer.

**Evidence.**

- In 6 of 6 sealed skill passes, thinking or brief CoT beats the edit. The edit wins only at a 3-token direct answer.
- The edit often stacks with 3-shot prompting but not with CoT: letter-before drops −29.75 when the two are combined.
- Arithmetic edits lose 8.25 and 11.0 points under brief CoT, which reaches 95.5% without any edit.
- The same pattern explains WSP-Scope's IFEval and MATH gains: the edit repairs a pathology that a decoding change also removes.

### 5.3 H3: knowledge rewriting is the robust niche, and it scales

**H3.** Writing a stored association into dedicated, otherwise idle MLP slots yields a local change that reasoning cannot replace, because the model must look up such a fact; it cannot derive it. It also adds to prompting with the facts. Run 3 refines this hypothesis: alone, the change is weaker than prompting and weaker than a matched LoRA.

**Evidence.**

- **0.8B:** the irregular-past edit (58.6%) beats 3-shot (33.6%) and thinking (2.6%), and stacks with 3-shot (74.1%).
- **4B CounterFact, 2 of 2 runs:** all four pre-registered criteria pass.
  - With all 40 facts in the prompt, the unedited 4B model confuses neighbouring subjects (NS .54–.58). The edit's memories fire only on the edited subject, which restores NS to about .85.
  - An explicit override instruction is weaker than plain context (S .715).
  - MMLU, ARC and GSM are unchanged.
- **Persistence without context (run 3, qcf17_4b, pre-registered PRIMARY).** On 40 new facts with no-context development rows, the frozen edit (38 L8 neurons) reached no-context S .682 (PS +51.25 ± 5.59, 41/0; NS −1.5 ± 0.86; regression clean). That is far above the original (.092) but **below the pre-registered bar of .763** set by the best prompting route. PRIMARY: **3/4, FAIL**. Designing for this route did not move it beyond run 1's incidental .679. The same edit stacks well with context (all-ctx S .939).

**Comparison with LoRA (an honest boundary).**

- Trained on the same rows, LoRA inserts more: past tense +84.5 vs +46.6; CounterFact with facts in context S .989 / .987 vs .922 / .894.
- Runs 1–2 suggested a persistence advantage: LoRA trained on context-bearing rows scored .055 / .087 without context, against the edit's .679 / .408. **Run 3 shows that this was a comparison artifact.** A LoRA trained on the same no-context rows reached no-context S **.898** (ES 1.00, PS .79, NS .935; MMLU −0.25 ± 0.47, ARC 0.0), against the edit's .682.
- LoRA also carries off-target costs in several tasks: GSM −32 and −37; capital −6.5.
- The edit is 40 neurons with no measurable general-ability change.
- **The edit is not stronger than fine-tuning, and not more persistent.** It is far more compact (38 neurons in one layer vs a dense delta on 96 matrices), fully auditable, and needs no training compute. It is also slower: about 3 h of editor time against about 50 min for LoRA.

### 5.4 H4: editor logs reveal two distinct edit types

**Successful edits reuse one shared component:**

- Arithmetic: L22 attention head 12, found independently in 2 of 2 blind runs.
- Word routing: an attention-routing correction.
- Knowledge: about 40 near-dead L6 (0.8B) or L8 (4B) MLP neurons, keyed on the subject token's MLP input. Each gate opens only for its own subject. Each write combines the embedding difference E[new] − E[true] with the model's own hidden state of the new object.

**Failed edits share a different signature:** low-rank writes at the last layers (L23/L28) that reweight a handful of output tokens.

- When the answer is not linearly present upstream, such a write becomes a prior shift (BBH salient: letter D → E/C) or a rule bound to one wording (BBH dyck: +8 on the official wording, −3 on unseen wording).

**This yields a practical, log-level diagnostic:** an edit concentrated in last-layer output reweighting predicts a non-transferring or prior-shift result.

### 5.5 H5: the LLM supplies the structure, not the values

**H5.** The editor's main contribution is the edit's *structure*: which neurons it uses, which keys open their gates, and which subject each memory serves. Its hand-designed write values are the weaker part.

**Evidence (qcf17_4b, a host-only same-footprint comparator, pre-registered with its interpretation, sealed once).** Only the editor's 38 L8 neurons were trained by gradient (gate/up rows and down columns, ≈ 0.29M values), on the same no-context rows:

| no-context sealed S | ES | PS | NS | S |
|---|---|---|---|---|
| LLM edit | .625 | .600 | .885 | .682 |
| 38 neurons trained from the **editor's** values | 1.00 | .775 | .905 | **.884** |
| the same 38 neurons trained from the **original** values | .975 | .425 | .90 | .668 |
| dense LoRA on 96 matrices | 1.00 | .7875 | .935 | .898 |

- **The editor's subspace is sufficient.** Trained from the editor's values, 38 neurons nearly match a dense LoRA, with no MMLU or ARC change.
- **Training cannot rediscover that structure in the same budget.** Started from the original values, the same neurons reach only the edit's level, and paraphrase success falls to .425.
- **Interpretation.** The LLM finds a good place and gating for each memory, and its write values realize about 70% of what that structure allows. This suggests a concrete research direction within the inference-only contract: better value design, for example per-fact write vectors derived from observed output errors. It is also an argument that LLM-written edits and gradient training are complementary rather than competing.

## 6. Process lessons (tools that changed outcomes)

- **The `writers` observation.** Blind runs failed to find the arithmetic mechanism until linear-attention mixers became observable; afterwards, 2 of 2 found it.
- **Larger general preserve sets** (1,000-row multiple-choice choice pass, ≥ 100 GSM items) exposed costs that small checks hid. A dyck edit lost 9 GSM points after passing a 20-item check.
- **A tool-enforced freeze rule** refused borderline candidates (z 1.81).
- **Multi-wording neighbour preserves** removed a near-neighbour leak (−16.7 → 0).
- **A host-provided answer-start activation bank** let editors check every new gate row against unrelated prompts without spending forward passes. It removed the multiple-choice damage seen in earlier sessions.

## 7. Limitations and unaddressed criticisms / 미대응 비판

- **No ROME/MEMIT comparison.** This is the most important missing baseline for H3.
- **One editor family and one prompt design.** Most results are single runs, and several sealed sets are small (40–80 cases).
- **Models up to 4B only.** 7B, 9B and 27B were blocked by disk capacity and quantized-weight tooling.
- **Host-selected tasks.** A 27-task BBH screen and pre-registered gates mitigate this; they do not remove it.
- **The editor prompt carried earlier-session recipe notes** (e.g., the layer mapping for knowledge edits). The 4B knowledge result is recipe transfer, not blind discovery.
- **Costs.** An editor run takes 45–90 minutes and 0.3–0.8M new input tokens, against 10–30 minutes for LoRA. No efficiency advantage is claimed.
- **"Math gains are not real math ability."** Confirmed for our arithmetic edits (H2).
- **Artifacts are not public** (see below), so readers cannot yet verify the numbers.

## 8. Conclusion

Beyond loop repair, LLM-written inference-only edits have a narrow but real niche: **writing stored associations**. There they add to strong prompting (S .89–.94 with facts in context, vs .75–.79 for prompting alone), scale to 4B, leave general ability intact, and remain inspectable down to individual neurons. Alone, without context, they do not yet match prompting, and a matched LoRA beats them. Skill edits pass sealed tests but are erased by reasoning baselines. The gap-type view (H1) and a one-minute prescreen predict in advance which case applies.

## Artifact availability

- Run workspaces, editor transcripts, sealed item lists and evaluation JSON are retained locally but not yet published. Publishing them requires redaction review.
- This draft should not be cited as verified evidence until a curated, hash-bound subset is released, as was done for the earlier [case studies](../../experiments/README.md).

## References

- Takehi, R., Teoh, J., Zhang, A., Khattab, O. (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* Research blog. https://rikiyat.github.io/blog/weight-space-programming/
- Hu et al. (2026). *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations.* Shanghai AI Lab.
- POSTECH (2026). LLM-written lookup-table models. arXiv 2611.22055.
- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS. https://arxiv.org/abs/2202.05262
- Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., Bau, D. (2023). Mass-Editing Memory in a Transformer. ICLR. https://arxiv.org/abs/2210.07229
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Suzgun, M. et al. (2022). Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. https://arxiv.org/abs/2210.09261
- Zhou, J. et al. (2023). Instruction-Following Evaluation for Large Language Models. https://arxiv.org/abs/2311.07911
- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems. https://arxiv.org/abs/2110.14168
- Hendrycks, D. et al. (2021). Measuring Massive Multitask Language Understanding. https://arxiv.org/abs/2009.03300
- Clark, P. et al. (2018). Think you have Solved Question Answering? Try ARC. https://arxiv.org/abs/1803.05457
