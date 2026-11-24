# Related Work: revision draft (2026-11-24)

New BibTeX entries are in [references_add_1124.bib](references_add_1124.bib). `references.bib` itself is unchanged.

## Citation check and corrections to the current draft

**How it was checked.** Every arXiv id was checked against its arXiv abstract page (title, full author list, date). Venues were checked against the arXiv comment or journal-ref field, the ACL Anthology BibTeX, or PMLR. Blogs and forum threads were opened directly.

**Result.**
- All 9 entries in `references.bib` resolve and match their sources.
- All 11 works in the manuscript's reference list exist, but 3 have errors.
- All 28 new entries are verified, with two partial exceptions:
  - The NeurIPS 2024 venue for Arditi et al. is not stated on arXiv.
  - ThinkEdit's EMNLP 2025 track (main or findings) is not stated.

**Corrections and caveats.**

1. **Section 2 and Section 7 are now out of date on ROME/MEMIT.** Both say ROME and MEMIT were not run. A host-only comparator has since been run on the qcf17 no-context setting:

   | Method | Sealed S | Notes |
   |---|---|---|
   | LLM edit | .682 | |
   | ROME | .806 | MMLU −2.4 |
   | MEMIT | .897 | MMLU −0.4 |
   | LoRA | .898 | |

   Two disclosures must go with these numbers: (a) ROME and MEMIT are our own reimplementations using GPT-J default hyperparameters; (b) the generation-based comparison was run on the validation split only. Section 7 should keep the ROME/MEMIT item, reworded to cover these two limitations.
2. **The POSTECH paper is cited without title or authors, and the "lookup-table models" description is not supported by its abstract.**
   - Correct citation: Ok and Lee, *Weight-Space Programming Doesn't Need Transformers*, arXiv:2611.22055 (POSTECH).
   - What it does: frontier LLMs, Claude Opus 5.5 among them, generate weights and design architectures from scratch, and the paper proposes an associative-memory model.
   - Suggested description: "LLM-written models from scratch."
3. **WSP-Scope is cited as "Hu et al., Shanghai AI Lab" with no identifier.**
   - Correct citation: Hu, Tang, Li, Zhu, Feng, Zhang, Chen and Hu, arXiv:2611.09382. The title page lists three affiliations: Shanghai Artificial Intelligence Laboratory, Tongji University and Fudan University.
   - The draft says "its appendix attributes much of the effect to fewer truncations." This was not re-verified against the PDF in this pass. Either cite the specific table, or reword to our own observation (the base model hits the cap on most dev samples).
4. **Takehi et al. (2026).** The authors are MIT CSAIL and the editing model is GPT-6 Astra. The Qwen loop-repair experiment used code execution and forward runs, not raw weights alone. This is Honnibal's main criticism, and it applies equally to our MCP setup. The blog title "natively read & write" should not be repeated as our own description.
5. **Manuscript reference list.**
   - ARC title is truncated. It should read "... Try ARC, the AI2 Reasoning Challenge."
   - MEMIT's second author is written "Sen Sharma, A." Use arXiv's form, "Sharma, Arnab Sen", which `references.bib` already uses.
   - LoRA is ICLR 2022.
   - MMLU is ICLR 2021.
   - MATH (`hendrycks2021math`) is NeurIPS 2021, so it can become `@inproceedings`.
6. **Missing from `references.bib` although cited in the manuscript:** LoRA, BBH, IFEval, MMLU, ARC, WSP-Scope and POSTECH. All are added to the new bib.
7. **Model scale.** The current draft covers 0.8B to 4B. This section says "up to 4B (9B in progress)." Change it only if the qcf20_9b sealed result is reported in the manuscript.
8. **Not cited, because unverifiable here:** a Hugging Face re-evaluation reported in community summaries (thinking-off versus edit). No primary source was found.

---

## 2. Related work

### 2.1 LLM-driven weight editing: Weight-Space Programming and its critiques

**WSP and its follow-ups.**
- **WSP [@takehi2026wsp].** A frontier LLM inspects a network and writes its parameters directly, with no optimizer. The best-known language-model result is a loop repair on Qwen3.5-0.8B under thinking mode: a direction subtracted from 64 `down_proj` columns in each of two layers raised MATH-500 from 7.9% to 48.1%.
- **WSP-Scope [@hu2026wspscope].** Ran 16 editing trajectories on Qwen3.5-0.8B and 9B. It reported large gains from very localized edits (29.45 points from 1,023 parameters), uneven transfer, and 6 of 16 trajectories that never validated an improvement.
- **Ok and Lee [@ok2026wspnotransformer].** Have LLMs write models from scratch rather than edit pretrained ones.

**The critiques.**
- **Missing baselines.** Public critiques argue that the headline gains reflect removal of a decoding pathology plus missing baselines [@hn2026wspthread; @honnibal2026misleading].
- **Thinking-off and steering baselines.** Wang et al. [@wang2026closerlook] show that turning thinking off, with no edit, recovers most of the gain: 53.3% versus 57.7% for the edit and 12.3% for the base model with thinking on. Barry [@barry2026lp] shows two things:
  - A sparse rank-1 edit found by a linear program matches the LLM's edit.
  - Its mean push, applied as a 1x steering vector, recovers essentially the whole fix.
- **Weight-reading claims.** Ahmed [@ahmed2026wspaudit] audits weight-reading claims with label-permutation controls and calls for pre-registration.

**How we differ.**
- We reproduce the thinking-off result on WSP-Scope's own splits.
- We then ask which tasks remain editable once the strongest unedited configuration is the baseline.
- Our editor works only through forward observations and explicit CPU arithmetic. No solver, gradient or strength sweep is used.
- Every claim goes through pre-registered gates on a sealed split, plus regression checks.

We do not claim that the LLM understands the model better than a solver. Barry's result and our own comparators below suggest it does not.

### 2.2 Knowledge editing and its evaluation

**Methods.**
- **Locate-then-edit.** ROME [@meng2022rome] and MEMIT [@meng2023memit] find factual associations in mid-layer MLPs and write closed-form updates.
- **Hypernetworks.** MEND [@mitchell2022mend] trains a hypernetwork to produce edits.
- **Null-space projection.** AlphaEdit [@fang2025alphaedit] projects MEMIT updates onto the null space of preserved knowledge.

**Evaluation critique.** Yang et al. [@yang2025mirage] show that teacher-forced, truncated evaluation greatly overstates editing success compared with autoregressive generation.

**How we compare.**
- **Where our edits sit.** Our successful knowledge edits also sit in mid-layer MLPs. They are written differently: the LLM allocates subject-gated, near-dead neurons and sets their values by hand, without a covariance solve.
- **Effect size.** We are candid that this is weaker. On no-context CounterFact insertion at 4B, our edit reaches S = .682 against .897 for a reimplemented MEMIT and .898 for a LoRA on the same rows (Section 5.3). Under a generation metric of the kind Yang et al. advocate (validation split only), the gap is wider.
- **What we claim instead.** Our claim for knowledge edits is not effect size. It is footprint (tens of neurons in one layer), auditability, and the finding that the LLM-chosen structure is useful: training only those neurons nearly matches LoRA.

### 2.3 Activation steering, directional ablation and logit bias

**Steering methods.** Steering adds a fixed direction to the residual stream at inference:
- ActAdd [@turner2023actadd]
- RepE [@zou2023repe]
- CAA [@rimsky2024caa]

**Steering written into weights.**
- Arditi et al. [@arditi2024refusal] write the same kind of direction into weights by orthogonalizing every residual writer.
- Steer2Edit [@sun2026steer2edit] converts steering vectors into rank-1 component edits.
- Xu et al. [@xu2026whysteering] formalize steering as a bias-only parameter update.
- Adila et al. [@adila2026activationshifts] relate fine-tuning to activation shifts.

**Logit bias.** Decode-time logit bias is the simplest output-side baseline. NoWait [@wang2025nowait] is an instance applied to reasoning.

**How we differ.** These results imply that any always-on rank-1 write is close to a steering vector, and the WSP loop fix appears to be one [@barry2026lp]. Our edits are designed to be gated: query-gated rank-1 updates, and neurons keyed to specific subjects. The fraction of tokens on which an edit is active is the measurable difference from steering. We have not yet run steering or logit-bias baselines on the skill and format tasks, so for those tasks steering equivalence remains an unaddressed criticism.

### 2.4 Parameter-efficient fine-tuning

LoRA [@hu2022lora] is our reference for what gradient training achieves on the same rows. It is run by the host only and is never offered to the editor.

**How we differ.** LoRA beats our edits on effect size wherever we compared them. Our setting differs in compute: GPU use is inference only, with no backward pass or optimizer state. This matters mainly when training memory is the binding constraint. We do not claim an efficiency advantage in wall-clock time or tokens: an editor run takes 45 to 90 minutes, against 10 to 30 minutes for LoRA.

### 2.5 Reasoning-model looping and token efficiency

**Overthinking, looping and length control.**
- Small and distilled reasoning models overthink [@chen2024overthinking] and fall into repetition loops, more often at small scale and low temperature [@pipis2026loops].
- The Qwen3.5-0.8B model card itself warns about thinking loops [@qwen35modelcard].
- Training-free remedies include:
  - budget forcing [@muennighoff2025s1]
  - suppressing reflection tokens [@wang2025nowait]
  - steering away from reflection states [@chen2025seal]
  - projecting out a short-reasoning direction from attention heads [@sun2025thinkedit]

**How we use this.** This literature supplies the baselines that any loop-repair claim must beat: thinking off, budget forcing, logit suppression and steering. In our study these baselines account for most of the published WSP math gains, so we treat loop repair as the removal of a known defect rather than as new capability. Our contribution is not a better loop fix. It has three parts:
- a protocol that separates defect removal from improvement;
- gated sparse edits that beat strong prompting in one niche (stored associations, with facts also in context);
- defect discovery: localizing failure mechanisms from forward observations under inference-only compute.

**Unaddressed criticisms / 미대응 비판 (related to this section).**
- No steering or logit-bias comparators on the skill and format tasks.
- No label-permutation control of the kind Ahmed proposes.
- ROME and MEMIT are reimplementations, compared on validation data only.
- Most editor arms are single runs.
