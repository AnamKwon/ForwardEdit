# Cross-task study, 2026-11-08 to 11-22: where LLM-written weight edits help, and where they don't

[Results guide](RESULTS.md) · [한국어](CROSS_TASK_STUDY_1122.ko.md) · [Review criteria](REVIEW_GUIDE.md)

**Scope and omissions.** This page summarizes a later study series. It was run with a development branch of the tools: the v2 tool surface plus the `writers`, `errors` and causal-trace observations. That branch is not yet in this repository. The run workspaces, editor transcripts, sealed item lists and per-run evaluation JSON are **not bundled**, so the numbers below cannot be checked against published bytes here. Treat this page as a host summary, not as public evidence, and read it together with the evidence-backed [results guide](RESULTS.md). The statement there that no bundled study passed every adoption criterion still holds for the bundled studies.

## Why this study

There is an open criticism that the math gains reported for weight editing do not show real mathematical ability. We therefore tested many non-math tasks under one protocol, and compared every edit against the strongest unedited baseline we could measure.

## Setting

- **Editor and edited models.** Claude Opus edits Qwen3.5-0.8B through MCP; some studies also use Qwen3.5-2B, Qwen3.5-4B (BF16) and Qwen2.5-0.5B-Instruct.
- **Inference-only contract.** The GPU runs inference only: no backward pass, optimizer, least-squares/pinv solve or automatic search. An LLM-designed, one-shot multi-error update is allowed. It is built from forward-observed error vectors and key activations.
- **Protocol.**
  - A pre-registration is written before each launch.
  - Validation is blind: the editor sees aggregates only and may submit at most 2 candidates.
  - The tool enforces the freeze rule.
  - Each sealed test set is evaluated once, paired against the original model.
  - Every frozen edit gets a general-ability regression check (GSM8K, MMLU, ARC-C).
- **Comparators.**
  - The strongest no-edit baseline: direct, few-shot, brief CoT or thinking prompts.
  - A host-only LoRA trained on the same discovery rows. It is an upper-bound reference only, never an edit.

## Main results (sealed sets, Δ vs original)

| Task | Model | Original | LLM edit | General ability | Strongest no-edit / LoRA |
|---|---|---:|---|---|---|
| 2–3 digit add/sub | 0.8B | 48.5–74% | +4.8 to +6.8 (2 of 2 blind runs found the same L22 head) | no loss | brief CoT 95.5% vs edit 67.75%; the edit costs −8 to −11 under CoT |
| Thinking-loop repair (IFEval) | 0.8B | 20.8% | +44.8 | no loss | matched by thinking-budget forcing, 70.1% |
| Last letter of a word | 0.8B / 2B / Qwen2.5-0.5B | 53–72% | +4.0 to +9.75 | small ARC-C cost in 3 of 4 runs | a thinking baseline beats the edit |
| 3rd word, first letter | 0.8B / 2B | 13% / 32% | +11.0 / +10.5 (PRIMARY PASS) | borderline ARC-C / no loss | thinking 32.25% vs 24.0% (0.8B); 96.5% vs 42.5% (2B) |
| Inserting 58 irregular past tenses | 0.8B | 10.3% | +46.6 on unseen wordings (54 fixed / 0 broken) | no loss | beats 3-shot (33.6%) and stacks with it (74.1%); LoRA +84.5 |
| Inserting 120 facts (2 relations) | 0.8B | 15.4% | +55.0 | ARC-C −0.85 (PRIMARY fail) | LoRA +80.4, with costs elsewhere |
| Medical multiple choice | 0.8B | 48.6% | +3.0 / +1.9 (n.s.) | no loss | LoRA +4.4, equal off-domain (a letter-prior shift); GSM −37 |
| BBH dyck_languages | 2B | 5% | +8.0 on the official wording, −3.0 on unseen wording | GSM −9.0 (fail) | — |
| BBH salient translation | 4B | 56.4% | no freeze (a letter-preference shift) | — | CoT 65.2% |
| **CounterFact, 40 rewritten facts** | **4B** | best no-edit S .760 / .785 | **S .922 / .894** (2 of 2 runs, all 4 pre-registered criteria) | MMLU/ARC/GSM unchanged | **LoRA S .989 / .987** with facts in context |
| CounterFact, no-context design (run 3) | 4B | best no-edit S .763 | no-context S .682 (pre-registered PRIMARY 3/4, FAIL); with context .939 | unchanged | LoRA on the same no-context rows: S .898 |

Negative results are kept on purpose. Several tasks produced no freeze: unseen irregular verbs, comparatives, spelling-change past tenses and the "Have a nice day" persona. Some persona and format edits passed on their target but failed preservation. The lowercase edit raised compliance from 0 to 88.7% but cost −18.9 on other formats. A no-comma edit cost −16 on GSM8K.

## What we learned

1. **A strong baseline erases most skill-like gains.** In 6 of 6 sealed skill passes, a no-edit thinking or brief-CoT prompt beat the edit. The edit wins only when the model must give a 3-token direct answer, and it does not stack with CoT. This partly confirms the math criticism: no edit improved math *reasoning*.
2. **Knowledge edits are the robust case.** Inserting or rewriting a stored association survives the strongest baseline. At 0.8B, the irregular-past edit beats and stacks with 3-shot prompting. At 4B, the CounterFact edit beats every prompting route, including an explicit "these facts override what you knew" instruction, by .11–.16 S.
3. **Skill edits do not scale; knowledge edits do.** Every gap an edit closed at 0.8B is already closed at 4B without editing. The gaps that remain at 4B need computation, not a readout change. On those tasks the edits either shift a prior or become rules bound to one wording.
4. **The editor logs explain the black box.** Successful runs share a mechanism:
   - Arithmetic used one attention head.
   - Knowledge used about 40 nearly unused MLP neurons, keyed on the subject token. They act as gated memories that fire only for the edited subject.
   - Failed runs wrote low-rank changes at the last layers that reweight a few output tokens.
5. **A cheap pre-screen predicts transfer.** It checks whether the best residual writer pushes toward the gold answer on at least 80% of wrong rows. The prediction matched the outcome on 6 tasks post hoc and on 2 prospective tasks. It gives false positives in multiple-choice letter formats.
6. **LoRA beats the edit in a fair comparison.** In runs 1–2 the edit seemed more persistent without context (.679 / .408 vs .055 / .087), but those LoRAs were trained on rows that carried the context. In run 3, both methods used the same no-context rows. LoRA reached .898 and the edit .682, which also missed the pre-registered prompting bar of .763. The edit's remaining advantages are footprint (38 neurons in one layer), auditability and no training compute.

7. **The LLM supplies the structure, not the values.** A host-only comparator trained only the editor's 38 neurons. Starting from the editor's values, it reaches no-context S .884, close to a dense LoRA. Starting from the original values, the same neurons reach .668. So the editor's choice of neurons and gating is what makes the edit work, and its hand-set values are the bottleneck.

## Cost

| | LLM edit | LoRA |
|---|---|---|
| Time per run | 45–90 min | 10–30 min |
| GPU | inference only, 17–79 min | training |
| Main cost | 0.3–0.8M new input tokens plus 8–16M cached tokens per run | GPU training |

## Capacity limits

- 7B, 9B and 27B were not tested. The blockers were disk space and missing tooling for quantized weights.
- The 4B LoRA comparison first ran out of host RAM. It completed after the reference weights were reduced to scalar norms.

## Unaddressed criticisms / 미대응 비판

- **"Math gains are not real math ability."** Partly confirmed (finding 1).
- **"Gains vanish against a proper baseline."** Confirmed for every skill-like edit. Not confirmed for knowledge edits.
- **"Results do not scale."** Tested only up to 4B. The knowledge rewrite scaled (2/2); the skill edits did not.
- **Host-picked tasks.** Partly mitigated by a 27-task BBH screen and pre-registered gates, but the host still chose among screened tasks.
- **Same editor family and prompt.** All runs used one editor family and prompt. Most results are single runs, and several sealed sets are small.
- **No ROME/MEMIT comparison** for the knowledge edits.
- **Artifacts not bundled** (see Scope and omissions).
