# Fine-tuning comparison design

[Project](../README.md) · [한국어](FINETUNING_COMPARISON.ko.md) · [Retained results](RESULTS.md)

**Status: proposed comparison, not executed or preregistered.** No sufficiently trained fine-tuning baseline is available in the retained results. The 12-story AdamW trainability check is excluded from method comparisons. Its validation decline cannot establish a general weakness of fine-tuning, and the effect of training-data size was not isolated.

The repository's [execution contract](INFERENCE_ONLY_CONTRACT.md) and [contributor instructions](../AGENTS.md) prohibit optimizer/LoRA training. This document does not authorize an exception, supply training code or revive archived scripts. Existing external training artifacts can be audited without training; a new training baseline requires an explicit user-authorized exception and a separate training workspace. Direct editing remains inference-only.

## Questions and comparison arms

| Question | Data policy | Interpretation |
|---|---|---|
| What is the effect of the adaptation method? | Give editing and fine-tuning the same eligible development data and labels; disclose what each actually reads | A common-data comparison, with all tuning costs recorded |
| How competitive is editing with practical fine-tuning? | Let fine-tuning use a substantial, relevant corpus and validation-selected configuration; disclose the editor's smaller data exposure | A practical comparison across different data and resource budgets |

The second comparison must not restrict fine-tuning to the editor's handful of examples. The first must also use a meaningful corpus rather than repeat the 12-story check. Report both separately; neither identifies the causal effect of data volume alone.

Use the same immutable original checkpoint, tokenizer and dtype for each arm: strongest measured unedited prompt/decoding configuration; LLM direct editing; a properly configured supervised fine-tuning baseline; and a LoRA baseline where applicable. Start each adapted arm from the original, not from a preliminary edit. Disclose training scope and any material implementation differences. A resource-limited arm cannot stand in for an unmeasured stronger baseline.

## Data and selection

For the recorded math task, use the exact original **Qwen3.5-0.8B** revision from study 006. A concrete data source is the official [GSM8K training split](https://github.com/openai/grade-school-math), which has [7,473 examples](https://huggingface.co/datasets/openai/gsm8k/blob/25fb0412dbcc2e5128322234d829fbd6ece72c7c/dataset_infos.json), supplemented by the [MATH training corpus](https://github.com/hendrycks/math) when testing that domain. These are eligible source corpora, not a promise that a fixed count is sufficient. Hold out validation before training; record actual counts after filtering, deduplication and validation separation. Do not use official test answers for adaptation. Record source revisions, licenses, input hashes, label formatting, unique examples, unique tokens and total processed tokens.

For a separate TinyStories comparison, use a substantial eligible portion of the [official training corpus](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/README.md) and separate validation/evaluation material. The dataset card states that the original models were trained on that corpus: continued training on it must not be described as learning unseen stories. Do not mix TinyStories language-modeling results with Qwen math scores to rank methods.

Choose data volume, learning-rate schedule, regularization, stopping rule, trainable scope and tuning budget using development/validation only. Report learning curves and the selected checkpoint, including rejected settings and unsuccessful runs. Increasing corpus size alone is not a guarantee of better validation performance. Before an independent evaluation, bind all selected checkpoints and evaluation configuration by hash. Exposed historical tests are descriptive replay material; a new comparison needs an evaluator-controlled, selection-unseen set. Public benchmark items are not guaranteed absent from model pretraining.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) describes generation modes and sampling behavior. Record thinking mode, prompts, parser, token limits and stopping conditions. Compare adapted and unedited models under common conditions within each mode; also report the strongest measured applicable unedited configuration. Report truncation, parsing failures and accuracy separately.

## Shared evaluation and report

| Measure | Required reporting for every arm |
|---|---|
| Target quality | Accuracy/NLL appropriate to the task, paired per-item changes and uncertainty |
| Target answer retention | Old-correct answers lost / originally correct answers; numerator, denominator and uncertainty |
| Unrelated-task preservation | Separate predeclared tasks; the same paired scoring for each model |
| Resources | Data/label preparation, tuning, editor/model inference, training, wall time, hardware, tokens and cost; energy only if measured |
| Selection | Data and checkpoint hashes, all attempts, selected configuration and freeze record |

Report quality and preservation together at predeclared operating points. The historical 5% rule is not a universal requirement; do not rewrite the old decision or choose a new threshold after seeing new tests. Tiny old-correct denominators need explicit uncertainty and adequate evaluation size. Fewer changed weights do not by themselves establish lower total cost.

## Unaddressed criticisms / 미대응 비판

This is a design, not evidence that either method wins. Dataset eligibility, final data volume, validation split, training budget, baseline configuration, preservation tasks and fresh evaluation material still need to be fixed before execution. The original checkpoints and complete historical evaluation assets are not distributed in the public evidence bundle. An external checkpoint without training provenance or compatible evaluation cannot fill these gaps. No new training or independent test was run for this document.
