# ForwardEdit workflow explainer

English · [한국어](README.ko.md)

An **English-first public explainer**, with a matching Korean edition: 136 seconds, 1920×1080, 30 fps. It explains the LLM/MCP workflow, the recorded Qwen 006 gain and failed adoption gate, and the need to compare with fine-tuning on substantial task data. Both editions use on-screen captions; recorded narration and music are not included.

## Preview and edit

```bash
cd videos/forwardedit-explainer
npm ci
npm run dev -- --port=3333
```

Select **ForwardEdit-English** for the primary public edition or **ForwardEdit-Korean** for Korean. Each edition also has an individual scene folder. Text, images and scenes have authored Studio editing targets. Matrix cells, measured-data groups and the progress indicator are computed from code/data. Result values are bound to `src/recorded-evidence.json` rather than independently editable as claims.

To export a requested MP4, use Studio's **Render** button, or run:

```bash
npx remotion render ForwardEdit-English out/forwardedit-en-comparison.mp4
npx remotion render ForwardEdit-Korean out/forwardedit-ko-comparison.mp4
```

Earlier `*-results.mp4` and `*-context.mp4` exports contain the superseded comparison scene. This revision uses `*-comparison.mp4`.

Follow the [Remotion license](https://www.remotion.dev/docs/license) for your usage. Bundled font notices are in `licenses/`.

## Story

| Time        | Scene                                                                         |
| ----------- | ----------------------------------------------------------------------------- |
| 00:00–00:10 | Introduction                                                                  |
| 00:10–00:24 | External LLM, MCP tools and target model                                      |
| 00:24–00:40 | Contract, API, actual modules, weights and forward observations               |
| 00:40–00:54 | Numerical derivation, alternatives and falsification                          |
| 00:54–01:10 | Explicit coordinates and values, checks and CPU candidate write               |
| 01:10–01:22 | Development feedback, candidate lineage, revision or restoration              |
| 01:22–01:34 | Freeze selection before independent confirmation                              |
| 01:34–01:50 | Retained Qwen 006 accuracy gain and failed preservation gate                  |
| 01:50–02:06 | Fine-tuning comparison requirements and missing evidence              |
| 02:06–02:16 | Write correctness, behavioral effect, independent improvement and limitations |

## What the video represents

- The matrix, `0.120 → 0.110`, input `+0.50` and API screens are **illustrative values and abbreviated mock calls**, not recorded model observations or a new experiment. No model was executed or edited for this video.
- The Qwen scene uses **retained historical measurements**, separately from mock examples. Its metric snapshot comes from `scripts/build_readme_figures.py`. The fine-tuning scene shows **comparison requirements that remain unmeasured**. The 12-story AdamW check is excluded from method comparisons and preserved as an [archival record](../../docs/RESULTS.md#fine-tuning-reference-comparison). The [comparison design](../../docs/FINETUNING_COMPARISON.md) requires substantial task data, validation selection and common evaluation. No archived code was executed.
- Connection helpers are drawn from `standalone.py`; basic FP32 observation/edit examples are drawn from `server.py` and `docs/USAGE.md`. Available tools and evaluation paths differ by runtime. This is a conceptual overview, not a complete executable call transcript.
- Fixed-input local arithmetic does not establish a model-wide behavioral prediction. `ready_for_causal_probe` denotes preparation for a bounded intervention, not a causal or performance certificate.
- The workflow uses explicit CPU arithmetic and inference-only model execution. It does not use backward, gradients, optimizers, automatic loss fitting, coordinate/strength sweeps or GPU training.
- The video does not claim general model improvement or superiority over WSP. It retains the README's limitation that every published study failed at least one declared acceptance criterion.
- The two AI-generated illustrations are conceptual art, not a depiction of actual layer counts, weight arrangement or a complete computation graph.

Sources: the repository's [README](../../README.md), [Korean README](../../README.ko.md), [results guide](../../docs/RESULTS.md), [architecture](../../DESIGN.md), [API guide](../../docs/USAGE.md), [execution contract](../../docs/INFERENCE_ONLY_CONTRACT.md), [review guide](../../docs/REVIEW_GUIDE.md), actual tool definitions and the metric snapshot's source reports. Production storyboards record source hashes.

## Interpreting the result scene

Qwen 006 gained 85 net correct answers (86 rescued, one lost) and retained 17/18 originally correct answers. Its frozen 5% loss rule was study-specific: with a denominator of 18 it effectively allowed zero losses. Missing that rule does not negate the observed accuracy gain; the original protocol rejection remains unchanged.

Fine-tuning can also forget prior capabilities. Both methods need the same target-quality and original-correct retention evaluation, plus separate unrelated-task checks. The archival token-accuracy table has been removed from the video. Quality, retention and cost comparisons against fine-tuning on substantial data remain unmeasured. [Evidence and research references](../../docs/RESULTS.md#interpreting-adoption-and-forgetting).

## Assets and production files

- `public/generated/`: copies of the actual outputs of Codex's built-in `imagegen`; the inspection cutout retains its alpha channel.
- `production/asset-log.json`: final generation prompts, mode, asset paths and image hashes.
- `production/captions.en.srt`, `production/captions.ko.srt`: subtitles for each edition.
- `production/narration.en.txt`, `production/narration.ko.txt`: scripts ready for a future voice recording.
- `production/storyboard.en.json`, `production/storyboard.ko.json`: scene/caption timing and source hashes.

Run `npm run captions` after changing scene timings or caption text. The exporter reads static authored JSX; it does not run a model or execute generated code.

## Checks

```bash
npm run lint
npm run build
npm run captions
npm run check:preview
```

The preview check requires a running Studio and an installed Chrome/Chromium. Set `FORWARDEDIT_STUDIO_URL` or `FORWARDEDIT_CHROME` if necessary. It seeks through the actual Studio WebMCP callbacks, checks eleven positions in each language and saves browser errors and screenshots under `out/qa/`. It does not export a video or run model experiments.

한국어판은 **ForwardEdit-Korean**을 선택합니다. 영어판이 공개용 기본 버전이며, 두 버전의 장면 순서와 길이는 같습니다. 화면 문구와 자막은 각 언어의 소스에서 별도로 편집할 수 있습니다.
