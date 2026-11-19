# ForwardEdit workflow explainer

English · [한국어](README.ko.md)

An **English-first public explainer**, with a matching Korean edition: 104 seconds, 1920×1080, 30 fps. It explains how an external LLM observes weights, specifies an edit through MCP, reads feedback and verifies a frozen candidate. Both editions use on-screen captions; recorded narration and music are not included.

## Preview and edit

```bash
cd videos/forwardedit-explainer
npm ci
npm run dev -- --port=3333
```

Select **ForwardEdit-English** for the primary public edition or **ForwardEdit-Korean** for Korean. Each edition also has an individual scene folder. Text, images and scenes have authored Studio editing targets. Matrix cells and the progress indicator are computed from code.

To export a requested MP4, use Studio's **Render** button, or run:

```bash
npx remotion render ForwardEdit-English out/forwardedit-en.mp4
npx remotion render ForwardEdit-Korean out/forwardedit-ko.mp4
```

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
| 01:34–01:44 | Write correctness, behavioral effect, independent improvement and limitations |

## What the video represents

- The matrix, `0.120 → 0.110`, input `+0.50` and API screens are **illustrative values and abbreviated mock calls**, not recorded model observations or a new experiment. No model was executed or edited for this video.
- Connection helpers are drawn from `standalone.py`; basic FP32 observation/edit examples are drawn from `server.py` and `docs/USAGE.md`. Available tools and evaluation paths differ by runtime. This is a conceptual overview, not a complete executable call transcript.
- Fixed-input local arithmetic does not establish a model-wide behavioral prediction. `ready_for_causal_probe` denotes preparation for a bounded intervention, not a causal or performance certificate.
- The workflow uses explicit CPU arithmetic and inference-only model execution. It does not use backward, gradients, optimizers, automatic loss fitting, coordinate/strength sweeps or GPU training.
- The video does not claim general model improvement or superiority over WSP. It retains the README's limitation that every published study failed at least one declared acceptance criterion.
- The two AI-generated illustrations are conceptual art, not a depiction of actual layer counts, weight arrangement or a complete computation graph.

Sources: the repository's [README](../../README.md), [Korean README](../../README.ko.md), [architecture](../../DESIGN.md), [API guide](../../docs/USAGE.md), [execution contract](../../docs/INFERENCE_ONLY_CONTRACT.md), [review guide](../../docs/REVIEW_GUIDE.md) and actual tool definitions. The production storyboards record SHA-256 hashes of those source files.

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

The preview check requires a running Studio and an installed Chrome/Chromium. Set `FORWARDEDIT_STUDIO_URL` or `FORWARDEDIT_CHROME` if necessary. It seeks through the actual Studio WebMCP callbacks, checks nine positions in each language and saves browser errors and screenshots under `out/qa/`. It does not export a video or run model experiments.

한국어판은 **ForwardEdit-Korean**을 선택합니다. 영어판이 공개용 기본 버전이며, 두 버전의 장면 순서와 길이는 같습니다. 화면 문구와 자막은 각 언어의 소스에서 별도로 편집할 수 있습니다.
