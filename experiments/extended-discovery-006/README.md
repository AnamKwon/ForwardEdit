# Source-informed Qwen editing: study 006

[Case index](../README.md) · [Recorded outcome](study-outcome.json)

This Qwen3.5-0.8B study used 50 successful logical development jobs, 51 accepted attempts including one environment failure, and two independent paired jobs. These are batches, not epochs, internal autoregressive forward counts or 50 weight updates. The selected source-informed host branch edited MLP down projections in layers 15 and 19. The original WSP method informed the intervention; this is not blind rediscovery. Published learned vectors were not imported, according to the retained record.

| Independent official subset | Items | Original correct | Edited correct |
|---|---:|---:|---:|
| GSM8K | 128 | 13 | 76 |
| MATH-500 | 64 | 5 | 27 |
| Combined selected subset | 192 | 18 | 103 |

The fixed-baseline gain was 44.2708 percentage points, with a 97.5% paired interval [35.9375, 52.6042] points. There were 86 wrong-to-right changes and one right-to-wrong change. All 86 rescues came from previously unparsed answers in the retained analysis. The original unparsed count was 174, versus 73 edited; this does not by itself distinguish every cutoff from other parsing failures.

**Overall adoption failed.** One of 18 originally correct answers was lost: 5.5556%, above the frozen 5% limit. `study_complete` remains false and final host approval was rejected. The positive accuracy comparison is preserved together with this failed gate; thresholds were not relaxed after test exposure.

The selected candidate changed 124,928 stored values; 123,452 changes remained effective at GPU BF16 precision. Model/tensor binaries are omitted.

Read [protocol](artifacts/protocol.json), [source-selected weight samples](source-selected-weight-values.json), [explicit edit](source-informed-edit-created.json), [candidate manifest](artifacts/candidates/source-informed-l15l19-001/manifest.json), [50-job/candidate trace](research-fifty-trace.json), [control analysis](causal-and-transfer-interim-analysis.json), [freeze](final-freeze.json), [independent receipt](independent-confirmation-receipt.json), [per-problem confirmation](host-confirmation/confirmation-report.json), [final outcome](study-outcome.json) and [rejected final review](final-review-response.json). [Data amendment timing](data-amendment-timing-clarification.json) preserves the answer-correction history. [Post-test integrity](post-test-freeze-integrity.json) records retained frozen assets.

## Unaddressed criticisms / 미대응 비판

The comparison does not establish superiority over thinking-off/other strong unedited configurations, runtime steering or a non-LLM recipe. The source-informed host branch is not independent evidence that an editor discovered the source mechanism from scratch. The result is a selected official subset under fixed decoding, not a full-benchmark score or newly learned mathematics. Conditional answer preservation failed, and broad unrelated-task preservation was not established. Model/array/data assets and historical runtime are not fully distributed, so this text bundle alone cannot reproduce inference. No inference or selection was rerun for publication.
