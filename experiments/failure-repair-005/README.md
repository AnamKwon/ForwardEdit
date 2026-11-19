# Opposing controls and normal damage: TinyStories 005

[Case index](../README.md) · [Historical report](../../docs/FAILURE_REPAIR_005.md)

This follow-up replaced an earlier failure proxy with actual generated repetition contexts. The editor performed candidate 01 → 02 → 03 revisions and an opposing, norm-matched candidate 04 from the original. Restoring and reducing edits lowered normal-case damage, but no candidate passed the development gate. Candidate 03 was retained only for diagnosis and independent confirmation.

| Independent metric, 128 stories | Original | Candidate 03 |
|---|---:|---:|
| Mean repeated 8-gram fraction | 6.0597% | 6.5118% |
| Cutoff rate | 68.75% | 66.4063% |

Repetition reduction was -0.4521 percentage points, with 95% interval [-2.5611, 1.6950] points. Normal reference NLL increased by 0.002622; 91/128 stories had worse reference NLL. Candidate adoption failed. Both directions sometimes improved development repetition, weakening the simple directional mechanism story.

Read [protocol](protocol.json), [retrospective diagnosis](retrospective-diagnosis.json), [initial basis](basis-01.json), [iteration log](iterations.json), [actual editor notes](notes.md), [MCP receipts](receipts.jsonl), [final decision](decision.json), [freeze](frozen-choice.json), [held-out result](heldout-result.json), [parameter/normal audit](parameter-normal-audit.json) and [final audit](final-audit.json). Candidate manifests preserve explicit values, restores and lineage.

## Unaddressed criticisms / 미대응 비판

The initial interpretation relied on a few contexts and whole writer columns also served normal predictions. Controls did not establish a repetition-specific mechanism. The independent mean worsened and its interval includes zero. Aggregate NLL preservation within a tolerance is not improvement. Strongest unedited configurations, unrelated-task preservation and general transfer were not established. The diagnostic candidate already failed development selection; its large confirmation is not a recommended default resource policy.
