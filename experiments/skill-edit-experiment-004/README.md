# Cumulative editing: TinyStories 004

[Case index](../README.md) · [Historical report](../../docs/SKILL_EDIT_EXPERIMENT_004.md)

An external Claude editor used project skills and MCP observations to author three primary edits and one control. Two cumulative revision edges were recorded. Actual parameter application was checked; the final candidate changed 256 values. A 64-story independent confirmation did not establish repetition improvement.

| Independent metric | Original | Candidate |
|---|---:|---:|
| Mean repeated 8-gram fraction | 7.2791% | 7.0812% |
| Cutoff rate | 71.875% | 64.0625% |

Mean repetition reduction was 0.1979 percentage points, with a 95% paired interval of [-2.9104, 3.2062] points. Reference NLL increased by 0.006389. The declared repetition gate failed, and the candidate was not adopted.

Read [protocol](protocol.json), [editor request](request.md), [initial basis](basis-01.json), [proposal](proposal-01.json), [iteration log](iterations.json), [editor notes](notes.md), [complete retained MCP receipts](receipts.jsonl), [frozen choice](frozen-choice.json), [held-out result](heldout-result.json) and [verification summary](verification-summary.json). [Qualitative examples](qualitative-examples.json) retain changed generations; [host audit](host-audit.json) records numerical checks. Candidate plans are in `candidate-*/manifest.json`.

## Unaddressed criticisms / 미대응 비판

Development gains were concentrated in one example, and early failure proxies did not accurately represent generated repetition. Independent confidence intervals include zero. Repetition and reference NLL do not establish narrative quality or new ability. Strongest unedited/steering baselines and transfer to other models were not established. Large activation arrays, tensor binaries and full text datasets are omitted.
