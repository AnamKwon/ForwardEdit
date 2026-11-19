# Direct numeric edits: TinyStories 001

[Case index](../README.md) · [Historical report](../../docs/DIRECT_VALUES_VERIFICATION.md)

Two candidates each changed six actual values in an existing TinyStories checkpoint through MCP. The records verify manual plan arithmetic, loaded model values and CPU/CUDA inference. Both candidates worsened development NLL and were rejected. No held-out evaluation was performed.

Read [the contract](contract.txt), [actual tool ledger](ledger.json), [candidate 01 old/new values](candidate-01/manifest.json), [candidate 02](candidate-02/manifest.json), [host audit](host-audit.json) and [final decision](decision.json), in that order. The manifests include exact expected/replacement numbers and a semantic hypothesis that was only weakly supported or contradicted.

The audit reports baseline validation NLL 2.925428, candidate 01 2.926468 and candidate 02 2.938720. Correct parameter application did not establish an improvement. Candidate tensor binaries and the model are omitted; these manifests are inspectable plans, not standalone model downloads.

## Unaddressed criticisms / 미대응 비판

Only 12 discovery and 12 validation stories were used. Initial neuron selection lacked token-position activation signs, and a newline-format mismatch need not be a model failure. Semantic localization, independent performance, strongest unedited baselines and broader preservation were not established. This is a numerical-edit demonstration, not evidence of new ability or WSP superiority.
