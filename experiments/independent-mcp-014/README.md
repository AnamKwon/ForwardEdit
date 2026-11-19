# MCP-only Qwen study: sessions 009–014

[Case index and session timeline](../README.md) · [Exact aggregate report](study-report.json)

Fresh official Claude/Codex client sessions used the project MCP and skills without loop-engineering. This was one continued study: Codex inherited Claude's candidate and observations. Earlier environment, budget and permission failures are retained, so these sessions must not be presented as independent provider-quality trials.

| Fixed 64-problem development comparison | Correct | Original correct lost |
|---|---:|---:|
| Original | 5 | — |
| Claude initial candidate | 7 | 3 / 5 |
| Codex revision | 10 | 3 / 5 |

The Codex editor read the parent's generations/results before restoring three values. This is one verified feedback-dependent revision. The final candidate changed 3,069 stored values, with 3,067 actual GPU BF16 changes. Aggregate gold-reference preservation passed, but the 60% loss of originally correct development answers failed the answer-preservation gate. The candidate was frozen as `diagnostic_only`.

Independent 192-problem accuracy was **25 → 22 correct**, with 13 wrong-to-right and 16 right-to-wrong changes. The gain was -1.5625 percentage points; the 97.5% paired interval was [-7.8125, 4.6875] points. The loss of originally correct answers was 16/25 = 64%. Improvement and preservation gates failed. The diagnostic confirmation did not justify adopting the candidate.

Eight successful development logical jobs and two independent paired jobs were executed, with one earlier aborted accepted attempt. This is not a 50-pass reproduction. Usage records span all six sessions; historical API-equivalent estimates are not actual subscription invoices or current prices.

Read the [session timeline](../README.md#same-study-session-timeline-009014), [protocol](protocol.json), [user request](user-request.txt), [actual editor events](editor-events.jsonl), [development review](development-review.json), [feedback-before-revision audit](feedback-flow-audit.json), [parent-feedback behavior](parent-feedback-behavior.json), [final public editor message](editor-final-public.txt), [frozen candidate](workspace/frozen-candidate.json) and [final aggregate report](study-report.json). [MCP audit](mcp-only-audit.json) and individual `workspace/mcp-audit/` requests/responses retain tool activity. Candidate manifests and job result/snapshot JSON are included; arrays and long generations they reference are omitted.

The editor's final message predates host-only independent confirmation and correctly states that it had not seen that test. Later independent results are in `study-report.json`. Earlier session records are under the adjacent `independent-mcp-009` through `independent-mcp-013` directories.

## Unaddressed criticisms / 미대응 비판

The observed development gain did not transfer, and both development and independent old-correct preservation failed. Aggregate reference NLL cannot substitute for answer preservation. There was only one feedback revision, not a deep iterative-improvement study. The host knew earlier studies, sessions inherited artifacts, and client restrictions are not OS-container isolation. Strongest unedited/steering baselines, broad task preservation and algorithmic advantage over WSP were not established. Running a large independent confirmation for an already ineligible diagnostic candidate was a resource-policy limitation, not a recommended default workflow.
