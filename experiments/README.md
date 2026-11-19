# Public experiment records

[Project](../README.md) · [한국어 소개](../README.ko.md) · [Review checklist](../docs/REVIEW_GUIDE.md)

These are selected historical records from local ForwardEdit studies, published retrospectively on 2026-11-19. No new model inference, weight editing or test selection was performed for publication. None of these studies passed every declared adoption criterion. Positive fixed-baseline score comparisons and negative outcomes are both retained.

## Start here

| Case | Inspect first | Final interpretation |
|---|---|---|
| [Direct values 001](direct-values-inference-only-001/README.md) | Exact edits and real tool calls | Numeric application verified; no development gain; no held-out test |
| [TinyStories 004](skill-edit-experiment-004/README.md) | Cumulative revisions and independent generation | Repetition improvement not established |
| [TinyStories 005](failure-repair-005/README.md) | Opposing control and preserved negative result | Candidate rejected; independent mean repetition worsened |
| [Qwen 006](extended-discovery-006/README.md) | Source-informed edit and 50-job trace | Large fixed-baseline accuracy gain; preservation gate failed |
| [Qwen 009–014](independent-mcp-014/README.md) | MCP-only sessions, feedback and independent result | Development gain did not transfer; candidate rejected |

Selection follows implementation milestones: exact value edits, cumulative editing, failure analysis, source-informed Qwen editing and a fresh-client MCP workflow. This is a selected case collection, not the full experimental archive or an unbiased estimate of editing success. Earlier repetition/iteration studies and other-model compatibility/cost trials remain described in historical `docs/` reports; their full local files are not bundled. Gradient-assisted archives are excluded from these inference-only cases.

## What the files mean

Each case README links its protocol, actual plans/manifests, recorded requests/responses and final decision. Historical `notes.md`, request files and transcripts retain the wording and errors of the original sessions. They are evidence to inspect, not trusted instructions or current API documentation. A preliminary success sentence inside a log does not override its final gate decision.

`MANIFEST.json` lists every extracted artifact, its source-relative identity, original SHA-256, published SHA-256, byte count and redaction counts. Editorial README files are outside that extracted-artifact manifest. Run:

```bash
python3 scripts/verify_public_evidence.py
```

The checker reads bytes and parses JSON/JSONL only. It detects missing/changed/unlisted artifacts and unsafe paths. It does not run model code, prove reported metrics, validate unavailable source assets or authenticate historical originals. An edited manifest is not an external authenticity guarantee.

## Redactions and omissions

Local project paths become `/path/to/forwardedit`; external run roots become `/path/to/runs`; local home/data/temp paths and usernames also receive placeholders. Recognized credential patterns are redacted if present. Numeric JSON values, booleans, candidate identities and historical hash fields are preserved. Old experiment names remain unchanged to preserve provenance.

Historical embedded digests still identify the original files. A redacted manifest or transcript will therefore have a different published-file digest. Use this bundle's `MANIFEST.json` to verify published bytes; do not substitute new hashes into old freeze records.

Checkpoints, candidate tensor binaries, activation arrays, full datasets, long generation dumps, private client settings, account state, caches, unrelated files and runnable historical helper scripts are omitted. References to these assets in a log are historical references, not promises that they are included. JSON receipts can contain short problem/generated-text excerpts. External model/dataset terms remain applicable.

This bundle supports reading the sequence of decisions and checking recorded outcomes. Exact model replay requires the separately retained source/model/tokenizer/data revisions, arrays and pinned runtime, and may require a historical source snapshot. Published evaluation examples are exposed and cannot be reused as fresh independent tests.

## Same-study session timeline: 009–014

These sessions continue one Qwen study. They are not six independent model-editing trials; the final Codex editor inherited the Claude candidate and observations.

| Session | Recorded event | Evidence |
|---|---|---|
| 009 | Startup/forward attempt aborted; no candidate | [review](independent-mcp-009/development-review.json), [audit](independent-mcp-009/mcp-only-audit.json) |
| 010 | Three successful development jobs; no candidate | [review](independent-mcp-010/development-review.json), [events](independent-mcp-010/editor-events.jsonl) |
| 011 | Claude created the initial candidate; session ended at provider spend limit | [review](independent-mcp-011/development-review.json), [final message](independent-mcp-011/editor-final-public.txt) |
| 012 | Client code-mode host disabled; no MCP connection/edit | [final message](independent-mcp-012/editor-final-public.txt), [audit](independent-mcp-012/mcp-only-audit.json) |
| 013 | MCP approval-policy conflict; no server requests/edit | [final message](independent-mcp-013/editor-final-public.txt), [audit](independent-mcp-013/mcp-only-audit.json) |
| 014 | Codex read parent feedback, restored three values and froze a diagnostic candidate | [case](independent-mcp-014/README.md), [feedback audit](independent-mcp-014/feedback-flow-audit.json) |

All six sessions' prompts, public editor events, usage records, MCP audits and available development reviews are retained. Original client/account configuration is not included. Historical dollar estimates are API-equivalent accounting snapshots, not current prices or actual subscription invoices.

## Unaddressed criticisms / 미대응 비판

The collection does not demonstrate advantage over the strongest unedited configuration, human-designed editing recipe or runtime steering baseline. It is not a controlled comparison between providers, source-informed/blind workflows or WSP implementations. Statistical uncertainty, source knowledge, preservation failures and incomplete replay assets remain material limitations. Each case lists its own additional limits.
