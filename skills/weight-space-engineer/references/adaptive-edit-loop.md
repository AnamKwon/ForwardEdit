# Adaptive editing from accumulated forward evidence

Read when the user requests repeated parameter edits, trajectory analysis, or a
research budget comparable to a multi-pass experiment. This is a workflow for the
existing MCP tools, not an automatic parameter selector or a server-enforced
research state machine. Follow the inference-only contract throughout.

## Declare what completion means

Distinguish a budget ceiling from a requested research depth. If the user asks for
a study using a specified number of successful forward batches, do not silently
replace it with an optional ceiling and stop after one failed revision. State the
stopping policy before running. A useful next pass may collect evidence rather
than change weights. Stop for exhausted resources, unavailable execution, or no
justified next measurement; record the reason and the incomplete depth explicitly.
Do not fill a quota with duplicate passes, unsupported edits or a strength sweep.

Keep total provider responses/costs, setup failures, scientific forward batches,
candidate writes and feedback-dependent revisions separate. All applicable total
limits still include failed attempts; separating counters is not a budget reset.
Perform mechanical schema/runtime preflights before spending most editor capacity.

## Build a case bank, not just a score history

Use `record_observation` for source-bound case-bank updates and explicit links to
the earlier update. Include problem/split, model/patch identity, generation job and
artifact hashes, input/continuation token identity, selected positions, outcome
and the reason for selecting each case. Link arrays instead of printing vectors.
Use `workspace_status` and saved updates to recover state after a fresh session.

Maintain separate outcome fields: correct answer, wrong answer, unparsed answer,
token cutoff, and observed repetition. They can overlap: an unfinished generation
can already contain a correctly graded answer. Continuing to reason is not by
itself an error. Mark semantic step labels as checked, inferred or unknown.

Accumulate several kinds of evidence as needed:

- gold-reference teacher-forced traces;
- the model's own successful free generations;
- its failed/free-generation traces;
- originally correct cases newly damaged by the current edit;
- cases gained by a parent candidate but lost by its child.

Reference solutions are useful anchors, not unique valid token sequences. A first
textual difference from a reference or from the original model is not automatically
the first mathematical error.

## Localize before assigning a role

Start with sparse observations across plausible layers/module boundaries, then
increase temporal or case coverage near a suspected transition. Use the actual
module sources and axes. A large final-layer contrast can reflect an earlier
failure's consequence; it does not locate that failure's origin.

Compare matched problem/length/reasoning stages where possible. Different token
prefixes, easier problems and answer endings are confounds. Separate average
representations from token trajectories. Cross-case directions and signed matrix
contributions nominate hypotheses; controlled interventions must test them.

With vLLM generation, `research_make_contexts` plus a teacher-forced observation
pass can replay saved generated prefixes. Record the backend/precision distinction.
After an edit changes the actual trajectory, collect observations on its **new**
prefixes, particularly newly damaged cases. Replaying only the original successful
prefix cannot establish what happens on the edited model's free generation.

## Check the probability actually measured

Bind every probability claim to model/patch, prefix, prediction position and token
ID. `target_logprobs` scores the recorded next token. If the desired alternative
token is not that token, its probability is not measured by this array.

`1 - P(observed token)` is the probability of all other vocabulary tokens, not the
probability of one desired token. Unless the support is demonstrably binary, do
not equate them. Mark an unavailable alternative-token probability as unmeasured;
use an available legitimate observation or identify the missing tool capability.

Raw teacher-forced log-probability also differs from the sampling distribution
after temperature, penalties and truncation. Do not label raw probabilities as
actual sampling/termination probabilities. A projected hidden-state shift is not
itself a change in the desired token's probability.

## Close each feedback cycle

For each intervention record: hypothesis, supporting cases, explicit value rule,
predicted local effect, predicted free-generation effect, preservation condition,
falsifier and the exact jobs that will test these predictions. Use the existing
preparation tool before writing. Its ready status does not certify causality.

After inference, record which predictions held and which failed, then choose an
explicit disposition: retain, diagnose, revise, roll back or reject. Maintain a
best **accepted** checkpoint separately from the current probe. A harmful probe
does not become the accepted parent simply because it is the newest file.

If a mechanism is contradicted, identify the next unanswered question. Possible
next actions include new matched contexts, another module boundary, a controlled
reversal, or a different justified direction/target. Choose from evidence rather
than blindly repeating the same direction or enumerating coordinates/strengths.
Rolling back and starting a new branch can be a feedback-dependent decision;
record its lineage honestly instead of claiming cumulative improvement.

Use `compare_candidate_parameters` for numeric lineage and compare actual saved
generations for behavioral lineage. Check both original-to-child and parent-to-child
rescues/regressions. Neither file delta nor reference-average preservation proves
free-generation preservation. A component-removal control that fails its prediction
requires a new explanation, not an unearned success label.

Use small diagnostics to reject explanations cheaply. Only promising candidates
need the full fixed development evaluation. Short generations remain diagnostics.
Freeze a qualified candidate before independent confirmation; a diagnostic-only
freeze does not itself justify an expensive independent test. If no justified
candidate exists, retain the original and report the unmet objective. An already
exposed independent cohort cannot guide the next edit or candidate selection.
