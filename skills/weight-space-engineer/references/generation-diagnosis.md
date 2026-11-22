# Diagnose when a local edit fails to transfer

Use when a measured token or activation changes and end-to-end performance does
not follow, or when output length and automatic scores tell different stories.

1. **Audit the outputs.** Compare verified jobs with
   `research_compare_generations(reference, candidate, output)`. Each run is
   `{"job_id":"…","model":"original"}` or uses `"edited"`. The jobs must preserve
   source configuration, split, prompt token IDs, and batch/sample order.
   Separate parser grades, termination, and output length. If the task has a
   single-line answer wrapper, pass its literal opening and closing strings as
   `final_answer_envelope`. Otherwise review bounded generation windows with the
   task's own answer rules. For example, check whether a matched answer was
   submitted, quoted while discussing the instruction, or followed by more
   reasoning. Keep automatic and audited scores explicit. The tool reports exact
   repeated token windows in the last 2,048 tokens as triage evidence. Labels for
   semantic repetition and correct stopping require reading the response.
2. **Localize a difference.** Inspect the first differing continuation offset
   on repairs, regressions, and preserved cases. A changed token or a repeated
   window is an observation rather than a semantic error label. If one run ends
   first, the longer run may have no matching token from the shorter run. Check
   finish reasons before interpreting that gap. Compare unconstrained generation
   and scaffolds within their own paired conditions.
3. **Measure the two choices.** Use the same generation prompts, batch order,
   backend, and dtype with Transformers and opt in to
   `observations.generation_logits`. Supply explicit `token_ids` and
   `positions` mapping example IDs to zero-based continuation offsets. Include
   both observed competing tokens. Include the model's actual EOS IDs when the
   hypothesis concerns stopping. Verify that the replay produced the expected
   prefix and choices before using its logits as evidence. Raw values at a
   shared prefix support a statement about local token preference. After paths
   diverge, raw scores are conditioned on different inputs.
4. **Probe an exact shared prefix if needed.** Call `research_make_contexts`
   with `sample` and per-ID `continuation_lengths` taken from recorded offsets.
   It retains the whole input prompt and refuses to silently shorten the chosen
   prefix. Compare both models at that same input with `first_token_ids` or use
   teacher-forced observations for the module and position of interest. Identify
   a fresh prefill as such. Its kernel shape may differ from the original cached
   decode and can affect near ties.
5. **Test the proposed mechanism.** Choose probes that can separate input
   interpretation, intermediate state updates, answer selection, and termination
   according to the current evidence. Check nearby valid positions and unrelated
   cases before attributing a representation to a task role. Record the local
   prediction, disconfirming outcome, and expected end-to-end effect before any
   new write. A local shift or better formatting can coexist with unchanged
   accuracy. Carry both repairs and regressions into the validation decision.
