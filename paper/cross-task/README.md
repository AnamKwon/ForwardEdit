# Cross-task manuscript draft

**Working title:** *Beyond Loop Repair: When Do LLM-Written Weight Edits Help? A Pre-Registered Cross-Task Study*.

- [English manuscript](manuscript.md)
- [한국어 원고](manuscript.ko.md)
- [Short study summary](../../docs/CROSS_TASK_STUDY_1122.md)

**Relation to the [retrospective ForwardEdit draft](../README.md).** That draft audits five bundled case groups from public evidence. This draft reports a later prospective series: 15+ tasks, pre-registered sealed tests, strongest-baseline and LoRA comparisons, and Qwen3.5 up to 4B. It proposes five hypotheses:

- **H1, gap type:** whether a task is editable depends on the kind of gap.
- **H2, baseline erasure:** skill edits lose once the model may reason.
- **H3, knowledge rewriting:** writing stored associations is the robust niche.
- **H4, log-level edit signatures:** successful and failed edits leave different traces in the editor logs.
- **H5, structure vs values:** the LLM supplies the edit's structure; gradient training cannot rediscover that structure in the same budget, but it can improve the LLM's values.

**Evidence status.** The run artifacts are **not bundled**. The numbers come from host evaluation files and cannot be verified from this repository yet. ROME/MEMIT baselines are missing.
