# CPU arrays without Python execution

`research_export_tensor(checkpoint,tensor,output)` exports exact stored weights to `exports/*.npy`. Forward results bind observation `.npy` files to actual module, original/edited model, example ID, positions and source hashes. Read these identities before using an observation. An array reference is `{"path":"exports/weight.npy","sha256":"<returned hash>"}`. Supply only returned paths/hashes from this workspace.

`array_compute(operation,inputs,output,options)` writes a new `analysis/*.npy` plus a provenance JSON. Every input is an array reference. It returns shape, hash, norm, min/max and a bounded preview. Computations are CPU algebra only, never network execution, fitting or weight selection.

- `mean`, `sum`, `norm`: one input; options `axis` (integer, list or omitted), `keepdims` boolean. `normalize`: one input, `axis` optional; rejects zero norm.
- `add`, `subtract`, `multiply`, `divide`, `dot`, `matmul`: two inputs; standard NumPy shape rules. `scale`: one input and `factor` finite number. No fitted coefficients or gradient estimators.
- `transpose`: one input and optional `axes`. `reshape`: `shape` list.
- `slice`: `slices` list, each entry an integer or `[start,stop]`/`[start,stop,step]`; null is an open bound. `take`: `indices` explicit list and optional `axis`. Use actual observed/model axes.
- `concatenate`: inputs and `axis` (default 0). `abs`: one input.
- `topk`: one vector; `k` in 1..256, `absolute` optional. Output array contains indices, with signed `top_values` in the result. Ranking is a statistic, not proof of a semantic circuit.

For a JSON edit plan, `{"$array":{"path":"analysis/direction.npy","sha256":"<hash>"}}` can replace a numeric vector or matrix such as `direction`, `left`, `right`, `values` or `expected_values`. Tensor names, explicit indices, coefficients, limits, hypothesis and arithmetic remain the editor's decisions. `create_candidate_from_arrays(plan,output)` expands these values and calls the actual project's candidate writer. The resolved full numeric plan is retained. With a parent, `plan.operations` contains only the current incremental steps; `rationale` needs `observation`, `hypothesis`, `value_derivation`, `prediction`, `disconfirming_result`.

`prepare_edit_from_arrays` has the same fields as `research_prepare_edit`, except `direction` is an array reference and numeric fields in `plan` may contain `$array` references. It invokes the actual initial readiness checker on source-bound observations. Inspect returned `decision`/`reasons`. For a revision's readiness, supply the full proposed source-bound sequence (use `steps` for `kind:sequence`); for writing the revision supply only incremental operations. Keep both records and do not hide a `defer`.

There is no tool to execute arbitrary Python, fit a loss, compute gradients or select a model edit automatically. Arithmetic references are conveniences for explicit values, not automatic parameter learning. Shapes, previews, source/parent values and actual GPU effects must still be checked.
