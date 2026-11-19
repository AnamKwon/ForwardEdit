"""Optional MCP adapter for SDK1 and SDK2. All paths are relative to --workspace."""
import argparse
from pathlib import Path

from .mcp_compat import FastMCP

from .core import inspect, prepare, read_matrix, write_candidate


def build_server(root: Path) -> FastMCP:
    server = FastMCP("forwardedit")

    @server.tool()
    def inspect_checkpoint(checkpoint: str) -> dict:
        """List tensor names/shapes/dtypes and the source hash without loading all weights."""
        return inspect(root, checkpoint)

    @server.tool()
    def read_tensor_slice(checkpoint: str, tensor: str, rows: list[int], columns: list[int]) -> dict:
        """Read at most 4096 matrix values using explicit row/column indices."""
        return read_matrix(root, checkpoint, tensor, rows, columns)

    @server.tool()
    def preview_edit(plan: dict) -> dict:
        """Check a hypothesis-bearing plan and return numeric deltas; no behavioral evaluation."""
        return prepare(root, plan)[1]

    @server.tool()
    def create_candidate(plan: dict, output: str) -> dict:
        """Apply explicit set_values/scale/project_out/rank_one plan; no optimizer or fitting."""
        return write_candidate(root, plan, output)

    @server.tool()
    def write_parameter_values(checkpoint: str, source_sha256: str, tensor: str,
                               rows: list[int], columns: list[int], expected_values: list[list[float]],
                               values: list[list[float]], hypothesis: str, limits: dict, output: str) -> dict:
        """Write LLM-specified numeric values at exact coordinates after verifying the values it read."""
        return write_candidate(root, {"checkpoint": checkpoint, "source_sha256": source_sha256,
            "hypothesis": hypothesis, "limits": limits, "operations": [{"kind": "set_values",
                "tensor": tensor, "rows": rows, "columns": columns,
                "expected_values": expected_values, "values": values}]}, output)

    @server.tool()
    def evaluate_baseline(model_dir: str, dataset_path: str, splits: list[str] | None = None) -> dict:
        """Fresh local CPU FP32 baseline token metrics; optional explicit splits."""
        from .evaluation import evaluate_baseline as run
        return run(root, model_dir, dataset_path, splits)

    @server.tool()
    def diagnose_model(model_dir: str, dataset_path: str, tensors: list[str], split: str = "discovery",
                       max_examples: int = 8, top_k: int = 8, candidate_dir: str | None = None,
                       device: str = "cpu") -> dict:
        """Observe original/edited forward activations only; no backward or gradients."""
        from .diagnostics import diagnose_model as run
        return run(root, model_dir, dataset_path, tensors, split, max_examples, top_k,
                   candidate_dir=candidate_dir, device=device)

    @server.tool()
    def describe_parameter(model_dir: str, tensor: str, rows: list[int], columns: list[int],
                           candidate_dir: str | None = None) -> dict:
        """Read actual values with module ownership, structural role and explicit input/output axes."""
        from .inference import describe_parameter as run
        return run(root, model_dir, tensor, rows, columns, candidate_dir)

    @server.tool()
    def score_candidate(model_dir: str, candidate_dir: str, dataset_path: str,
                        splits: list[str] | None = None, device: str = "cpu") -> dict:
        """Development forward feedback after LLM-authored edits; excludes heldout and all training."""
        from .inference import score_candidate as run
        return run(root, model_dir, candidate_dir, dataset_path, splits, device)

    @server.tool()
    def probe_model(model_dir: str, dataset_path: str, candidate_dir: str | None = None,
                    split: str = "discovery", max_examples: int = 2, top_k: int = 5,
                    max_new_tokens: int = 16, device: str = "cpu", stop_at_eos: bool = False,
                    use_cache: bool = False) -> dict:
        """Next-token probabilities and actual greedy continuation on CPU/CUDA, inference only."""
        from .inference import probe_model as run
        return run(root, model_dir, dataset_path, candidate_dir, split, max_examples,
                   top_k, max_new_tokens, device, stop_at_eos, use_cache)

    @server.tool()
    def analyze_parameter_contexts(model_dir: str, dataset_path: str, tensor: str,
                                   rows: list[int], columns: list[int], pairs: list[dict], output: str,
                                   proposed_values: list[list[float]] | None = None,
                                   device: str = "cpu", candidate_dir: str | None = None) -> dict:
        """Save signed token-position contributions and fixed-input predictions; no parameter changes."""
        from .analysis import analyze_parameter_contexts as run
        return run(root, model_dir, dataset_path, tensor, rows, columns, pairs, output,
                   proposed_values, device, candidate_dir)

    @server.tool()
    def prepare_edit_proposal(analysis_path: str, basis: dict, limits: dict, output: str) -> dict:
        """Bind rationale/numeric derivation to analysis; defer or prepare a causal probe, never certify causation."""
        from .analysis import prepare_edit_proposal as run
        return run(root, analysis_path, basis, limits, output)

    @server.tool()
    def confirm_generation(frozen_path: str, dataset_path: str, original: bool = False) -> dict:
        """Heldout forward only after candidate/protocol/data hashes are frozen; no reselection."""
        from .confirmation import confirm_generation as run
        return run(root, frozen_path, dataset_path, original)

    @server.tool()
    def compare_candidate_parameters(model_dir: str, candidate_dir: str,
                                     parent_candidate_dir: str | None = None) -> dict:
        """Read exact parent/child/original values and retained/restored/new changes; never runs a model."""
        from .lineage import compare_candidate_parameters as run
        return run(root, model_dir, candidate_dir, parent_candidate_dir)

    @server.tool()
    def revise_candidate(model_dir: str, parent_candidate_dir: str, operations: list[dict],
                         hypothesis: str, rationale: dict, limits: dict, output: str) -> dict:
        """Apply LLM-specified incremental edits to a verified parent, retaining exact cumulative history."""
        from .lineage import revise_candidate as run
        return run(root, model_dir, parent_candidate_dir, operations, hypothesis, rationale, limits, output)

    @server.tool()
    def read_parameter_entries(checkpoint: str, tensor: str, flat_indices: list[int]) -> dict:
        """Read exact explicit entries of any tensor rank, including normalization weights/biases."""
        from .computation import read_parameter_entries as run
        return run(root, checkpoint, tensor, flat_indices)

    @server.tool()
    def describe_model_structure(model_dir: str, prefix: str = "", max_modules: int = 128,
                                 offset: int = 0) -> dict:
        """Page actual module ownership, parameter shapes and shared identities; no forward."""
        from .computation import describe_model_structure as run
        return run(root, model_dir, prefix, max_modules, offset)

    @server.tool()
    def inspect_module_computation(model_dir: str, modules: list[str], candidate_dir: str | None = None) -> dict:
        """Read actual loaded forward source, standard equations and owned parameters; no forward."""
        from .computation import inspect_module_computation as run
        return run(root, model_dir, modules, candidate_dir)

    @server.tool()
    def analyze_matrix_geometry(checkpoint: str, tensor: str, rows: list[int], columns: list[int],
                                rank: int = 4, right_operand: dict | None = None) -> dict:
        """Read bounded weight SVD and optional A @ B; does not choose or write edits."""
        from .computation import analyze_matrix_geometry as run
        return run(root, checkpoint, tensor, rows, columns, rank, right_operand)

    @server.tool()
    def build_repetition_contexts(reference_path: str, probe_paths: list[str], example_ids: list[str],
                                  output: str, ngram_size: int = 8, min_occurrences: int = 3,
                                  max_positions: int = 4) -> dict:
        """Build discovery-only paired contexts from saved actual original generations; no forward or edits."""
        from .contexts import build_repetition_contexts as run
        return run(root, reference_path, probe_paths, example_ids, output, ngram_size, min_occurrences, max_positions)

    @server.tool()
    def trace_token_flow(model_dir: str, dataset_path: str, selections: list[dict], examples: list[dict],
                         output: str, split: str = "discovery", candidate_dir: str | None = None,
                         device: str = "cpu") -> dict:
        """Trace actual selected module/token values and live tensor identity links via frozen forward."""
        from .computation import trace_token_flow as run
        return run(root, model_dir, dataset_path, selections, examples, output, split, candidate_dir, device)

    @server.tool()
    def research_export_tensor(checkpoint: str, tensor: str, output: str) -> dict:
        """Export a bounded exact parameter tensor as a hash-bound CPU analysis array; no edits or forward."""
        from .research_analysis import research_export_tensor as run
        return run(root,checkpoint,tensor,output)

    @server.tool()
    def research_prepare_edit(plan: dict, tensor: str, rows: list[int], columns: list[int],
                              contexts: list[dict], direction: list[float], basis: dict, output: str) -> dict:
        """Analyze explicit affine edits against verified recorded inputs, without a new forward or fitting."""
        from .research_analysis import research_prepare_edit as run
        return run(root, plan, tensor, rows, columns, contexts, direction, basis, output)

    @server.tool()
    def research_make_contexts(job_id: str, ids: list[str], output: str, which: str = 'original',
                               max_tokens: int = 8192) -> dict:
        """Materialize actual generation prefixes for later teacher-forced observations; no inferred labels."""
        from .research_analysis import research_make_contexts as run
        return run(root, job_id, ids, output, which, max_tokens)

    @server.tool()
    def evaluate_candidate(model_dir: str, candidate_dir: str, dataset_path: str, options: dict) -> dict:
        """Single candidate confirmation; options contain predeclared min_nll_gain and gates."""
        from .evaluation import evaluate_candidate as run
        return run(root, model_dir, candidate_dir, dataset_path, **options)

    @server.tool()
    def run_experiment(config: dict, output: str) -> dict:
        """Bounded validation-only selection of supplied plans, frozen then confirmed once."""
        from .experiment import run_experiment as run
        return run(root, config, output)

    @server.tool()
    def research_forward(spec: dict) -> dict:
        """Reserve a bounded asynchronous frozen forward; never selects edits."""
        from .research import research_forward as run
        return run(root, spec)

    @server.tool()
    def research_status(job_id: str) -> dict:
        """Reconcile terminal output or worker death and return compact counters."""
        from .research import research_status as run
        return run(root, job_id)

    @server.tool()
    def research_result(job_id: str) -> dict:
        """Return verified terminal metrics and hashed artifact manifests."""
        from .research import research_result as run
        return run(root, job_id)

    @server.tool()
    def research_read_array(path: str, indices: list[list[int]]) -> dict:
        """Read up to 4096 explicit finite entries of a workspace numpy array."""
        from .research import research_read_array as run
        return run(root, path, indices)

    @server.tool()
    def research_read_generations(job_id: str, ids: list[str], max_chars: int = 300,
                                  which: str = "original") -> dict:
        """Read selected actual generation excerpts with bounded text and grades."""
        from .research import research_read_generations as run
        return run(root, job_id, ids, max_chars, which)

    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve(strict=True)
    if not root.is_dir():
        parser.error("workspace must be a directory")
    # SDK stdio uses worker threads for file I/O. Keep the event loop scheduled
    # on runtimes where a thread completion alone fails to wake its selector.
    import anyio

    async def serve():
        async def tick():
            while True:
                await anyio.sleep(0.05)
        async with anyio.create_task_group() as group:
            group.start_soon(tick)
            try:
                await build_server(root).run_stdio_async()
            finally:
                group.cancel_scope.cancel()

    anyio.run(serve)


if __name__ == "__main__":
    main()
