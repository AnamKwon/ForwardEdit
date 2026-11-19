import argparse
import json
from pathlib import Path

from .core import inspect, local, prepare, write_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect").add_argument("checkpoint")
    for name in ("preview", "apply"):
        command = sub.add_parser(name)
        command.add_argument("plan")
        if name == "apply":
            command.add_argument("output")
    baseline = sub.add_parser("baseline")
    baseline.add_argument("model_dir")
    baseline.add_argument("dataset_path")
    baseline.add_argument("--splits", nargs="+")
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("model_dir")
    diagnose.add_argument("dataset_path")
    diagnose.add_argument("tensors", nargs="+")
    diagnose.add_argument("--split", default="discovery")
    diagnose.add_argument("--max-examples", type=int, default=8)
    diagnose.add_argument("--top-k", type=int, default=8)
    diagnose.add_argument("--candidate-dir")
    diagnose.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    describe = sub.add_parser("describe")
    describe.add_argument("model_dir")
    describe.add_argument("tensor")
    describe.add_argument("--rows", nargs="+", type=int, required=True)
    describe.add_argument("--columns", nargs="+", type=int, required=True)
    describe.add_argument("--candidate-dir")
    score = sub.add_parser("score")
    score.add_argument("model_dir")
    score.add_argument("candidate_dir")
    score.add_argument("dataset_path")
    score.add_argument("--splits", nargs="+", default=["discovery", "validation"])
    score.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    probe = sub.add_parser("probe")
    probe.add_argument("model_dir")
    probe.add_argument("dataset_path")
    probe.add_argument("--candidate-dir")
    probe.add_argument("--split", default="discovery")
    probe.add_argument("--max-examples", type=int, default=2)
    probe.add_argument("--top-k", type=int, default=5)
    probe.add_argument("--max-new-tokens", type=int, default=16)
    probe.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    probe.add_argument("--stop-at-eos", action="store_true")
    probe.add_argument("--use-cache", action="store_true")
    analyze = sub.add_parser("analyze-parameter")
    analyze.add_argument("spec", help="JSON with model/dataset/tensor/coordinates/pairs and optional proposed_values")
    analyze.add_argument("output")
    proposal = sub.add_parser("prepare-proposal")
    proposal.add_argument("analysis_path")
    proposal.add_argument("request", help="JSON containing basis and limits")
    proposal.add_argument("output")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("model_dir")
    evaluate.add_argument("candidate_dir")
    evaluate.add_argument("dataset_path")
    evaluate.add_argument("--options", required=True, help="workspace JSON gate options fixed before heldout")
    experiment = sub.add_parser("run-experiment")
    experiment.add_argument("config")
    experiment.add_argument("output")
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect(args.workspace, args.checkpoint)
    elif args.command == "baseline":
        from .evaluation import evaluate_baseline
        result = evaluate_baseline(args.workspace, args.model_dir, args.dataset_path, args.splits)
    elif args.command == "diagnose":
        from .diagnostics import diagnose_model
        result = diagnose_model(args.workspace, args.model_dir, args.dataset_path, args.tensors,
            args.split, args.max_examples, args.top_k, candidate_dir=args.candidate_dir, device=args.device)
    elif args.command == "describe":
        from .inference import describe_parameter
        result = describe_parameter(args.workspace, args.model_dir, args.tensor,
            args.rows, args.columns, args.candidate_dir)
    elif args.command == "score":
        from .inference import score_candidate
        result = score_candidate(args.workspace, args.model_dir, args.candidate_dir,
            args.dataset_path, args.splits, args.device)
    elif args.command == "probe":
        from .inference import probe_model
        result = probe_model(args.workspace, args.model_dir, args.dataset_path, args.candidate_dir,
            args.split, args.max_examples, args.top_k, args.max_new_tokens, args.device,
            args.stop_at_eos, args.use_cache)
    elif args.command == "analyze-parameter":
        from .analysis import analyze_parameter_contexts
        spec = json.loads(local(args.workspace, args.spec).read_text())
        result = analyze_parameter_contexts(args.workspace, output=args.output, **spec)
    elif args.command == "prepare-proposal":
        from .analysis import prepare_edit_proposal
        request = json.loads(local(args.workspace, args.request).read_text())
        result = prepare_edit_proposal(args.workspace, args.analysis_path,
            request["basis"], request["limits"], args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate_candidate
        options = json.loads(local(args.workspace, args.options).read_text())
        result = evaluate_candidate(args.workspace, args.model_dir, args.candidate_dir, args.dataset_path, **options)
    elif args.command == "run-experiment":
        from .experiment import run_experiment
        config = json.loads(local(args.workspace, args.config).read_text())
        result = run_experiment(args.workspace, config, args.output)
    else:
        plan = json.loads(local(args.workspace, args.plan).read_text())
        result = (prepare(args.workspace, plan)[1] if args.command == "preview"
                  else write_candidate(args.workspace, plan, args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
