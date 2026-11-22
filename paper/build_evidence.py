"""Extract manuscript numbers from public JSON; no model execution or fitting.

Run from any directory. --check compares retained output with current source bytes.
--figures additionally creates static SVG/PDF figures (requires matplotlib).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
SNAPSHOT = "32f62aef193fb6cac4dc84db6128657ecfe2be4f"


def extract():
    sources = {}

    def read(relative):
        raw = (ROOT / relative).read_bytes()
        sources[relative] = {"published_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return json.loads(raw)

    prefix = "experiments/"
    direct = read(prefix + "direct-values-inference-only-001/host-audit.json")
    direct_decision = read(prefix + "direct-values-inference-only-001/decision.json")
    repetitions = []
    for case in ["skill-edit-experiment-004", "failure-repair-005"]:
        d = read(prefix + case + "/heldout-result.json")
        p = read(prefix + case + "/protocol.json")
        b, c, comp = d["baseline"], d["candidate"], d["comparison"]
        assert b["prompts"] == c["prompts"] == len(b["per_prompt"]) == len(c["per_prompt"])
        assert abs((b["mean_repeat_fraction"] - c["mean_repeat_fraction"]) - comp["mean_repeat_gain"]) < 1e-12
        repetitions.append({
            "case": case, "items": b["prompts"], "metric": "mean repeated generated 8-gram fraction",
            "original_percent": 100 * b["mean_repeat_fraction"],
            "edited_percent": 100 * c["mean_repeat_fraction"],
            "gain_pp": 100 * comp["mean_repeat_gain"],
            "gain_interval_pp": [100 * v for v in comp["gain_ci95"]], "confidence": 0.95,
            "original_cutoff_percent": 100 * b["cutoff_rate"],
            "edited_cutoff_percent": 100 * c["cutoff_rate"],
            "reference_nll_increase": comp["reference_nll_increase"],
            "adoption_passed": d["adoption_passed"],
            "validation_eligible": d["validation_eligible"],
            "bootstrap_draws": p["confirmation"]["paired_bootstrap_draws"],
            "bootstrap_seed": p["confirmation"]["bootstrap_seed"],
        })
    outcome = read(prefix + "extended-discovery-006/study-outcome.json")
    report = read(prefix + "extended-discovery-006/host-confirmation/confirmation-report.json")
    protocol006 = read(prefix + "extended-discovery-006/artifacts/protocol.json")
    controls = read(prefix + "extended-discovery-006/causal-and-transfer-interim-analysis.json")
    mcp = read(prefix + "independent-mcp-014/study-report.json")
    protocol014 = read(prefix + "independent-mcp-014/protocol.json")
    feedback = read(prefix + "independent-mcp-014/feedback-flow-audit.json")
    mathematics = []
    for case, s, original, edited, accepted in [
        ("extended-discovery-006", report["statistics"], outcome["independent"]["original_correct"], outcome["independent"]["edited_correct"], outcome["verification"]["native_host_final_review_approved"]),
        ("independent-mcp-009-014", mcp["independent_confirmation"], mcp["independent_confirmation"]["original_correct"], mcp["independent_confirmation"]["edited_correct"], mcp["all_predeclared_acceptance_gates_passed"]),
    ]:
        assert edited - original == s["wrong_to_right"] - s["right_to_wrong"]
        assert abs((edited - original) / s["items"] - s["gain"]) < 1e-12
        mathematics.append({
            "case": case, "items": s["items"], "original_correct": original, "edited_correct": edited,
            "original_percent": 100 * original / s["items"], "edited_percent": 100 * edited / s["items"],
            "gain_pp": 100 * s["gain"], "gain_interval_pp": [100 * v for v in s["paired_bootstrap_interval"]],
            "confidence": s["confidence"], "wrong_to_right": s["wrong_to_right"], "right_to_wrong": s["right_to_wrong"],
            "original_correct_lost_percent": 100 * s["right_to_wrong"] / original,
            "adoption_passed": accepted,
        })
    rows = report["rows"]
    assert len(rows) == report["statistics"]["items"]
    row_counts = {}
    for label in ["original", "edited"]:
        row_counts[label] = {
            "correct": sum(r[label + "_grade"] is True for r in rows),
            "unparsed": sum(r[label + "_grade"] is None for r in rows),
            "cutoff": sum(r[label + "_finish"] == "max_new_tokens" for r in rows),
        }
        assert row_counts[label]["unparsed"] == report["statistics"][label + "_unparsed"]
    assert row_counts["original"]["correct"] == mathematics[0]["original_correct"]
    assert row_counts["edited"]["correct"] == mathematics[0]["edited_correct"]
    rescued_unparsed = sum(r["original_grade"] is None and r["edited_grade"] is True for r in rows)
    assert rescued_unparsed == outcome["independent"]["rescued_unparsed"] == 86
    assert outcome["independent"]["rescued_parsed_wrong"] == 0
    assert feedback["verified_feedback_dependent_revisions"] == mcp["feedback_revision_count"] == 1
    return {
        "scope": "Retrospective extraction from published JSON; intervals copied, not newly estimated; no model replay.",
        "evidence_snapshot_commit": SNAPSHOT,
        "sources": sources,
        "direct_values_001": {
            "validation_examples": direct["baseline"]["validation"]["examples"],
            "original_validation_nll": direct["baseline"]["validation"]["nll"],
            "candidates": [{"id": c["candidate"], "changed_values": c["changed_parameters"], "validation_nll": c["cpu_metrics"]["validation"]["nll"], "manual_patch_exact": c["manual_patch_exact"], "loaded_values_exact": c["actual_parameter_exact"]} for c in direct["candidates"]],
            "selected_candidate": direct_decision["selected_candidate"], "heldout_evaluated": direct["heldout_evaluated"],
        },
        "repetition": repetitions,
        "mathematics": mathematics,
        "qwen006_details": {
            "row_counts": row_counts, "rescued_unparsed": rescued_unparsed,
            "parameters": outcome["parameter_changes"], "accounting": outcome["accounting"],
            "decoding": protocol006["decoding"], "confirmation_policy": protocol006["independent_test"],
            "control_limitations": controls["limitations"],
        },
        "mcp009_014_details": {
            "validation": [{k: v[k] for k in ["original_correct", "edited_correct", "items", "conditional_old_correct_loss"]} for v in mcp["validation"]],
            "stored_changes": mcp["candidates"][-1]["report"]["changed_parameters"],
            "gpu_bf16_changes": mcp["validation"][-1]["actual_changed_GPU"],
            "feedback_revisions": feedback["verified_feedback_dependent_revisions"],
            "successful_development_jobs": mcp["development_successful_batches"],
            "independent_paired_jobs": mcp["independent_successful_paired_batches"],
            "reference_preservation_passed": mcp["reference_preservation"]["passed"],
            "min_accuracy_gain": protocol014["min_accuracy_gain"],
            "old_correct_loss_limit": protocol014["max_conditional_old_correct_loss"],
            "bootstrap_draws": protocol014["bootstrap_draws"], "bootstrap_seed": protocol014["bootstrap_seed"],
        },
    }


def figures(evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    panels = [
        (evidence["repetition"], ["TinyStories 004", "TinyStories 005"], "Repetition reduction (pp; 95% intervals)"),
        (evidence["mathematics"], ["Source-informed Qwen 006", "MCP-only Qwen 009–014"], "Answer-accuracy gain (pp; 97.5% intervals)"),
    ]
    for ax, (rows, labels, title) in zip(axes, panels):
        for i, row in enumerate(rows):
            v, (lo, hi) = row["gain_pp"], row["gain_interval_pp"]
            ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt="o", capsize=5, color="#3158D5", markersize=7)
            assert not row["adoption_passed"]
        ax.axvline(0, color="#68717A", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_ylim(1.6, -0.6)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Recorded fixed-baseline effects: no candidate passed every adoption criterion", fontsize=12)
    fig.savefig(PAPER / "figures/recorded-effects.svg")
    fig.savefig(PAPER / "figures/recorded-effects.png", dpi=180)
    fig.savefig(PAPER / "figures/recorded-effects.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    data = extract()
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target = PAPER / "evidence.json"
    if args.check:
        if target.read_text() != encoded:
            raise SystemExit("evidence.json differs from published source bytes; inspect before rebuilding.")
        print(f"Verified manuscript extraction against {len(data['sources'])} published JSON sources; no model execution.")
    else:
        target.write_text(encoded)
        print(f"Wrote {target.relative_to(ROOT)} from {len(data['sources'])} published JSON sources.")
    if args.figures:
        (PAPER / "figures").mkdir(exist_ok=True)
        figures(data)
        print("Wrote static SVG/PDF figures using retained historical intervals.")


if __name__ == "__main__":
    main()
