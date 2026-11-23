"""Render README figures from retained JSON only; no inference or model editing.

Optional dependency: matplotlib. Run with python3 scripts/build_readme_figures.py.
Historical gradient-training JSON is retained as a separate archival record,
excluded from method comparisons and never run. The comparison figure shows
missing evidence, not invented experimental results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/results"
INK, BLUE, GRAY, CORAL = "#182638", "#3158D5", "#AAB5C1", "#C14E36"
BINDINGS = []


def read(relative: str):
    data = (ROOT / relative).read_bytes()
    BINDINGS.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
    return json.loads(data)


def collect():
    outcome = read("experiments/extended-discovery-006/study-outcome.json")
    protocol = read("experiments/extended-discovery-006/artifacts/protocol.json")
    audit = read("experiments/direct-values-inference-only-001/host-audit.json")
    archive = read("docs/evidence/mcp-training-verification.json")["cases"]["pretrained"]
    training = archive["training"]
    assert sum(r["original"] for r in outcome["independent"]["by_source_correct"].values()) == outcome["independent"]["original_correct"]
    assert sum(r["edited"] for r in outcome["independent"]["by_source_correct"].values()) == outcome["independent"]["edited_correct"]
    return {
        "scope": "Retrospective figure extraction only; no new inference, training or selection.",
        "qwen006": {
            "independent": outcome["independent"],
            "source_informed": outcome["parameter_changes"]["source_informed_host_branch"],
            "stored_changed_values": outcome["parameter_changes"]["stored_changed_values"],
            "actual_bf16_changed_values": outcome["parameter_changes"]["actual_gpu_bf16_changed_values"],
            "stored_changed_fraction": outcome["parameter_changes"]["stored_changed_fraction"],
            "preservation_limit": protocol["independent_test"]["max_previously_correct_to_wrong_fraction"],
            "overall_status": outcome["overall_status"],
        },
        "archival_trainability_check": {
            "source_sha256": archive["source_sha256"],
            "eligible_for_method_comparison": False,
            "reason": "Only 12 training stories / 319 target tokens; untuned post-patch trainability check.",
            "before": training["before"], "after": training["after"],
            "optimizer_steps": training["steps"],
            "optimizer_updated_entries": training["optimizer_changed_parameters"],
            "preceding_rank_one_entries": training["mcp_edited_cells"],
        },
        "direct_value_checks": {
            "before": audit["baseline"],
            "candidates": [{"id": c["candidate"], "after": c["cpu_metrics"],
                            "stored_changed_entries": c["changed_parameters"]}
                           for c in audit["candidates"]],
        },
        "finetuning_comparison": {
            "status": "not_measured",
            "adequate_data_finetuning": "not_measured",
            "common_test_and_retention": "not_measured",
            "end_to_end_resources": "not_measured",
            "design": "docs/FINETUNING_COMPARISON.md",
        },
        "sources": BINDINGS,
    }


def axis_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E7EA", linewidth=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D2D9DF")
    ax.tick_params(axis="both", length=0, pad=10, labelsize=11, colors=INK)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=150, facecolor="white", metadata={"Software": "ForwardEdit retained-evidence figures"})
    fig.savefig(OUT / f"{stem}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def qwen_figure(data, lang):
    ko = lang == "ko"
    q = data["qwen006"]
    result = q["independent"]
    stats = result["statistics"]
    sources = list(result["by_source_correct"].values())
    rows = sources + [{"items": result["items"], "original": result["original_correct"], "edited": result["edited_correct"]}]
    fig = plt.figure(figsize=(13.2, 6.8))
    ax = fig.add_axes([0.075, 0.25, 0.59, 0.53])
    fig.text(0.075, 0.93, "Qwen 006 · 기록된 수학 정답률 상승" if ko else "Qwen 006 · Recorded math-accuracy gain", fontsize=21, color=INK, weight="bold")
    fig.text(0.075, 0.865, "원문 참고 개입 · 고정 원본 대비 · 독립 평가의 선정 부분집합" if ko else "Source-informed intervention · fixed original · selected independent subsets", fontsize=12, color="#596879")
    x = [0, 1, 2]
    before = [r["original"] / r["items"] * 100 for r in rows]
    after = [r["edited"] / r["items"] * 100 for r in rows]
    ax.bar([v - .19 for v in x], before, .34, color=GRAY, label="원본" if ko else "Original")
    ax.bar([v + .19 for v in x], after, .34, color=BLUE, label="LLM 편집" if ko else "LLM-edited")
    for i, r in enumerate(rows):
        for shift, value, count in [(-.19, before[i], r["original"]), (.19, after[i], r["edited"])]:
            ax.text(i + shift, value + 2.5, f"{count}/{r['items']}\n{value:.2f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("정답률 (%)" if ko else "Accuracy (%)", fontsize=12, color=INK)
    ax.set_xticks(x, ["GSM8K\n128 items", "MATH-500\n64 items", "합계\n192문항" if ko else "Combined\n192 items"])
    axis_style(ax)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=11)
    fig.text(.72, .70, f"+{stats['gain'] * 100:.2f} pp", fontsize=29, weight="bold", color=BLUE)
    lo, hi = stats["paired_bootstrap_interval"]
    fig.text(.72, .62, f"97.5% paired interval\n[{lo * 100:+.2f}, {hi * 100:+.2f}] pp", fontsize=12, color=INK, linespacing=1.8)
    fig.text(.72, .46, "사전 규칙 미충족" if ko else "FIXED RULE MISSED", fontsize=17, weight="bold", color=CORAL)
    fig.text(.72, .34, ("새 정답 86개 · 손실 1개 · 순증 85개\n기존 정답 17/18개 유지\n18개의 5%는 사실상 손실 0개 허용" if ko else "86 rescued; 1 lost; net +85 correct\n17/18 old-correct answers retained\n5% of 18 permits zero losses"), fontsize=11, color=INK, linespacing=1.8)
    footer = ("전체 벤치마크 점수가 아닙니다. 최강 무편집 설정·파인튜닝과의 대조는 측정하지 않았습니다.\n86개 오답→정답 전환은 모두 이전에 파싱되지 않은 답변에서 나왔습니다. 새 수학 지식 습득을 입증하지 않습니다."
              if ko else "Not full-benchmark scores. Strongest unedited settings and a matched fine-tuning baseline were not measured.\nAll 86 wrong-to-right transitions started from unparsed answers; new mathematical knowledge is not established.")
    fig.text(.075, .10, footer, fontsize=10, color="#596879", linespacing=1.8)
    fig.text(.075, .025, "Source: experiments/extended-discovery-006/study-outcome.json + frozen protocol", fontsize=9, color="#7C8793")
    save(fig, f"qwen-006.{lang}")


def comparison_figure(data, lang):
    ko = lang == "ko"
    assert data["finetuning_comparison"]["status"] == "not_measured"
    fig = plt.figure(figsize=(13.2, 7.2))
    fig.text(.075, .92, "파인튜닝 비교 · 아직 필요한 근거" if ko else "Fine-tuning comparison · Evidence still needed", fontsize=21, weight="bold", color=INK)
    fig.text(.075, .84, "12문장 학습 동작 검사는 방법 비교에서 제외" if ko else "The 12-story trainability check is excluded from method comparisons", fontsize=13, color=CORAL)
    ax = fig.add_axes([.075, .31, .9, .43])
    ax.axis("off")
    rows = ([
        ["충분한 과제 데이터로 검증·선택한 파인튜닝", "미측정"],
        ["같은 원본·공통 test·목표 및 비대상 보존", "미측정"],
        ["데이터·조정·추론·학습을 포함한 전체 비용", "미측정"],
    ] if ko else [
        ["Fine-tuning on substantial task data, selected on validation", "Not measured"],
        ["Same original, common test and target/unrelated retention", "Not measured"],
        ["Total data, tuning, inference and training resources", "Not measured"],
    ])
    table = ax.table(cellText=rows, colLabels=["비교 요건" if ko else "Comparison requirement", "상태" if ko else "Status"],
                     colWidths=[.77, .23], cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#E8EDF3" if row == 0 else "#F5F7F9")
        cell.set_text_props(color=CORAL if row > 0 and col == 1 else INK,
                            weight="bold" if row == 0 else "normal")
    footer = ("공통 데이터의 방법 비교와 더 큰 학습 자료의 실용 비교를 각각 보고합니다.\n현재는 설계이며 실험 결과가 아닙니다. 성능·보존·효율의 우위는 아직 입증되지 않았습니다."
              if ko else "Report a common-data comparison and a larger-data practical comparison separately.\nThis is a design, not experimental results. Quality, retention and efficiency superiority remain unestablished.")
    fig.text(.075, .18, footer, fontsize=11, color="#596879", linespacing=1.8)
    fig.text(.075, .055, "Design: docs/FINETUNING_COMPARISON.md · Retained record: docs/evidence/mcp-training-verification.json", fontsize=9, color="#7C8793")
    save(fig, f"finetuning-comparison-plan.{lang}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_dir = ROOT / "videos/forwardedit-explainer/public/fonts"
    for filename in ["NanumSquare-Regular.ttf", "NanumSquare-Bold.ttf"]:
        font_manager.fontManager.addfont(font_dir / filename)
    family = font_manager.FontProperties(fname=font_dir / "NanumSquare-Regular.ttf").get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "svg.hashsalt": "forwardedit-readme-evidence"})
    data = collect()
    (OUT / "figure-data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    # The video reads the same selected metric snapshot as the static figures.
    (ROOT / "videos/forwardedit-explainer/src/recorded-evidence.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    for locale in ["en", "ko"]:
        qwen_figure(data, locale)
        comparison_figure(data, locale)
    print("Rendered 4 bilingual figures (PNG + SVG); source hashes: docs/assets/results/figure-data.json")


if __name__ == "__main__":
    main()
