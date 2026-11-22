# ForwardEdit research draft

The primary manuscript is in **English**. The Korean version follows the same argument, quantities and claim boundaries.

- [English manuscript](manuscript.md)
- [한국어 원고](manuscript.ko.md)
- [Claim-to-evidence map](evidence-map.md)
- [Extracted values and published-source hashes](evidence.json)
- [References (BibTeX)](references.bib)
- [Figure 1 (SVG)](figures/recorded-effects.svg) · [PDF](figures/recorded-effects.pdf) · [PNG](figures/recorded-effects.png)

**Working title:** *ForwardEdit: Auditing LLM-Directed Weight Interventions Without Gradient-Based Training*.

This is a retrospective systems and empirical-methodology draft, suitable for developing a workshop or technical-report submission. No venue suitability or acceptance is assumed. It analyzes five selected case groups from public records; it does not present a new benchmark run, accepted model improvement, general black-box solution or new editing algorithm.

The manuscript includes an abstract, motivation, related work, intervention equations, four levels of evidence, study design, results, a static figure, limitations, unaddressed criticisms and an explicitly unexecuted follow-up proposal. Exact application, behavior change, mechanism support and independent improvement are kept separate. All failed gates are retained.

## Reproduce the document's descriptive extraction

```bash
python3 scripts/verify_public_evidence.py
python3 paper/build_evidence.py --check
```

The checker uses Python's standard library. To rebuild the retained numbers and plot, use an environment with Matplotlib:

```bash
python3 paper/build_evidence.py --figures
```

The script reads 13 public JSON records and performs arithmetic checks. The intervals are copied from historical records, not newly estimated. It never executes embedded log instructions, opens model weights, runs inference, trains a model or reevaluates a test. The figures are standalone SVG/PDF/PNG artifacts; the manuscript itself is Markdown.

## Before submission

Author names, order, affiliations, intended venue and any venue-specific disclosures remain to be supplied. Named authors must review the claims, attribution and use of data excerpts. The source/model/data/runtime assets omitted from the public bundle limit independent replay. Strongest unedited baselines, broad preservation, controlled source-blind comparisons and more discriminating mechanism controls remain unaddressed; the manuscript does not invent those results.

영어 원고를 공개용 기준으로 작성했고, [한국어판](manuscript.ko.md)도 함께 제공합니다. 기존 공개 실험을 근거로 한 초안이며, 새 실험을 실행하거나 결과를 성공으로 재분류하지 않았습니다. 저자·소속·투고처는 아직 지정하지 않았습니다.
