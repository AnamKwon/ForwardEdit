# Experiment names / 실험 이름표

Readable names for internal experiment identifiers. Papers and reports use the readable names; the original identifier is kept here (and in the manuscript's "Experiment identifiers" appendix) so every result stays traceable to the host evidence ledger, pre-registrations and result files.

내부 실험 ID를 읽기 쉬운 이름으로 바꾼 표입니다. 논문과 리포트는 읽기 쉬운 이름을 쓰고, 원래 ID는 이 표(와 논문의 "Experiment identifiers" 부록)에 남겨 호스트 근거 장부, 사전 등록, 결과 파일과 연결합니다.

**Scheme / 규칙.** `<model size> <task family> <distinguishing variant>`. "Fact-Edit / 사실 편집" is a CounterFact rewrite of 40 facts. "Blind / 블라인드" means no earlier-session recipe notes in the editor prompt. "A/B" are two recipe-informed runs on different fact sets; numbered blind runs are successive attempts.

## Current study (manuscript v2, reports of 2026-11-23/24)

| English name | 한국어 이름 | Original ID | Model | One-line description |
|---|---|---|---|---|
| 4B Fact-Edit A | 4B 사실 편집 A | `qcf15_4b` | Qwen3.5-4B | All-context CounterFact rewrite; editor prompt carried recipe notes from earlier sessions; sealed PRIMARY pass. |
| 4B Fact-Edit B | 4B 사실 편집 B | `qcf16_4b` | Qwen3.5-4B | Replication of Fact-Edit A on 40 new facts, same recipe-note prompt; adds the "override" prompt route. |
| 4B Fact-Edit (no-context design) | 4B 사실 편집 (무맥락 설계) | `qcf17_4b` | Qwen3.5-4B | Facts must hold with no context (development rows carry none); recipe notes kept; LoRA/ROME/MEMIT comparators run here. |
| 4B Blind Fact-Edit 1 | 4B 블라인드 사실 편집 1 | `qcf18_4b` | Qwen3.5-4B | All-context rewrite with recipe notes removed; found an L10 low-rank write; failed to freeze (preserve reserve exhausted). |
| 4B Blind Fact-Edit 2 | 4B 블라인드 사실 편집 2 | `qcf19_4b` | Qwen3.5-4B | Second blind attempt with the reserve guard; erasure writes at L6/L10/L14; failed to freeze (unexposed items consumed). |
| 9B Blind Fact-Edit | 9B 블라인드 사실 편집 | `qcf20_9b` | Qwen3.5-9B (nf4, edited linears bf16) | Blind all-context rewrite under the four-role split and dose cap; froze; failed the regression criterion. |
| 9B Fact-Edit, 40 rows (LLM edit-40) | LLM 편집-40 | `qcf21_9b` | Qwen3.5-9B (nf4) | Editor sees only the 40 new-fact rows (data-matched to LoRA-40); froze; sealed S .737, failed. |
| 9B Fact-Edit, 200 rows (LLM edit-200) | LLM 편집-200 | `qcf23_9b` | Qwen3.5-9B (nf4) | Editor sees the 200 LoRA-200 rows; did not freeze (GSM −5 over the 3-point limit). |
| 9B LoRA-40 / 200 / 620 | LoRA-40 / 200 / 620 | `qcf20_9b` (L-ft / L-ed / L-all) | Qwen3.5-9B (QLoRA nf4, r 8) | Host comparator on the same rows as the matching editor arm; sealed S .743 / .994 / .996. |
| 0.8B cap study | 0.8B 상한 연구 | `qcap_08b` | Qwen3.5-0.8B | Umbrella for the five blind four-role CounterFact runs below (cap ablation plus replicates); one shared sealed set. |
| 0.8B Fact-Edit (2% cap) | 0.8B 사실 편집 (2% 상한) | `qcap_on` | Qwen3.5-0.8B | Changed-parameter cap 2%; blind, all-context, four roles; sealed PRIMARY pass. |
| 0.8B Fact-Edit (no cap) | 0.8B 사실 편집 (상한 없음) | `qcap_off` | Qwen3.5-0.8B | Same as the 2% cap arm with the cap removed (the cap never bound); sealed FAIL. |
| 0.8B Fact-Edit (2% cap) replicate 1 | 0.8B 사실 편집 (2% 상한) 반복 1 | `qcap_rep1` | Qwen3.5-0.8B | Independent rerun of the 2% cap condition; sealed PRIMARY pass. |
| 0.8B Fact-Edit (2% cap) replicate 2 | 0.8B 사실 편집 (2% 상한) 반복 2 | `qcap_rep2` | Qwen3.5-0.8B | Rerun of the 2% cap condition; passed confirmation, sealed FAIL. |
| 0.8B Fact-Edit (2% cap) replicate 3 | 0.8B 사실 편집 (2% 상한) 반복 3 | `qcap_rep3` | Qwen3.5-0.8B | Rerun of the 2% cap condition; no freeze (confirmation GSM bound). |
| 4B Defect Scan (thinking off) | 4B 결함 조사 (thinking 끔) | `scan_4b_off` | Qwen3.5-4B | Greedy scan of a 1,100-item development mix, thinking off, 1,024 new tokens. |
| 4B Defect Scan (thinking on) | 4B 결함 조사 (thinking 켬) | `scan_4b_on8k` | Qwen3.5-4B | Same items, thinking on, 8,192 new tokens. |

## Earlier cross-task series (2026-11-08 to 11-22)

Names for identifiers that appear in older notes and summaries. These runs are background (manuscript Section 4.1), not part of the v2 evidence tables.

| English name | 한국어 이름 | Original ID | Model | One-line description |
|---|---|---|---|---|
| 0.8B Arithmetic (multi-error update) | 0.8B 산술 (다중 오류 업데이트) | `qarith3` | Qwen3.5-0.8B | 2–3-digit addition/subtraction, first arm allowed one-shot multi-error updates. |
| 0.8B Third-Word Letter | 0.8B 세 번째 단어 첫 글자 | `qword3` | Qwen3.5-0.8B | First letter of the third word of a four-word phrase (routing task). |
| 0.8B Third-Word Letter replicate 2 | 0.8B 세 번째 단어 첫 글자 반복 2 | `qword3_r2` | Qwen3.5-0.8B | Replication of the third-word task on a second sealed set. |
| 2B Third-Word Letter | 2B 세 번째 단어 첫 글자 | `qword3_2b` | Qwen3.5-2B | Same task, cross-model run on 2B. |
| 0.8B Last Letter | 0.8B 마지막 글자 | `qlast` | Qwen3.5-0.8B (also 2B, Qwen2.5-0.5B variants) | Last letter of a word; non-math rule skill. |
| 0.8B Irregular Past Tense | 0.8B 불규칙 과거형 | `qverb` | Qwen3.5-0.8B | Simple past of irregular verbs (lexical memory; no freeze). |
| 0.8B Format Constraints | 0.8B 형식 제약 | `qcons3` | Qwen3.5-0.8B | Nine IFEval-style verifiable format instructions. |
| 0.8B Format Constraints (no-comma only) | 0.8B 형식 제약 (쉼표 금지만) | `qcons_nc` | Qwen3.5-0.8B | Constraint task narrowed to the no-comma instruction. |
| 0.8B No-Comma Persona, round 1–3 | 0.8B 쉼표 금지 페르소나 1–3차 | `qnocomma` (r1, r2, r3) | Qwen3.5-0.8B | Bake a "never use commas" style into the weights; rounds differ in preserve sets and editor notes. |
| 0.8B Capital Overwrite B | 0.8B 수도 덮어쓰기 B | `qover2` | Qwen3.5-0.8B | Counterfactual overwrite of 30 known capitals; clean replication of the first overwrite run. |
| 0.8B Invented-Word Alias C | 0.8B 가짜 단어 별칭 C | `qalias3` | Qwen3.5-0.8B | Made-up word "glorp" means plural; third replication. |
| 0.8B Medical Domain | 0.8B 의학 도메인 | `qmed` | Qwen3.5-0.8B | Medical-domain knowledge, LLM edit vs host-only LoRA. |
| 2B BBH Dyck, round 1 | 2B BBH 괄호 닫기 1차 | `qdyck_2b` | Qwen3.5-2B | BBH dyck_languages (close the bracket sequence). |
| 2B BBH Dyck, round 2 | 2B BBH 괄호 닫기 2차 | `qdyck_r2_2b` | Qwen3.5-2B | Same task with synthetic discovery items; same sealed set as round 1. |
| 4B BBH Salient-Error Detection | 4B BBH 번역 오류 탐지 | `qsalient` (`qsalient_4b`) | Qwen3.5-4B | BBH salient_translation_error_detection; first 4B editor run. |
| 0.8B Thinking IFEval Loop Repair | 0.8B thinking IFEval 루프 수리 | `qthink` (`qthink_claude`) | Qwen3.5-0.8B | Rank-1 L12 down-proj edit that stops thinking loops; compared with budget forcing. |

Older identifiers not listed here keep their original ID until they are cited in a paper or report.
