# 정상 예측 손상 분석과 피드백 기반 편집 반복

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

2026-11-04. TinyStories-1M에서 **후속 편집 4회와 대조 편집 1회**를 실제 수행했다. 마지막 편집은 이전 후보의 정상 예측 손상을 줄이면서 일부 반복 감소를 유지했지만, 고정된 최소 개선 기준을 충족하지 못했다. 판정은 **no_gain**, 원본 유지다. 독립 데이터에서 학습 성공을 확인한 결과는 아니다.

## 무엇을 반복했나

Codex 호스트와 loop-engineering의 Claude Opus 5.5가 협업했다. Claude 실행은 2회였고, 편집 반복 횟수와 구분한다. [이전 실험](SMALL_MODEL_REPETITION.md)의 반복 감소 후보를 분석 대상으로 삼아, 실제 추론 결과를 읽고 다음 가설을 바꾸었다. 첫 반복의 결과를 보기 전에 다음 후보들을 일괄 생성하지 않았다.

이전 후보는 정상 reference 16개 중 14개에서 NLL이 악화했다. 위치별 관측에서는 작은 layer-1 출력 변화가 큰 최종 token log-probability 변화와 함께 나타났다. 같은 위치의 긴 정상 글을 대조하자, 제거했던 방향과 정상 글의 초반/후반 평균 입력 차이의 cosine이 -0.964였다. 이는 방향의 정렬이지 ‘96% 위치 코드’나 의미 역할의 증명이 아니다. 긴 정상 대조는 discovery reference들을 이어 붙인 글이어서 분포 차이가 남는다.

피드백에 따라 다음 편집을 수행했다.

1. 위치 차이와 직교하는 실패/정상 대비 방향으로 대체했다. 정상 NLL은 보존됐지만 반복 억제 효과가 사라졌다.
2. 이전 후보에서 초반 입력 방향을 보호했다. 반복은 줄지 않고 정상 NLL도 악화해 이 설명만으로는 부족했다.
3. 초반 방향만 제거했다. 정상 NLL은 거의 유지됐지만 반복은 증가했다.
4. 상대 부호를 바꾼 결합 방향을 대조한 뒤, 원래 수정의 교차 성분만 유지하는 마지막 편집을 했다.

각 새 후보의 선언된 parent는 이전 실험의 seed였다. 실패한 후보를 원복하고 다른 부분을 수정한 **피드백 기반 재편집**이며, 성공한 변경을 4단계 연속 누적한 체인은 아니다. 원본 checkpoint는 고정하고 parent 대비 증분과 원본 대비 최종 절대값을 함께 보존했다. 이력 (`runs/edit-iteration-002/iterations.json`, 로컬 전용)의 첫 4개 기록과 기존 패치들은 후속 보완에서도 정확히 유지됐고, 수정 가설과 보완 관측은 추가 기록했다.

## 최종 수정값의 근거

대상은 `transformer.h.1.attn.attention.out_proj.weight`의 행 0..63, 열 32..39, 총 512개 값이다. 실제 저장 parameter 3,745,984개의 약 0.01367%다. 모든 새 후보가 이 block을 수정했고, 다른 값은 유지했다.

원래 방향 r을 관측된 초반 입력 방향 e와 그 직교 방향 v로 분해하면 `r = a e + b v`, `a≈0.72818`, `b≈0.68538`이다. 원래 수정 `-Wrrᵀ`에는 두 자기 성분과 교차 성분이 함께 들어간다. 별도 성분을 제거한 앞선 실험만으로 교차 성분의 효과를 확정할 수 없었다. 마지막 후보는 자기 성분을 복원하고 다음 교차항만 남겼다.

```text
W_final = W_source - ab [(W_source e)vᵀ + (W_source v)eᵀ]
```

계수는 원래 방향의 분해에서 고정했다. 점수를 보며 strength/angle을 탐색하지 않았다. 최종 tensor 상대 L2 변화는 약 0.08718이다. 수정 전 분석 (`runs/edit-iteration-002/analysis-10.json`, 로컬 전용), 근거 (`runs/edit-iteration-002/basis-01.json`, 로컬 전용), 제안 검토 (`runs/edit-iteration-002/proposal-01.json`, 로컬 전용)를 먼저 만들었고, 실제 판정은 ready_for_causal_probe였다. 이는 국소 개입의 준비 상태이며 인과/성능 인증은 아니다.

## 동일한 validation 16개에서의 결과

반복률은 EOS 이전 생성 토큰의 중복 8-gram 비율을 문장별 평균한 값이다. NLL은 정상 reference 문장의 예측 손실로 낮을수록 좋다. 최대 128개 새 토큰, CUDA greedy, EOS 종료와 cache 설정을 고정했다.

| 단계 | 반복률 ↓ | reference NLL ↓ | reference accuracy ↑ |
|---|---:|---:|---:|
| 원본 | 10.690% | 3.73125 | 29.508% |
| 이전 seed | 3.883% | 3.77519 | 27.541% |
| 편집 1: 위치 차이 제외 | 10.574% | 3.72748 | 29.672% |
| 편집 2: 초반 방향 보호 | 10.553% | 3.76850 | 28.361% |
| 편집 3: 초반 방향만 제거 | 11.331% | 3.73219 | 29.672% |
| 대조: 상대 부호 반전 | 12.561% | 3.73536 | 29.836% |
| 편집 4: 교차항 유지 | 6.523% | 3.75416 | 28.689% |

마지막 편집은 원본 대비 반복률 **4.167%p 감소**, NLL **+0.02291**, accuracy **-0.820%p**였다. 정상 예측 보존 한도 NLL +0.03, accuracy -2%p를 통과했다. 중앙 생성 길이는 128, 짧은 출력 비율은 0이었다. 그러나 최소 반복 감소 **5%p**에 못 미쳤고, validation paired bootstrap 95% 구간 **[-0.364, 9.750]%p**에도 0이 포함됐다. 4개 문장에서는 반복이 증가했다.

이는 손상과 반복 감소의 절충을 개선한 개발 관측이다. 같은 validation을 분석과 수정에 반복 사용했으므로 독립 성능 확인으로 해석하지 않는다. 모든 후보가 고정 선택 기준을 실패해 선택은 null로 고정했다. 봉인된 test 32개는 열거나 평가하지 않았다. 기준을 결과에 맞춰 낮추거나 test로 후보를 재선택하지 않았다.

## 실제 parameter 흐름을 확인한 범위

MCP로 같은 입력·위치에서 후보 5개를 모두 다시 관측했다. source와 실제 patch에서 계산한 출력 변화가 실제 module 출력 차이와 **최대 3.25×10⁻⁸ 미만 오차**로 일치했다. 최종 후보도 사전 예측과 약 3.24×10⁻⁸ 오차였다. 가중치 적용과 국소 affine 계산은 확인됐다. 국소 변화량만으로 정상 token 확률 손상을 예측할 수는 없었다.

원본/seed를 layer-7 attention 출력에서도 비교했다. 평균 변화는 정상 위치 약 0.0079, 실패 위치 약 0.0074였고, 원래 출력 norm 대비 약 6%였다. 이는 해당 downstream 모듈까지 변화가 전달됐다는 관측이다. 어떤 위치·경로가 NLL 손상이나 반복 감소를 일으켰는지는 식별하지 못했다. 최종 후보의 downstream 경로 전체를 관측한 것도 아니다.

초기 편집 3회와 대조군은 표준 proposal 도구를 생략하고 사전 이력에 근거를 기록했다. 첫 검토에서 그 누락과 수정 후 내부 출력 관측 누락을 발견했다. 보완 단계에서 5개 후보의 실제 내부 출력을 관측하고, 마지막 편집은 표준 proposal 절차로 진행했다. 이 사후 보완을 초기 편집 전에 수행한 검증으로 표현하지 않는다.

독립 감사 (`runs/edit-iteration-002/host-audit.json`, 로컬 전용)는 source/data/protocol hash, 모든 patch 재계산, parent/child 값, 기록 순서와 validation 지표, 실제 국소 출력을 확인한다. 실제 MCP의 최종 parameter 비교 (`runs/edit-iteration-002/mcp-final-lineage.json`, 로컬 전용)에서도 512개 재수정값과 hash가 확인됐고 모델 forward는 0회였다. 감사 재실행은 모델 추론이나 test 읽기를 하지 않는다.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B runs/edit-iteration-002/audit_iterations.py runs/edit-iteration-002 runs/edit-iteration-002/host-audit.json
```

학습·gradient·optimizer·numerical gradient·자동 loss fitting은 실행하지 않았다. 모델 실행은 training tripwire를 둔 MCP frozen inference를 사용했다. 총 32개 도구 요청 중 추론 도구 요청은 26개이고 나머지는 CPU 후보 생성·제안 검토였다. 한 도구 요청 안에는 여러 실제 forward가 있으므로 이를 26번의 단일 forward로 해석하지 않는다. 새 후보 예산 5개와 native 실행 2회를 사용하고 종료했다.

## 도구 검증과 루프 승인 상태

원본/parent/child의 실제 수치를 비교하는 `compare_candidate_parameters` MCP 도구를 추가했다. 유지·재수정·원복·신규 변경을 구분하며 학습이나 모델 실행은 하지 않는다. 진단/평가 스킬에도 피드백 이력과 후보 고정 조건을 반영했다. 호스트 프로젝트 전체 테스트는 **80 passed**, 실제 stdio parameter 비교를 보강한 관련 테스트는 **3 passed**였다. worker 복사본에는 tests가 없어 그곳의 pytest는 테스트를 수집하지 못했으며 통과 증거로 사용하지 않는다.

호스트가 수치를 재계산한 것과 native 승인 상태는 구분한다. 최종 native 기록 (`runs/edit-iteration-002/final-loop-result.json`, 로컬 전용)은 **failed, verified:false**다. retained finding을 닫으려면 현재 실행 기반 execution_refs가 필요한데, 최초 루프에는 native checks가 선언되지 않았고 2회 예산도 종료됐다. 자동 승인 검증이 승인을 거절했으며 영수증을 만들거나 예산·기준을 바꿔 우회하지 않았다. 이는 호스트 감사 수치를 지우는 것도, 학습 성공을 의미하는 것도 아니다.

raw worker decision에는 과거의 ‘downstream 미관측’ 제한 문장이 남아 있으나, 같은 파일의 새 verification 절과 analysis-08/09가 실제 seed 관측을 기록한다. 호스트 검토 메모 (`runs/edit-iteration-002/host-review-notes.json`, 로컬 전용)에 이 불일치를 명시했고 원본 worker 파일은 유지했다. 최종 선택 기록 (`runs/edit-iteration-002/frozen-choice.json`, 로컬 전용)은 no_gain, selected_candidate=null, heldout_accessed=false다.
