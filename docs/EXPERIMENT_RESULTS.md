> 역사 기록: 아래 optimizer/gradient 및 fixture 훈련 결과는 현재 사용자의 무경사 직접 편집 요청을 충족하지 않는다. 실행하지 않으며 성공 근거에서 제외한다. 현재 계약은 [INFERENCE_ONLY_CONTRACT.md](INFERENCE_ONLY_CONTRACT.md)다.

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

# 2026-11-04 직접 가중치 편집 실험

도구는 실제 모델 가중치를 바꾸고 forward 결과를 측정한다. **현재 사전학습 모델의 확립된 성능 개선은 없다.** 합성 모델 복구 성공, 사전학습 무경사 편집 실패, gradient 보조 편집의 불확실한 평균 개선을 모두 보존했다. 원본 체크포인트는 교체하지 않았다.

## 결과

gain은 `원본 NLL - 후보 NLL`이며 단위는 nats/token이다. 양수가 손실 감소다. 생성 benchmark 점수가 아니다.

| 실험 | 선택/확인 결과 | 판단 |
|---|---|---|
| 합성 작은 GPT-2, 의도적으로 손상한 MLP의 직접 감쇠 | validation gain 0.01919; test gain 0.01584, 95% bootstrap 구간 [0.01212, 0.01943] | 정해진 gate 통과. 합성 손상 복구에 한정 |
| 손상시키지 않은 TinyStories, Claude의 무경사 후보 12개 | 모든 validation NLL 악화; 가장 작은 악화도 0.00395 | 승자 없음. 최종 데이터 추론 안 함 |
| 같은 TinyStories, discovery gradient 보조 rank-one 후보 6개 | validation으로 선택 후 test NLL 2.97453 → 2.95155; gain 0.02298, 구간 [-0.00591, 0.05091] | 구간이 0을 포함해 gate 실패, 채택 안 함 |

## 합성 복구

`examples/tiny_lm_demo.py`는 seeded GPT-2 형태 모델을 작은 합성 전이 과제에 학습한 뒤 일부 MLP 뉴런의 출력을 의도적으로 증폭시켜 source를 만든다. **fixture 학습은 AdamW**, 이후 편집은 optimizer 없이 명시적 `scale` 연산이다. discovery에서 개입할 행을 찾고 validation으로 strength 0.5를 선택했다. 선택과 대조군을 고정한 뒤 test를 평가했다.

512개 가중치를 수정했다. test 192예제 중 47예제의 NLL은 악화했으나 전체 token-weighted NLL과 사전 정의 gate는 개선했다. 동일 원소 수·delta norm을 맞춘 무작위 뉴런 대조군의 test gain은 0.00607로 기준 0.01에 못 미쳤고, 반대 선택은 -0.00765였다. 무수정은 정확히 0이었다. 따라서 모든 예제가 좋아졌다고 해석하지 않는다.

원본·후보·고정 선택·대조군·데이터는 runs/tiny-lm-recovery (`runs/tiny-lm-recovery/`, 로컬 전용)에 있다. 가벼운 결과 사본은 [tiny-lm-recovery.json](evidence/tiny-lm-recovery.json)이다. 이 예제의 개입 위치 선택은 정해진 무경사 탐색 알고리즘이며, LLM이 실시간으로 각 행을 추론해 선택한 실험으로 부르지 않는다.

## 사전학습 무경사 편집

[TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M)의 revision `77f1b168e219585646439073245fe87e56b3023e`을 로컬 FP32 safetensors로 변환했다. 총 3,745,984 parameter, width 64, 8 Transformer block이며 기존 가중치를 손상하거나 학습하지 않았다. 모델 파일 SHA-256은 `fdb05ba2a15f0439691a37e2708b1a7f7560ee55b885b41a2fe65edb0c0c71d1`이다.

Claude Sonnet 5.5는 실제 discovery forward hook으로 MLP neuron·attention head의 기여 추정치와 평균 출력 방향을 관찰했다. Linear의 실제 `[out,in]` 구조와 tied embedding/head를 확인한 뒤 국소 감쇠/방향 제거 plan 12개를 작성하고 추론했다. 큰 기여가 불필요한 기여라는 가설은 validation에서 지지되지 않았다. 두 후보는 discovery 손실을 줄였지만 validation에서는 악화했다.

discovery와 validation은 각각 12개 직접 작성한 짧은 영어 이야기, 319·306 target token이다. baseline NLL은 각각 2.45486·2.92543이다. 후보 기준은 gain>0.0001, accuracy 하락≤0.01이었다. 모두 실패했으므로 승자를 고정하지 않았고 final 데이터는 이 실험에 제공하지 않았다. 호스트가 12개를 별도 코드로 다시 적용·추론해 모든 validation gain이 1e-8 이내로 재현됨을 확인했다.

실제 LLM의 plan과 관측은 [selection.json](evidence/tinystories-forward-only.json), 독립 재계산은 [recheck](evidence/tinystories-forward-only-recheck.json), 전체 로컬 자료는 runs/tinystories-forward-only (`runs/tinystories-forward-only/`, 로컬 전용)에 있다. 이 디렉터리의 `experiment.py --check`는 보조 재현 스크립트이며 독립 검증기로 취급하지 않는다.

## gradient 보조 비교

이 방식은 **무경사 LLM 편집 성공이 아니다.** 호스트 Codex가 마지막 두 MLP 출력 행렬을 대상으로 discovery NLL gradient의 첫 SVD 성분을 음의 방향으로 적용하는 비교 방식을 선택했다. 각각 상대 tensor norm 0.001·0.003·0.01의 rank-one plan, 총 6개를 validation에서 비교했다. optimizer state나 반복 학습은 없지만 업데이트 방향이 gradient에서 계산되므로 gradient 기반 적응이다. [propose_rank_one.py](../archive/gradient_experiments/propose_rank_one.py.txt)가 이 선택 가능한 비교 방식을 재현한다.

validation 기준으로 layer 6, 상대 norm 0.01을 고정했다. 실제 FP32 변경 원소는 16,383개(전체의 약 0.43735%)다. 선택 패치와 identity·같은 크기의 무작위 rank-one·반대 부호 대조군의 해시 및 기준을 먼저 고정한 뒤, 별도 12이야기·336 target token의 test를 확인했다. 최종 데이터는 이전 무경사 실험의 후보 선택에 쓰지 않았다.

평균 NLL은 줄었지만 12개 중 5개 이야기의 NLL은 악화했고 정확도는 38.095% → 37.798%로 소폭 내려갔다. 무작위 대조 gain은 -0.00113, 반대 부호 대조는 -0.23185였다. 선택 후보의 bootstrap 구간이 0을 포함하므로 gate는 실패했다. 다른 후보로 재선택하거나 기준을 낮추지 않았다. 작은 수작업 데이터에서 얻은 결과이며 통계적 일반화나 데이터 오염 부재를 입증하지 않는다.

[최종 비교](evidence/tinystories-gradient-assisted-summary.json), [선택 기록](evidence/tinystories-gradient-assisted-selection.json), [고정 해시](evidence/tinystories-gradient-assisted-frozen.json), 전체 로컬 자료 (`runs/tinystories-gradient-assisted/`, 로컬 전용)를 보존했다. 생성 반복·수학 문제·한국어 능력 개선은 평가하지 않았다.

## 구현과 검증의 구분

loop-engineering을 통해 Claude Sonnet 5.5가 평가기 초안과 위 무경사 실험을, Codex GPT-6.1-Sol이 진단·rank-one·CLI/MCP·실험 runner·스킬을 담당했다. 첫 Claude 평가기 초안은 테스트의 비종료 fixture와 너무 이른 test 평가가 발견되어 중단했다. 호스트가 이를 수정하고 실제 가중치 적용·logits 변화·패치 변조 거부를 추가 검증했다. 최초 중단 작업은 성공으로 계산하지 않는다.

Codex 격리 작업의 기존 core 보존 검사는 native service에서 통과했다. 새 기능의 통합 테스트와 위 실험 재계산은 호스트가 별도로 실행했다. native review 승인은 호스트 판단이며 모든 요건의 기계 검증을 의미하지 않는다. 사용 모델·호출 기록은 [loop-jobs.json](evidence/loop-jobs.json)에 있다. 두 모델의 품질 우열을 비교하는 실험은 아니다.

최종 결합 프로젝트에서 `.venv/bin/python -B -m pytest -q -p no:cacheprovider`를 실행해 **42개 통과**했다. 실제 CLI subprocess, stdio MCP, forward activation, autograd 민감도, 양쪽 저장 축의 rank-one, patch 변조 거부, 파라미터·logits 변경, 원본 보존, test 이전 선택 고정과 회귀 거부를 포함한다. 네 스킬의 형식 검증도 통과했다.

배포한 `examples/propose_rank_one.py`를 실행해 보존된 6개 gradient 보조 패치 값을 정확히 재현했다. 공개 CLI `run-experiment`로 같은 고정 후보·기준을 재실행한 integration replay도 index 2를 선택하고 confirmation 1회 후 `gate_failed`로 거부했다. 이 재실행은 이미 test 결과를 본 뒤의 재현 검증이며 새로운 독립 실험으로 계산하지 않는다. 기록은 `runs/tinystories-gradient-assisted/runner-replay/`에 있다.
