# 작은 모델 직접 가중치 편집 검증 — 2026-11-04

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

TinyStories-1M의 실제 반복 생성을 진단하고 직접 weight를 수정했다. 반복은 감소했지만 정상 reference 문장의 예측 성능이 미리 정한 한도보다 악화해 **no_gain, 원본 유지**로 종료했다. 일반 능력 개선이나 gradient 없는 학습의 성공을 입증한 결과는 아니다.

## 실행과 증거

Codex 호스트가 도구·기준선·독립 재계산을 담당하고, loop-engineering (`runs/small-model-repetition-001/loop-result.json`, 로컬 전용)의 Claude Opus 5.5가 실제 관측을 읽어 가설과 수정값을 설계했다. job은 `job-426d87850c02d14e8db75cd2e1717ce6`이다. native 자동 체크는 없으며 `verified:false`다. 호스트가 증거를 검토해 부정적 실험 보고를 수용한 것으로, 모델 개선 승인이 아니다.

모델은 기존 pretrained checkpoint 그대로 시작했다. 실제 저장 parameter 수는 3,745,984개, FP32다. source SHA-256은 `fdb05ba2a15f0439691a37e2708b1a7f7560ee55b885b41a2fe65edb0c0c71d1`이다. GPU RTX3090은 frozen forward inference에만 사용했다. backward, autograd.grad/backward, optimizer 생성은 MCP child에 tripwire를 적용했다. 자동 피팅·numerical gradient·strength sweep은 실행하지 않았다. CPU에서 기록된 활성값과 source weight로 LLM이 명시한 행렬 산식을 계산했다.

추론 전 protocol (`runs/small-model-repetition-001/protocol.json`, 로컬 전용)을 고정했다. 직접 작성한 짧은 이야기 discovery 16개, validation 16개를 사용하고 별도 32개를 봉인했다. 동일 template 계열의 작은 데이터이므로 표준 benchmark나 일반화 증거로 해석할 수 없다. CUDA greedy, 최대 128개 새 토큰, EOS 종료와 KV cache를 고정했다. 반복률은 EOS 이전 생성 토큰의 8-gram 중 중복 비율을 문장별 계산한 평균이다.

선택 기준은 반복률 5%p 이상 감소, reference NLL 증가 0.03 이하, accuracy 하락 2%p 이하, 짧은 출력 비율 증가 5%p 이하, 중앙 출력 길이 `min(32, 원본 중앙값×0.75)` 이상이었다. 기준은 결과를 보고 바꾸지 않았다.

## 관측과 수정 근거

실제 원본 생성에서 `It was a flower.`가 반복되는 등의 실패를 관측했다. discovery 평균 반복률은 약 17.85%, validation은 10.69%였다. layer 1·3·5·7의 attention output projection을 MCP로 관측했다.

첫 가설은 layer 1 heads 5·15·2·12의 failure-minus-reference 성분을 50% 제거하는 것이었다. 절대 signed projection을 확인하니 그 성분은 양쪽 모두 음수이고 정상 구간에서 더 컸다. 예측 방향과 정상 구간 변화 한도를 통과하지 못해 proposal-01 (`runs/small-model-repetition-001/proposal-01.json`, 로컬 전용)은 defer했고 실제 후보를 만들지 않았다. 차이가 크다는 사실만으로 제거 대상을 정하면 안 된다는 사례다.

두 번째 가설은 정상 구간에서 약하고 반복 구간에서 양의 투영이 나타난 layer 1 heads 8·9의 성분을 대상으로 했다. 대상 tensor는 `transformer.h.1.attn.attention.out_proj.weight`, shape `[64 out,64 in]`, 선택 열은 32..39, 행은 0..63이다. 기록된 활성값에서 선택한 단위 방향을 r, 선택 weight block을 W라 할 때 LLM이 명시한 수정은 `W_new = W - (W r) rᵀ`이다. strength 1은 해당 성분의 완전 제거를 뜻하며 sweep으로 선택하지 않았다. 반대 대조군은 `W + (W r) rᵀ`이다. basis-02 (`runs/small-model-repetition-001/basis-02.json`, 로컬 전용)에 이유와 대안 설명을 보존했다.

후보당 실제 512개 값, 전체 parameter의 약 0.01367%를 변경했다. 해당 tensor의 상대 L2 변화는 0.13336이었다. 예를 들어 `[0,32]`는 `0.00139345508 → 0.00284024584`, `[0,33]`은 `0.00641413778 → 0.00438145269`로 바뀌었다. 원본에서 매번 분기하며 변경을 누적하지 않았다.

실제 수정 후 동일 위치의 module 출력 차이는 사전 affine 예측과 최대 `2.7381e-8` 오차로 일치했다. 이는 수정값의 실제 적용과 국소 계산 경로를 확인한다. 해당 성분의 의미가 반복에만 전용이라는 증명은 아니다.

## validation 결과

| 후보 | 반복률 ↓ | reference NLL ↓ | reference accuracy ↑ | 중앙 생성 길이 | 판정 |
|---|---:|---:|---:|---:|---|
| 원본 | 10.690% | 3.73125 | 29.508% | 128 | 기준선 |
| 무수정 대조 | 10.690% | 3.73125 | 29.508% | 128 | 토큰까지 동일 |
| 성분 제거 | 3.883% | 3.77519 | 27.541% | 101 | NLL 보존 실패 |
| 반대 방향 | 21.870% | 3.75585 | 31.639% | 128 | 반복 기준 실패 |

제거 후보의 반복률 감소는 6.808%p이고 validation paired bootstrap 95% 구간은 약 [0.716,14.419]%p다. 그러나 reference NLL 증가 0.04394가 허용 0.03을 초과했다. 16문장 중 4문장은 반복이 오히려 증가했다. 같은 norm의 반대 수정에서는 12문장의 반복이 증가했다. 방향에 따른 행동 변화는 관측됐지만 정상 예측도 함께 손상돼 성능 개선 후보로 채택하지 않았다. validation의 사후 선택과 template 유사성 때문에 이 구간은 독립 일반화 검증이 아니다.

수정 부위 선택에는 기록된 64개 head를 비교한 사후 선택이 포함된다. 반복 위치는 24..128인 반면 정상 reference는 8..33 위치라 위치·길이·내용 교란도 있다. 다음 분석에서는 비슷한 위치·길이의 정상 continuation 대조가 필요하다. 단순 norm·평균 대비나 반복률 하나로 의미 역할과 전체 품질을 확정하면 안 된다.

## 도구 보완과 종료

- `probe_model`에 선택적 EOS 종료, KV cache, 종료 이유와 생성 길이를 추가했다. EOS 이후 계속 생성된 토큰이 반복 지표를 오염시키는 문제를 피한다. 기존 기본 decoding은 유지한다.
- 기록된 출력의 반복률·길이·정상 NLL/accuracy를 함께 확인하는 순수 평가 함수와 반대 사례 테스트를 추가했다.
- 초기 제안에서 LLM이 명시한 rank-one 산식과 실제 FP32 새 값의 정확한 일치를 검사한다. 도구가 수정 방향을 최적화하지 않는다.
- `confirm_generation`은 후보 고정 후 source/config/patch/protocol 및 초기 봉인 데이터 hash와 동일 decoding을 검사한다. 이번에는 후보가 없어 test 확인을 실행하지 않았다. 이 기능은 OS 접근 봉인이 아니다.
- 진단·평가 스킬에 signed projection, 위치 교란, EOS와 길이 보존, 후보 고정 후 test 확인을 반영했다.

호스트 초기 forward 5회, Claude 신규 요청 25회(실패 요청 포함), identity 1개와 nonidentity 후보 2개를 기록했다. 초기에 concurrent MCP 호출의 JSON decoding 오류와 중복 위치 입력 오류가 있었고, 순차 호출·중복 제거로 수정했다. 실패 요청도 예산에 포함했으며 새 탐색으로 숨기지 않았다. 도구 로그와 시도 이력은 ledger (`runs/small-model-repetition-001/ledger.json`, 로컬 전용), 최종 선택은 decision (`runs/small-model-repetition-001/decision.json`, 로컬 전용)에 있다.

호스트의 독립 감사 (`runs/small-model-repetition-001/host-audit.json`, 로컬 전용)는 source/protocol/public data hash, 정확한 patch 재계산, 실제 old/new 값, identity 토큰 일치, 모든 validation 지표를 확인했다. 재실행 명령은 다음과 같다. 모델 forward나 heldout 읽기는 수행하지 않는다.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B runs/small-model-repetition-001/audit_development.py runs/small-model-repetition-001 runs/small-model-repetition-001/host-audit.json
```

전체 테스트는 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider`로 **79 passed**였다. 두 수정 스킬의 형식 검증도 통과했다. native loop는 종료했고 선택은 null로 고정했다. 봉인된 32개 test는 worker에게 제공하지 않았고 평가·후보 재선택에도 사용하지 않았다.
