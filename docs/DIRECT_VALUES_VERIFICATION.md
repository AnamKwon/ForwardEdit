# 훈련 없는 LLM 직접 값 수정 검증

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

현재 계약은 [INFERENCE_ONLY_CONTRACT.md](INFERENCE_ONLY_CONTRACT.md)다. 원문 최초 프롬프트와 사용자 지시에 따라 backward·autograd·optimizer·훈련·numerical gradient·자동 loss fitting을 금지했다. 이전 AdamW/gradient 결과는 이 요청의 성공 근거에서 제외한다.

## 도구 변경과 실행 경계

- `describe_parameter`: 실제 weight 값, owning module, 저장 축, forward 식, 구조상 역할을 읽는다. 의미 기능은 확인할 가설로 표시한다.
- `write_parameter_values`: LLM이 읽은 expected_values와 직접 작성한 values를 특정 rows×columns에 쓴다. source hash·원래 값·유한성·shape·변경 예산을 검사한다. `set_values` plan도 지원한다.
- `probe_model`: 실제 next-token 분포와 greedy 생성. `diagnose_model`: 원본/후보의 실제 forward activation. `score_candidate`: discovery/validation의 per-example NLL/accuracy.
- 모든 inference model은 eval/frozen이고 forward는 inference_mode다. CUDA는 이 추론에만 사용한다. gradient 옵션·gradient 기반 runner mode는 거부한다.

기존 훈련 예제/테스트는 `archive/gradient_experiments/*.txt`로 옮겨 실행 경로에서 제외했다. 보관 자료는 삭제하지 않았으나 실행하거나 직접 편집 능력의 증거로 인용하지 않는다.

## 실제 Claude 실험

loop-engineering을 통해 Claude Sonnet 5.5가 직접 MCP를 호출했다. Codex 호스트가 도구 구현과 독립 검증을 담당했다. 편집 대상은 손상/재학습하지 않은 로컬 TinyStories checkpoint다. worker에게는 discovery/validation 각각12개의 짧은 이야기만 제공했다. 최종 데이터는 제공하지 않았다.

LLM은 GPU forward 결과를 보고 문장 뒤 newline token198이 자주 예측된다는 점을 관측했다. embedding은 수정하지 않고 token198에 대응하는 output 방향을 읽었다. 실제 마지막 MLP output 행렬의 두 residual 좌표와 작은 weight slice를 읽고, newline 방향에 기여할 수 있는 여섯 원소를 선택했다. 이는 작은 개발 데이터의 포맷 가설이며, newline 예측 자체가 모델 오류라는 뜻은 아니다.

실제 대상은 `transformer.h.7.mlp.c_proj.weight`, Linear `[64,256]`다. 아래 숫자는 LLM이 MCP에 명시한 값이다. 후보는 원본에서 각각 분기한다.

| 좌표 [row,column] | 조회한 원본 값 | 후보1 | 후보2 |
|---|---:|---:|---:|
| [34,16] | 0.035900675 | 0 | 0.0539 |
| [34,18] | 0.040977675 | 0 | 0.0615 |
| [18,4] | -0.052891526 | 0 | -0.0793 |
| [18,6] | -0.060694151 | 0 | -0.0910 |
| [18,21] | -0.057323437 | 0 | -0.0860 |
| [18,27] | -0.038828332 | 0 | -0.0582 |

후보1은 이 성분들이 newline을 유도한다는 가설 아래 0으로 만들었다. 하지만 실제 첫 관측 문맥의 newline log-prob는 -1.33616→-1.24995로 오히려 올라 가설이 반증되었다. LLM은 반대 역할을 가정해 후보2의 숫자를 직접 작성했다. 후보2에서는 -1.38513으로 내려갔으나 전체 NLL은 악화했다. weight 부호만으로 뉴런 activation 부호나 의미 역할을 확정할 수 없다는 한계를 기록했다.

| 후보 | discovery NLL | validation NLL | 결론 |
|---|---:|---:|---|
| 원본 CPU | 2.454863 | 2.925428 | 유지 |
| 후보1 CPU | 2.456361 | 2.926468 | NLL 악화, 첫 가설 반증 |
| 후보2 CPU | 2.467820 | 2.938720 | 일부 출력 변화는 예측 방향이나 NLL 악화 |

두 후보 모두 성능 개선으로 채택하지 않았다. 허용된4개 중2개만 실행하고 자동 피팅이나 추가 loss sweep 없이 종료했다. neuron의 의미 기능은 미확인이고, 두 결과로 ‘LLM이 가중치를 학습해 능력을 높였다’고 주장하지 않는다.

## 독립 확인

호스트가 원본 배열에 지정한 new 값만 직접 대입해 저장 패치와 원소별로 대조했다. 두 후보 각각 정확히6개 값이 바뀌었고 실제 로딩한 모델 값도 정확히 일치했다. 다른 parameter와 원본 hash는 보존되었으며 모든 parameter는 frozen, grad는 None이었다.

별도 실제 stdio MCP 서버에서 Tensor.backward·autograd.grad/backward·optimizer 생성 시 예외가 발생하도록 설정한 뒤 CPU/CUDA score와 GPU probe를 재실행했다. 예외 없이 같은 관측을 재현했다. worker의 CUDA NLL과 독립 재실행은 1e-8 이내로 일치했고 CPU/CUDA 차이는 1e-4 이내였다. GPU는 RTX3090에서 추론만 수행했다. 원본 SHA-256은 `fdb05ba2a15f0439691a37e2708b1a7f7560ee55b885b41a2fe65edb0c0c71d1`로 유지되었다.

전체 기본 테스트는 **50개 통과**했다. 기본 suite와 실제 MCP 서버 테스트 모두 training tripwire를 포함한다. 이 검사는 해당 실행 경로에 대한 증거이며 임의 shell을 통제하는 OS sandbox 증거는 아니다. 역할 가설의 진실이나 일반 성능 향상은 별도 문제다.

## 증거

- [LLM 도구 호출과 가설](evidence/direct-values-inference-only-ledger.json): 일부 큰 초기 관측은6000자에서 잘렸다고 명시되어 있다. 이를 완전한 원시 로그로 간주하지 않는다.
- [LLM 최종 결정](evidence/direct-values-inference-only-decision.json): selected_candidate=null, 한계와 no-training 자기점검
- [독립 호스트 검증](evidence/direct-values-inference-only-host-audit.json): 원소별 확인, CPU/CUDA 수치, 실제 no-training tripwire 실행
- [후보1 plan](evidence/direct-values-candidate-01-manifest.json), [후보2 plan](evidence/direct-values-candidate-02-manifest.json): 실제 old/new 값과 수치 변경량
- 전체 로컬 파일: `runs/direct-values-inference-only-001/`

native job은 `job-d51ea0a3a5b4eee21a7cde7ff497f42b`, loop는 `wsl-direct-values-20261104`다. 단일 Sonnet 호출에서2개 후보와12개 MCP 호출을 수행했다. 독립 호스트 검토로 완료를 승인하고 no_gain 상태로 고정했다. 이는 실험 완료 승인이지 개선 성공이나 모든 요건의 native 기계 검증이라는 뜻이 아니다.
