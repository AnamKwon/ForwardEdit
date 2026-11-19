# 공개 프롬프트·로그에서 재사용 스킬로

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

2026-11-04. 사용자 요청에 따라 [블로그](https://rikiyat.github.io/blog/weight-space-programming/)와 [공개 저장소](https://github.com/RikiyaT/lm_understand-logs)를 다시 조사했다. 아래는 공개 방법의 조사와 도구 구현이며 원 연구 성능의 독립 재현은 아니다.

## 확인한 원본

고정 commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`의 Experiment 8 p50c를 확인했다.

- [초기 프롬프트](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/turns/t001/prompt.txt): CPU Python kernel에서 현재/원본 weight를 읽고 계산한 tensor를 저장한다. 모델 실행은 별도 forward 도구로 분리하며 훈련·autograd·optimizer·loss에 대한 반복 fitting을 금지한다. 대상·방향·세기는 에이전트가 근거로 판단한다.
- [실제 tool_calls.jsonl](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/tool_calls.jsonl): 총 150개 record를 확인했다. 다운로드 SHA-256은 `4d772ae69caeb1099ee8d134375baa5738c8bdce20bc44236acd466d8c76c597`이다. zero-based record 7은 matched context의 residual/MLP 관측과 weight writer 분석, record 12는 LLM이 정의한 direction edit 함수와 save_edit 실행을 담는다. 코드·출력을 읽었으며 외부 코드를 실행하지 않았다.
- [최종 답변](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/final_answer.md): 선택한 일부 MLP 열, 방향 개입과 별도 문제의 paired 결과를 보고한다. 공개 보고값과 우리 실험 수치를 섞지 않는다.
- [apply_edit.py](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py): 원본·편집 tensor hash와 dtype 반올림까지 대조해 실제 평가한 가중치를 복원한다.

원문 저자는 Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab이다. 공개 저장소는 로그/데이터를 CC BY 4.0, 자체 스크립트를 MIT로 표시한다. 여기에는 원문 전체나 업데이트 코드를 복사하지 않고, 출처가 있는 조사와 독립 구현을 남긴다.

## 무엇을 고정하고 무엇을 LLM에 맡기는가

도구는 읽기·관측·행렬 계산·검증·저장을 반복 가능하게 만든다. 스킬은 각 관측이 어떤 질문에 답하는지와 근거의 한계를 안내한다. **수정할 module/parameter/좌표, 방향, 계수, 새 값과 다음 가설은 LLM이 정한다.** 특정 논문의 레이어·64개 뉴런·계수6을 새 모델의 기본값으로 넣지 않았다. 통계로 선택한 값을 정답처럼 자동 적용하지 않는다.

기존 project_out/rank_one의 계수 상한1은 원문과 같은 과제거 표현을 막고 있었다. 이번에는 caller가 유한한 비음수 계수를 명시하게 확장했다. 1을 넘으면 방향 반전도 가능하며, source hash·실제 변경률·상대 L2 예산은 계속 검사한다. attenuation scale은 감쇠 연산이라 [0,1]을 유지한다. 특정 계수의 지원은 그 계수가 좋다는 권고나 원문 재현을 뜻하지 않는다.

## 추가한 재사용 기능

| MCP | 제공하는 근거 |
|---|---|
| describe_model_structure | 실제 module 소유 계층, parameter shape/dtype, 공유 parameter alias |
| inspect_module_computation | 실제 로딩된 forward 소스/hash와 표준 모듈의 식, bias/norm/nonlinearity/residual 경로를 읽을 출발점 |
| read_parameter_entries | 2D 행렬에 제한되지 않는 명시적 parameter 좌표/값 조회 |
| analyze_matrix_geometry | 선택 block의 SVD/norm, 명시한 두 block의 곱; semantic role이나 편집값을 자동 결정하지 않음 |
| trace_token_flow | 같은 token 위치에서 여러 module의 실제 signed 입출력, 최종 target log-probability, 관측된 live tensor identity 연결 |
| revise_candidate | 검증된 parent에 LLM이 명시한 이번 수정만 합쳐 저장; 현재 값 확인, dtype 반올림, parent hash·근거 보존 |

`trace_token_flow`의 연결은 완전한 graph가 아니다. Tensor 객체가 달라지는 residual 덧셈·비선형 연산 등은 미관측 연결로 표시하며 호출 순서로 edge를 지어내지 않는다. actual forward 소스를 함께 읽어야 한다. 행렬의 합성도 중간 gate·softmax·RoPE·norm을 생략하므로 네트워크 전체 실행과 같지 않다.

## 스킬 진입점

[weight-space-engineer](../skills/weight-space-engineer/SKILL.md)가 구조 읽기 → 필요한 관측 선택 → LLM의 값 설계 → 기존 MCP 저장 → 재관측 → 독립 평가를 연결한다. 세부 분석법은 조건부 reference로 분리했다. 기존 diagnose/update-design/edit/evaluate 스킬은 이 도구를 활용하며 특정 모델의 성공 답을 복제하지 않는다. skill-creator와 loop-experiment-learning 지침을 적용했고, skill 형식 통과와 실제 모델 성능 향상은 구분한다.

우리의 [편집 반복 실험](EDIT_ITERATION_RESULTS.md)은 정상 예측 보존과 반복 감소의 절충만 부분 개선했고 선택 기준은 실패했다. 이 경험에서 ‘위치/길이 교란을 점검하라’, ‘수정 후 실제 흐름을 재관측하라’는 조건부 분석 질문을 가져왔다. 어떤 방향을 제거하면 항상 좋아진다는 규칙으로 승격하지 않았다. 새 스킬의 독립 transfer 효과나 일반적인 블랙박스 해석 능력은 아직 검증되지 않았다.

## 실제 검증

전체 테스트는 **87 passed**였다. 훈련하지 않은 known-value fixture에서 residual 덧셈을 거친 연결을 직접 tensor identity edge로 지어내지 않는지, 실제 signed 값·SVD 재구성·행렬곱·고차 tensor 좌표·dtype를 확인했다. parent 유지와 같은 좌표 재수정, stale 값 거부, 실제 stdio MCP의 누적 편집과 새 분석 API도 검사했다. 계수1 초과를 허용해도 명시한 L2 예산 초과를 거부한다.

별도로 실제 pretrained TinyStories에서 MCP로 구조/계산 소스/벡터 값/geometry/다중 module token 흐름을 조회했고 CUDA trace를 저장했다. 모델 source는 그대로이고 새 candidate는 만들지 않았다. verification.json (`runs/skill-tooling-003/verification.json`, 로컬 전용)과 실제 흐름 (`runs/skill-tooling-003/flow-report.json`, 로컬 전용)에 근거를 남겼다. 모듈간 observed identity 연결과 미관측 functional 경로를 구분했다. 이는 도구의 실제 동작 검증이며 새 성능 실험이나 학습 성공 주장이 아니다.
