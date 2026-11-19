# 직접 weight 편집의 실행 계약

ForwardEdit는 LLM이 실제 숫자와 forward 관측을 읽고 가설을 세워 직접 수정하는 Weight-Space Programming 도구다. AdamW·gradient 학습을 사용한 과거 실험은 현재 실행 계약과 맞지 않으며 이 방식의 증거에서 제외한다.

## 원문에서 확인한 금지

[Experiment8 p50c 최초 프롬프트](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/runs/p50c/turns/t001/prompt.txt)의 RULES에는 다음 구절이 있다.

> no backward passes, autograd or optimizers

같은 규칙은 gradient descent뿐 아니라 다른 훈련과 loss에 맞춘 반복 피팅도 금지한다. weight와 forward 결과를 분석하고 근거를 설명할 수 있는 직접 편집은 허용한다. 원문의 parameter 분석 kernel에는 GPU가 없고 모델 실행은 별도 forward tool로 제한된다. 우리도 CPU에서 값을 편집하고 GPU는 MCP 추론 도구에서만 사용한다.

기본 FP32 경로와 TinyStories 실험, 별도 BF16 research 경로와 Qwen 실험은 지원 범위·정밀도·평가 조건이 다르다. 원문의 embedding/head/final norm·전역 scalar 제한과 생성 조건을 모두 재현했다고 자동으로 주장하지 않는다. 역할 가설은 실제 module 구조, matched 성공/실패 관측, 국소 개입으로 검증한다.

## 구현 경계

- 모델은 eval 상태로 로드하고 requires_grad를 끈다. model forward는 inference_mode다.
- MCP에서 gradient 요청 옵션을 제거했다. 과거 Python gradient 요청은 실행 전에 거부한다. gradient 기반 설계 mode도 거부한다.
- write_parameter_values/set_values는 source hash와 실제 old value를 확인하고 LLM이 지정한 새 숫자를 정확한 좌표에 쓴다. hidden optimizer는 없다.
- describe_parameter는 현재 숫자와 실제 module/layout/구조상 역할을 반환한다. 의미 기능은 가설이라고 표시한다.
- probe_model은 실제 top-k와 greedy 생성, diagnose_model은 원본/후보의 forward activation, score_candidate는 개발 split의 실제 추론 지표를 반환한다. CUDA는 이 추론에만 사용한다.
- loss는 관측/검증 지표다. 수치 gradient, 무작위 검색, 좌표/세기 sweep, 자동 피팅으로 LLM 판단을 대체하지 않는다.
- 관측한 key 활성과 출력 임베딩 오차로 LLM이 층·문항·계수·산식을 명시한 1회 다중 오차 CPU 갱신은 허용한다. 반복 피팅, least-squares/pinv solve, optimizer와 GPU 학습은 계속 금지한다.
- 초기 수정 근거는 [parameter 분석](PARAMETER_ANALYSIS.md)에 따라 위치별 부호/기여와 old/new 수치의 관계를 기록한다. prepare_edit_proposal은 보류 또는 제한된 인과 probe 준비만 판단하며 설명문의 진실·의미 역할·성능을 승인하지 않는다.
- 테스트 전체와 실제 stdio 서버 테스트에는 backward/autograd/optimizer 실행을 거부하는 tripwire가 있다. 이는 테스트된 실행 경로에 대한 검증이며, 임의 shell 코드를 통제하는 OS sandbox라고 주장하지 않는다.

## 실험 기록

각 후보에서 조회한 좌표/old값, 모듈과 의미 가설, new값과 선택 이유, 예측, 실제 forward 효과, 회귀를 보존한다. 후보/forward 예산을 고정하고 개선이 없으면 no_gain/unknown을 보고한다. validation으로 후보를 고정한 뒤에만 최종 데이터를 평가한다. test를 보고 같은 후보군을 다시 선택하지 않는다.

성공을 세 단계로 구분한다: 숫자가 정확히 바뀌었는가, 예측한 행동이 바뀌었는가, 독립 데이터에서 성능이 개선했는가. 첫 단계의 성공만으로 ‘LLM이 가중치를 학습해 능력을 높였다’고 주장하지 않는다.

## 과거 실행의 처리

archive/gradient_experiments/*.txt는 과거 실수의 코드 기록이며 기본 실행/테스트 경로에서 제외했다. 과거 runs와 docs/evidence 데이터는 감사 가능하게 유지한다. AdamW·gradient를 사용한 결과와 fixture 학습을 섞은 예제를 이 계약의 성공 근거로 인용하지 않는다. 새 검증은 훈련 없이 수행한다.
