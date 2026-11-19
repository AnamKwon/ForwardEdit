---
name: weight-space-engineer
description: Investigate neural-network computation and author feedback-dependent direct parameter edits through MCP skills, without gradient training; use when interpreting weights or repairing model behavior.
---

사용자는 LLM이 실제 값을 읽고 대상·좌표·방향·계수·새 값을 판단하기를 원한다. 스킬과 MCP는 재사용할 관측·계산·저장 방법을 제공한다. 특정 레이어·뉴런·계수나 성공 방향을 기본값으로 주지 않는다. [실행 계약](../../docs/INFERENCE_ONLY_CONTRACT.md)의 학습·gradient·optimizer·numerical gradient·자동 loss fitting 금지를 따른다. 모델 실행은 MCP forward 도구만 사용한다.

처음에는 [분석 도구 사용법](references/analysis-playbook.md)을 필요한 부분만 읽는다. 실제 parameter 의미를 이해했다고 선언하기 전에 구조, 계산식, 관측, 개입의 증거 수준을 구분한다. 가중치의 숫자나 SVD 성분만으로 ‘반복 뉴런’, ‘추론 회로’ 같은 의미를 확정하지 않는다.

사용자가 여러 회차의 수정이나 벡터 흐름 추적을 요청하면 [관측 축적과 적응적 편집](references/adaptive-edit-loop.md)을 읽는다. 정상·실패·새 손상 사례를 버전별로 축적하고, 편집 후 실제로 달라진 생성 prefix를 다시 관측한다. 실패한 가설의 판정과 다음에 확인할 질문을 기록하며, 사용자가 요청한 실험 깊이를 단순 예산 상한으로 바꿔 조기 종료하지 않는다. 이 절차는 스킬 지침이며 서버가 의미 가설이나 반복 깊이를 자동 보장하는 기능은 아니다.

1. **계산 구조를 읽는다.** `inspect_checkpoint`, `describe_model_structure`로 모듈·소유 parameter·공유 weight를 찾는다. `inspect_module_computation`으로 실제 로딩된 `forward`를 읽고 bias·normalization·gate·nonlinearity·residual 경로를 확인한다. 작은 행렬은 `describe_parameter`/`read_tensor_slice`, 벡터·고차 텐서의 명시적 항목은 `read_parameter_entries`로 읽는다. 이름이나 shape를 근거로 축을 추측하지 않는다.
2. **가설에 필요한 관측을 선택한다.** weight-only geometry가 필요하면 `analyze_matrix_geometry`; 실제 토큰 흐름은 `trace_token_flow`; 위치별 writer 기여는 `analyze_parameter_contexts`; 평균·분포 관측은 `diagnose_model`을 사용한다. 모든 분석을 매번 실행하지 않는다. 실패와 정상의 실제 생성/예측을 확인하고, 위치·길이·내용 차이가 설명하는 대안도 둔다.
3. **LLM이 다음 변경을 결정한다.** 관측 → 구조적 역할 가설 → 명시적 새 값/산식 → 예측 → 반증 조건을 쓴다. 대상과 계수는 현재 증거에 근거해 정한다. norm·상관·행렬곱이나 다른 연구의 성공 값을 자동 추천으로 취급하지 않는다. 첫 의미 개입은 [초기 분석 계약](../../docs/PARAMETER_ANALYSIS.md)의 `prepare_edit_proposal`을 사용한다. defer/ready 모두 인과·성능 미확인이다.
4. **준비된 도구로 쓴다.** 첫 후보는 `write_parameter_values` 또는 `create_candidate`로 만든다. 다음 단계는 `revise_candidate`에 실제 parent, 이번 operations, 관측·수치 근거·예측·반증 조건을 전달한다. `set_values.expected_values`는 수정하려는 현재 parent 값이다. 이 도구는 이전 수정과 이번 명시적 변경을 source-bound sequence로 합치며 값을 고르지 않는다. `compare_candidate_parameters`로 유지·재수정·원복·신규 값을 확인한다.
5. **관측을 다시 읽고 반복한다.** 같은 입력/위치에서 parent/child의 module 출력·token log-probability와 실제 생성을 비교한다. 예상과 다르면 가설을 갱신한다. 실패 후보를 원복하면 그 사실을 기록한다. 미리 작성한 변형이나 무작위/좌표/strength sweep을 피드백 학습으로 세지 않는다. 횟수·추론 예산·허용 회귀는 실험 전에 정하고 실패를 숨기거나 사후에 완화하지 않는다.
6. **성능은 별도로 판단한다.** [평가 스킬](../weight-evaluate/SKILL.md)로 정상 사례 보존과 과제 성능을 함께 본다. validation을 반복 분석했으면 개발 결과로 표시한다. 최종 후보·hash를 고정한 뒤 별도 데이터로 확인하고 같은 test로 재선택하지 않는다. 값 변경, 내부 계산 경로 확인, 특정 행동 변화, 독립 성능 개선을 별개 결론으로 보고한다. 근거가 부족하면 보류/no_gain으로 남긴다.

긴 추론과 넓은 행렬은 [research 도구 안내](references/research-tools.md)를 읽는다. `research_forward`가 실제 추론과 성공 예산을 관리하고 `research_prepare_edit`는 해시로 연결된 실제 activation에 초기 분석 계약을 적용한다. 작은 모델의 기존 `prepare_edit_proposal`과 같은 준비 단계이며, 둘 다 개선 증명이 아니다.

사용자·교사 모델이 오류 설명이나 올바른 풀이를 제공하면 [교정 피드백 연결](references/corrective-feedback.md)을 읽는다. 피드백을 파라미터 역할의 증거로 단정하지 않고 관측 가능한 가설로 바꾸며, 힌트가 포함된 진단과 힌트 없는 가중치 개선 평가를 구분한다.

로컬 단일 safetensors를 지원한다. 기존 FP32 경로와 별도 research BF16 경로의 정밀도·지원 모델을 구분한다. trace의 tensor identity 연결은 부분 관측이다. 호출 순서는 완전한 computation graph가 아니며, 기능 연산이나 unselected 경로의 연결을 지어내지 않는다. 새 모델의 축·attention 구현·지원 범위는 실제 소스에서 확인한다. 이 스킬 자체는 아직 독립 모델 개선 성공을 입증하지 않았다.
