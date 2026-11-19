---
name: weight-evaluate
description: Evaluate LLM-authored direct edits through forward-only development feedback, controls and frozen independent confirmation; never train with an optimizer.
---

[실행 계약](../../docs/INFERENCE_ONLY_CONTRACT.md)을 따른다. backward/gradient/optimizer/훈련·자동 loss fitting은 금지다. 과거 AdamW 손실 감소는 직접 편집 성공이 아니다.

시작 전에 model/data hash·split·생성 설정·후보/forward/token 예산·최소 개선·허용 회귀를 정한다. 탐색 때 score_candidate(splits=[discovery,validation])로 개발 데이터만 추론한다. probe_model로 실제 생성/next-token을, diagnose_model(candidate_dir=...)로 예측한 국소 activation 변화를 확인한다. 모델 실행은 MCP 도구를 사용하며 kernel에서 직접 forward로 예산을 우회하지 않는다.

LLM이 관측을 해석하고 새 값을 명시한다. 자동 수치최적화/방향·좌표·strength sweep은 만들지 않는다. 반대·무수정·변경량을 맞춘 대조는 메커니즘 반증용이다. source와 실제 적용값도 대조한다.

validation으로 후보·plan/patch/hash/기준을 고정한 뒤 evaluate_candidate로 독립 확인한다. 이 도구는 모든 split을 평가하므로 탐색 후보마다 호출하지 않는다. run_experiment는 미리 작성된 제한된 가설 plan의 비교 도구이며 값을 학습하지 않는다. 같은 test를 본 뒤 차점 후보로 재선택하지 않는다.

숫자 수정·특정 행동 변화·성능 개선을 별도 결론으로 보고한다. paired 차이·불확실성·회귀·실패를 남긴다. NLL 감소로 수학/생성 능력 향상을 주장하지 않는다. 개선 없으면 원본을 유지하고 no_gain/unknown으로 종료한다.

초기 [parameter 분석](../../docs/PARAMETER_ANALYSIS.md)의 prefix·위치·old/new 값·로컬 예측을 보존하고, candidate_dir를 지정한 재분석에서 실제 module 출력/target log-probability를 대조한다. 고정 입력의 로컬 일치는 계산 경로 검증이며, 의미 역할이나 독립 성능 입증과 구분한다. prepare_edit_proposal이 서술 완성도와 일부 수치 조건을 통과시킨 것을 semantic role 검증으로 인용하지 않는다. 성공/실패 label과 pairing도 별도로 검증해야 한다.

생성 반복 실험에서는 EOS 이후 토큰을 평가에서 제외하고 원본·후보의 종료·cache·생성 길이와 decoding 방식을 고정한다. 기존 작은 모델의 greedy 경로와 research 샘플링 경로를 구분한다. 샘플링 실험을 greedy로 바꿔 효과를 주장하지 않는다. `summarize_generations`와 `compare_generations`는 기록된 결과만 계산한다. 반복 감소와 함께 정상 reference NLL/accuracy, 짧은 출력 비율·길이도 보존해야 한다. 반복 지표는 이야기 품질의 대용물이 아니다.

긴 샘플링 추론이나 backend가 다른 실험에서는 [대조군과 평가 단위](references/evaluation-controls.md)를 읽는다. 같은 seed만으로 생성 토큰의 완전한 재현성을 단정하지 않는다. 정상 손상은 전체 입력 대비 손실과 **원래 맞힌 입력 중 잃은 비율**을 구분하고, 원래 정답이 없으면 정답 보존을 확인했다고 표시하지 않는다.

작은 개발셋의 평균 개선은 문장별 기여와 paired 구간을 함께 본다. `compare_generations`는 최대 개선 문장을 제외한 평균 `gain_without_largest_prompt`를 반환하며, 사전에 정한 경우 `leave_largest_prompt_out_gain_positive`로 이를 선택 조건에 넣을 수 있다. `require_ci=False`에서 CI 검사 항목이 true인 것은 구간이 양수라는 뜻이 아니다. 실제 `gain_ci95`를 읽고, 개발의 반복 선택은 별도 독립 확인과 구분한다. 결과를 본 뒤 이 조건이나 기준을 유리하게 바꾸지 않는다.

생성 후보를 validation으로 고정하면 source/config/patch/protocol hash와 초기 봉인 데이터 hash를 가진 frozen_generation_choice를 저장한다. `confirm_generation`으로 원본과 선택 후보의 test 생성을 비교한다. 이 도구는 고정 값·초기 봉인 hash·동일 decoding을 검사하며, 호스트가 별도로 실험 종료와 파일 접근을 관리한다. test 결과로 후보·값·기준을 바꾸지 않는다. 후보가 없으면 test를 열어 재탐색하지 않는다.

편집 반복의 진척은 단계별 반복률뿐 아니라 정상 reference의 token log-probability/NLL·accuracy와 부모 대비/원본 대비 실제 parameter delta를 함께 본다. 이전 개발 데이터를 반복해서 본 경우 이를 독립 검증으로 표현하지 않는다. 편집 횟수나 weight 변화만으로 학습 성공을 선언하지 않고, 고정 후 별도 데이터에서 원본 대비 개선과 보존을 확인한다.

반복 편집은 [관측 축적과 적응적 편집](../weight-space-engineer/references/adaptive-edit-loop.md)을 따른다. 새로 손상된 사례는 실제 새 생성 prefix로 재관측하며, 기존 정답뿐 아니라 부모 후보가 새로 맞힌 사례의 손실도 추적한다. 이미 맞힌 문제의 종료 직전 위치를 고정해 얻은 확률 개선을, 달라진 생성 궤적에서도 같은 위치에 도달한다는 증거로 취급하지 않는다.

확률은 prefix·position·token ID와 함께 확인한다. `target_logprobs`가 관측된 다음 토큰을 점수화한 것이라면 `1-P(그 토큰)`을 다른 특정 토큰의 확률로 해석하지 않는다. 원하는 토큰이 측정되지 않았으면 미측정으로 남긴다. raw log-probability와 temperature·penalty·top-k/top-p 적용 뒤의 샘플링 확률도 구분한다.
