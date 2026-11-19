---
name: weight-update-design
description: Author explicit parameter values or localized matrix edits from actual weights and forward observations, reasoning about a failure mechanism without gradients or training.
---

[금지 조건](../../docs/INFERENCE_ONLY_CONTRACT.md)을 따른다. 사용자는 LLM이 weight를 해석해 직접 바꾸기를 원한다. AdamW/LoRA·gradient/numerical gradient·자동 loss fitting으로 대체하지 않는다. 과거 gradient 실험은 성공 근거가 아니다.

describe_parameter로 실제 모듈·축·작은 값을 읽고 probe_model/diagnose_model의 실패 관측과 연결한다. 해당 구조가 어떻게 실패를 만들 수 있는지 설명한다. 큰 값/activation만으로 제거할 대상을 정하지 않는다. 의미 역할을 모르면 가설/미확인이라고 남긴다.

가장 작은 가설 개입을 선택한다. write_parameter_values의 rows/columns/expected_values/values를 LLM이 명시한다. set_values plan도 같다. 구조적 근거가 있으면 scale/project_out/rank_one을 사용한다. rank-one delta는 LoRA 학습이 아니다. CPU에서 값을 지정하고 GPU는 추론만 한다.

읽은 값 → 역할 가설 → 수정 이유/숫자 → 예측 → 실제 forward 결과를 기록한다. 미리 정한 적은 후보/forward 예산에서 반증·dose-response를 확인한다. 무작위/좌표/strength sweep으로 loss를 최소화하는 solver를 만들지 않는다. 피드백은 다음 메커니즘 판단의 자료다.

실패를 모두 남기고 validation으로 후보를 고정한다. 개선이 확립되지 않으면 no_gain/unknown으로 끝낸다. [원문 코드 조사](../../docs/WEIGHT_EDIT_EVIDENCE.md)의 레이어 번호·방향·α를 다른 모델 기본값으로 옮기지 않는다.

초기 근거 기록은 [수정 전 분석 계약](../../docs/PARAMETER_ANALYSIS.md)을 사용한다. analyze_parameter_contexts의 분석 path/hash를 참조해 prepare_edit_proposal에 실패 관측·구조 역할·메커니즘·비교 근거·새 값의 산출 이유·대안 설명·예측·반증 조건·대조군을 작성한다. value_rule로 감쇠 산술을 검증하거나, 명시적 변경 좌표마다 이유를 적는다. desired output direction과 참조 영향 상한은 LLM이 이유를 설명해 선언하며 loss에 맞춰 자동 선택하지 않는다.

defer이면 추가 관측이나 가설 재검토로 돌아간다. ready_for_causal_probe는 작은 반증 개입을 위한 수치·설명 항목의 준비 상태이며 의미 역할/성능이 확인됐다는 뜻이 아니다. 판정은 선택적 실패 가설의 로컬 조건에 한정된다. 다른 유형의 편집을 이 규칙에 억지로 맞추거나 설명문을 증거로 승격하지 않는다. 실제 개입 뒤 같은 위치에서 예측·관측과 정상 사례 회귀를 확인해야 한다.

사용자가 편집 반복을 요청하면 각 단계에서 이전 후보의 실제 추론 피드백을 읽고 다음 수정 근거를 갱신한다. 후보를 쓰기 전에 parent 후보·patch hash·관측된 손상·새 값 산식·예측·반증 조건을 기록하고, 추론 뒤 예측과 실제 변화를 연결한다. source-bound overlay에서 누적 수정하려면 이전 변경을 포함한 최종 절대값을 재구성한다. `compare_candidate_parameters`로 유지·재수정·원복·신규 값을 구분한다. 원본에서 분기한 형제 후보나 미리 만든 strength 변형을 누적 학습 단계로 세지 않는다. rollback은 이유를 명시한다. 숫자 계보는 확인할 수 있지만 피드백 의존성과 성능은 별도 증거가 필요하다.
