---
name: weight-edit
description: Write LLM-authored numeric values at inspected coordinates or apply explicit localized matrix operations, preserving source and excluding all training.
---

[실행 계약](../../docs/INFERENCE_ONLY_CONTRACT.md)에 따라 훈련·gradient·optimizer·자동 fitting 없이 직접 편집한다. 먼저 원본 hash·module/layout·작은 실제 값을 읽는다.

write_parameter_values(checkpoint,source_sha256,tensor,rows,columns,expected_values,values,hypothesis,limits,output)를 사용한다. expected_values는 실제 읽은 원본 값, values는 LLM이 이유를 설명해 정한 새 값이다. hash/원래 값이 다르면 다시 읽고 판단한다. 값은 rows×columns 순서이며 최대4096개다. create_candidate의 kind=set_values plan도 같은 의미다.

scale/project_out/rank_one은 정당화된 국소 연산에 사용한다. 축을 추측하거나 gradient로 벡터를 계산하지 않는다. preview_edit와 저장 모두 변경 비율·tensor 상대L2·유한성을 검사한다. 예산은 행동 보존을 보장하지 않고 실패 때 임의로 늘리지 않는다.

후보는 전체 대체 텐서와 manifest 두 파일이다. 패치만 from_pretrained로 읽지 않는다. create_candidate는 원본에서 분기한다. 연속 개입은 revise_candidate에 검증된 parent와 이번 operations·rationale·예산을 전달하면 기존 변경을 유지하고 sequence로 합친다. expected_values는 해당 단계의 실제 parent 값이며, 각 단계의 dtype 반올림과 최종 원본 대비 예산을 검사한다. manifest에 parent hash·근거가 저장된다. compare_candidate_parameters로 실제 유지/재수정/복원을 확인한다.

수정값은 describe_parameter(candidate_dir=...), 효과는 probe_model/score_candidate/diagnose_model로 확인한다. 원본과 tied-weight 거부를 지키고 보고서는 후보 밖에 저장한다. 숫자가 바뀐 사실만으로 학습/능력 개선을 주장하지 않는다.

기본 가설 실험은 먼저 [분석과 수정 근거](../../docs/PARAMETER_ANALYSIS.md)를 기록한다. prepare_edit_proposal의 plan을 사용할 때 decision과 reasons, 원본/분석 hash, 직접 정한 값과 예산을 확인한다. defer 상태에서 같은 가설을 승인된 것처럼 실행하지 않는다. 준비된 plan은 제한된 인과 probe용이며 배포나 성능 채택 판정이 아니다. 저수준 도구의 수치 동작 테스트와 의미 가설 실험을 구분한다.

수정 위치·계수·벡터·새 값은 LLM이 판단한다. project_out/rank_one은 유한한 비음수 계수를 명시하며 1 초과도 표현할 수 있다. scale은 [0,1] 감쇠다. 계수의 표현 가능성과 행동 보존·성능은 별도이며 원문 값을 기본값으로 복사하지 않는다.

MCP 단독 클라이언트는 먼저 `edit_api_schema`로 plan의 실제 필수 필드와 operation 규격을 확인한다. `scale`의 `strength`는 제거 비율이므로 `W_new=(1-strength)*W`이며 0은 유지, 1은 삭제다. `project_out`/`rank_one`의 strength는 해당 식의 alpha다. 숫자 참조를 사용하면 `preview_edit_from_arrays`로 수치 예산을 검사한다. 도구가 선택 위치나 계수를 추천하지 않는다.
