# 직접 값 수정 방식

현재 허용되는 업데이트는 LLM이 이유를 설명해 직접 작성한 숫자 또는 명시적 국소 matrix 연산이다. [엄격한 실행 계약](INFERENCE_ONLY_CONTRACT.md)을 따른다. AdamW/LoRA/gradient-based 제안은 비교를 위한 실행 옵션도 아니며 현재 금지다.

| 모듈 | weight 배치 | forward | 입력 feature 선택 |
|---|---|---|---|
| torch.nn.Linear | [out,in] | x @ W.T + b | columns, axis1 |
| GPT2 Conv1D | [in,out] | x @ W + b | rows, axis0 |

실제 module class는 describe_parameter로 확인한다. MLP output의 입력 feature는 residual 방향을 쓰며 bias와 다른 residual 경로는 그대로 남는다. 특정 feature의 의미를 shape/norm만으로 알 수는 없다.

set_values는 LLM이 읽은 old 숫자와 명시한 new 숫자를 비교해 지정한 rows×columns에 쓴다. 순서와 shape를 보존하고 dtype 반올림 후 실제 변경량·L2 예산을 검사한다. 미분이나 loss fitting이 없다.

scale/project_out/rank_one은 국소 가설이 정당화될 때 사용한다. project_out/rank_one의 strength는 caller가 정한 유한한 비음수 값이며 1 초과도 허용한다. project_out은 1 초과에서 방향 성분을 반전시킬 수 있다. scale만 [0,1] 감쇠를 유지한다. 계수 지원으로 원문 최종 편집의 성능을 재현했다고 부르지 않는다. rank_one의 left/right는 실제 선택 block 행/열 길이이고 delta는 strength * outer(left,right)다. 낮은 rank라는 이유로 LoRA 훈련으로 부르지 않는다.

수정 전후 describe_parameter/diagnose_model/probe_model/score_candidate를 비교한다. LLM이 새 가설과 값을 판단하며 자동 strength/좌표 sweep은 하지 않는다. 숫자 변경·역할 가설 확인·성능 개선은 별도 결론이다. [공개 실행 코드 조사](WEIGHT_EDIT_EVIDENCE.md)는 외부 연구의 방법에 대한 자료이며 우리 모델의 성능 증거가 아니다.

누적 편집은 revise_candidate로 이전 검증된 후보에 명시적 operations를 추가한다. 같은 tensor는 sequence로 표현해 단계별 dtype 반올림과 현재 expected_values를 검증한다. 자동 값 선택은 없고 manifest에 parent hash와 LLM의 근거를 남긴다. 계산 구조와 분석 스킬은 [통합 진입점](../skills/weight-space-engineer/SKILL.md)을 참조한다.
