# 현재 모델을 읽는 방법

스킬은 판단 절차, MCP는 반복 가능한 계산을 제공한다. LLM이 대상과 수치를 정한다. 새로운 가설을 만들 때마다 임의 Python 업데이트 프로그램을 작성할 필요는 없다. 기존 도구로 표현하지 못하는 새 분석은 범위를 설명하고 구현·검증한 뒤 사용한다. 원문에서 사용한 방식과 우리의 지원 범위는 [출처 조사](../../../docs/WEIGHT_SPACE_SKILL_DESIGN.md)를 참조한다.

## 구조와 실제 계산식

`describe_model_structure(model_dir, prefix, max_modules, offset)`는 실제 소유 계층과 parameter alias를 페이지로 반환한다. 트리 순서와 실행 순서는 다르다. `inspect_module_computation(model_dir, modules, candidate_dir)`는 실제 `forward` 소스·hash·소유 parameter shape와, 표준 Linear/Conv1D/Embedding/LayerNorm의 식을 반환한다. 소스가 잘렸거나 helper 호출이면 해당 parent/helper 경로가 아직 미관측임을 남긴다.

Linear는 W[out,in], Conv1D는 W[in,out]이다. 같은 이름의 c_proj라도 실제 module class를 확인한다. norm·bias·embedding도 읽을 수 있지만 읽기 허용은 편집 허용을 뜻하지 않는다. `read_parameter_entries(checkpoint,tensor,flat_indices)`는 벡터·스칼라·고차 tensor의 좌표와 실제 값을 읽으며, `read_tensor_slice`는 명시적 2D block에 쓴다.

## 토큰이 여러 모듈을 지나는 관측

다음은 `trace_token_flow` 인자 형식이다. 이름·ID·위치·좌표는 현재 모델과 개발 데이터에 맞게 LLM이 정한다. 아래 placeholder를 그대로 실행하지 않는다.

```json
{
  "model_dir":"<local model>","dataset_path":"<development data>",
  "selections":[
    {"module":"<actual norm module>","input_features":[0,1],"output_features":[0,1]},
    {"module":"<actual projection module>","input_features":[0,1],"output_features":[0,1]}
  ],
  "examples":[{"id":"<development id>","positions":[4,12]}],
  "output":"flow-before.json","split":"discovery","device":"cpu"
}
```

최대 16개 모듈, 4개 예제, 예제당 256토큰·8위치, 축별 최대512좌표이며 전체 선택 scalar는65,536개 이하다. 이 한도 안에서 MLP hidden 축 전체를 읽을 수 있다. 반복 호출도 같은 총량 한도에 포함한다. position p는 입력 tokens[p]와 다음 tokens[p+1] 예측에 대응한다. signed_values는 rank3 [batch,token,last_axis]라는 sampling 가정을 사용한다. 실제 모듈 소스로 이 layout을 확인한다. 그 외 rank는 shape만 반환한다. 같은 위치가 없는 출력은 상태를 명시하며 값으로 꾸미지 않는다.

`input_same_tensor_as_event_output`은 같은 **살아 있는 Tensor 객체**를 실제로 관측한 연결이다. None이면 잔차 덧셈·activation·gate 같은 함수 연산 또는 선택하지 않은 모듈을 통과했을 수 있다. 완전한 그래프를 복원한 것으로 취급하지 않는다. 예를 들어 MLP up→GELU→down에서 up/down만 선택하면 가운데 연결은 미관측일 수 있다. block·norm·activation 경계를 추가로 선택하고 실제 소스의 식과 대조한다.

후보를 적용한 뒤 같은 examples/positions/selections로 다시 추적한다. 모듈 입력도 바뀌면 단순히 ΔW×원래 입력만으로 실제 출력 변화를 설명할 수 없다. 같은 입력에서의 affine 예측과 downstream 네트워크 효과를 구분한다. `trace_token_flow`는 기록을 저장하지만 두 결과의 의미를 자동 판정하지 않는다.

## weight geometry와 회로 가설

`analyze_matrix_geometry(checkpoint,tensor,rows,columns,rank,right_operand)`는 선택 block의 SVD·행/열 norm과 선택적 A@B를 계산한다. rank는 반환할 basis 개수이며 편집 rank를 자동 선택하지 않는다. 두 block은 동일 checkpoint에서 읽고 실제 저장 방향으로 곱한다. transpose나 head mapping이 필요하면 현재 module 소스와 좌표를 먼저 확인한다.

행렬곱은 중간 nonlinearity·normalization·RoPE·mask·gate·softmax·residual을 생략한다. 같은 차원이라고 기능적으로 연결된 것은 아니다. QK/OV 같은 분석에서도 실제 head 분할과 positional 변환을 확인하고, geometry 결과는 다음 관측/개입을 설계할 가설로 사용한다.

생성 실패를 분석할 때는 실제 실패 continuation이 포함된 dataset을 사용한다. 반복의 경우 `build_repetition_contexts`로 discovery 생성에서 실제 재발 구간과 동일 위치 정상 참조를 만들 수 있다. 정상 문장의 단순 재등장을 생성 루프의 대리 표본으로 사용하지 않는다.

MLP writer의 기여는 특정 token 위치의 실제 activation과 선택 weight를 연결해 `analyze_parameter_contexts`로 확인한다. 성공/실패 label은 호출자가 붙인 주장이다. 정상과 실패가 다른 위치·길이에 있으면 공통 위치 성분을 실패 전용 방향으로 오인할 수 있다. magnitude만 보지 말고 양쪽 signed projection과 실제 영향을 확인한다. 관측 상관이 큰 것이나 local 변화가 작은 것이 인과/보존 보증은 아니다.

## 명시적 수정과 누적

지원 연산은 set_values, attenuation scale, project_out, rank_one이다. 계수는 caller가 명시한다. scale만 [0,1] 감쇠 의미를 유지하고, project_out/rank_one은 유한한 비음수 계수를 받는다. project_out 계수가 1보다 크면 반전/과제거가 가능하다. 변경 비율과 tensor 상대 L2는 별도 caller-declared 예산으로 검사한다. 이 검사는 성능 보존을 뜻하지 않는다.

후속 편집은 `revise_candidate`에 parent와 이번 operations만 준다. rationale 필드는 observation, hypothesis, value_derivation, prediction, disconfirming_result다. 도구가 설명의 진실이나 causal role을 승인하지 않는다. 같은 tensor 연산은 sequence로 합치고 각 단계마다 저장 dtype으로 반올림한다. 최대 32단계다. 새 값의 expected_values는 실제 parent 값이며 stale 값은 거부한다. 변경하지 않은 parent 수정도 유지한다. 단순 `create_candidate`는 계속 원본 분기 도구다.

단위 방향 `u`에 대한 출력축 projector `P(a)=I-a*u*u.T`를 같은 좌표에 반복 적용하면 `P(b)P(a)=P(a+b-a*b)`다. 계수를 단순히 더한 것과 다르며, 과제거된 parent에 projector를 다시 적용하면 의도와 반대로 방향 성분을 키울 수 있다. 원본 기준 목표 세기를 바꾸려면 실제 parent 값에 대한 명시적 delta를 유도한다. 예를 들어 원본 `W0`의 세기를 `a`에서 `t`로 바꾸는 추가 변화는 `-(t-a)*u*(u.T@W0)`이며 적절한 `rank_one`으로 표현할 수 있다. 현재 도구의 축·벡터 길이·계수 부호 조건을 확인하고 실제 dtype 반올림과 최종 값을 다시 검증한다. 이 항등식은 같은 방향·같은 선택 좌표에 한정된다.

parent/child manifest와 실제 patch, parameter 비교, before/after 흐름, inference 평가를 함께 저장한다. prepare_edit_proposal은 원본 분석에 결합된 초기 가설 도구이며 parent 분석을 초기 원본 분석으로 속이지 않는다. 복합 수정에서 최종 source-bound 제안을 만들 때는 기존 retained 변경도 포함하고, parent 대비 새 효과는 별도 관측으로 설명한다. 세기나 좌표를 자동 피팅하는 코드는 만들지 않는다.

같은 문맥의 인접 위치 여러 개가 서로 다른 문맥 여러 개를 대신하지 않는다. `analyze_parameter_contexts.context_evidence`는 같은 token sequence/위치의 중복을 제거한 distinct_sequence_pairs와 문맥별 signed 평균을 제공한다. 서로 다른 ID에 동일 토큰열을 붙여도 새로운 문맥으로 세지 않는다. 이 개수도 통계적 독립이나 의미적 정답을 보증하지 않는다. 신호 선택에 쓴 프롬프트와 선택에 쓰지 않은 반증 프롬프트를 구분하여 부호·효과가 유지되는지 확인한다.

수정의 정방향과 반대 방향이 모두 지표를 개선하거나 정상 예측을 함께 손상시키면, 제안한 단순 방향 메커니즘이 결과를 설명하는지 재검토한다. 현재 source·같은 입력·동일 크기 개입의 실제 downstream 결과와 생성 텍스트를 비교한다. 양쪽 효과만으로 모든 인과 역할을 배제하지 않으며, 작은 local delta나 전체 변경 개수만으로 정상 기능 보존을 추정하지 않는다. 추가 세기 탐색으로 기준을 맞추는 대신 반증된 가설과 불확실성을 보존한다.
