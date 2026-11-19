# API와 CLI 사용법

[공개 소개](../README.md) · [실행 계약](INFERENCE_ONLY_CONTRACT.md) · [MCP 설정](STANDALONE_MCP.md)

아래 기본 CLI/평가 경로는 FP32다. BF16 장문 추론은 별도 [research 경로](../skills/weight-space-engineer/references/research-tools.md)를 사용한다. 모든 예시의 좌표와 숫자는 실제 관측으로 작성해야 한다.

## LLM이 사용하는 도구

| 도구 | 관측/동작 |
|---|---|
| inspect_checkpoint / read_tensor_slice | 실제 이름·shape·hash와 작은 weight 값 조회 |
| describe_parameter | 실제 값, module class, 구조상 역할, 입력/출력 축 조회; 후보 값도 조회 가능 |
| probe_model | 실제 next-token top-k와 greedy continuation; 선택적 EOS 종료·KV cache, 생성 길이·종료 이유·텍스트 반환 |
| describe_model_structure / inspect_module_computation | 실제 모듈 계층·공유 weight·forward 소스와 계산식 조회 |
| read_parameter_entries / analyze_matrix_geometry | 임의 rank 좌표 읽기와 선택 block의 SVD·행렬곱 |
| build_repetition_contexts | 실제 생성 반복 구간과 같은 위치 정상 참조를 discovery 문맥으로 구성 |
| trace_token_flow | 여러 모듈의 token별 signed 입출력과 관측된 tensor identity 연결 |
| revise_candidate | parent에 LLM이 명시한 이번 변경을 누적, 현재 값·근거·hash 보존 |
| compare_candidate_parameters | 원본·이전 후보·새 후보의 실제 값 비교; 유지·재수정·복원·신규 변경 수와 hash 확인 |
| confirm_generation | 고정 후보·원본의 test 생성; 초기 봉인 데이터 hash와 고정 source/config/patch/decoding 검사 |
| diagnose_model | 원본/후보의 실제 forward input/output mean·RMS 및 weight norm; gradient 없음 |
| analyze_parameter_contexts | 선택 token 위치의 부호 있는 activation·weight 기여·문맥 대비·지정 새 값의 로컬 효과를 저장 |
| prepare_edit_proposal | 분석에 수정값 산출 이유·대안·반증·보존 기준을 연결해 보류/인과 probe 준비를 구분 |
| write_parameter_values | LLM이 명시한 원래 값과 새 값을 검사하고 정확한 좌표를 수정 |
| preview_edit / create_candidate | 명시적 set_values·scale·project_out·rank_one plan 검증/저장 |
| score_candidate | 편집 중 discovery/validation만 추론해 per-example NLL/accuracy 반환 |
| evaluate_baseline / evaluate_candidate | 원본 평가 또는 고정 후보의 paired 확인 |
| run_experiment | 이미 작성한 제한된 가설 plan 비교·선택 고정·독립 확인; 새 값을 자동 생성하지 않음 |

흐름은 **읽기 → 실제 추론 → 역할/원인 가설 → LLM이 새 숫자 지정 → 수정 → 추론 피드백 → 가설 재검토**다. 원본과 후보의 값, 숫자 차이, 실제 logits/생성 변화, 성능 개선은 구분한다. MLP output이라는 구조상 역할과 ‘반복을 일으키는 뉴런’이라는 의미 가설을 혼동하지 않는다. norm/activation이 크다는 이유만으로 제거하지 않는다.

초기 단계에서는 [수정 전 parameter 분석](PARAMETER_ANALYSIS.md)으로 실제 입력 부호와 선택 weight 기여를 연결한다. 명시한 reference/failure 위치를 비교하고, 새 값이 의도한 로컬 방향을 만드는지·참조 사례를 얼마나 바꾸는지 확인한다. 근거가 빠지거나 모순되면 **defer**, 설명·일부 로컬 조건이 맞으면 **ready_for_causal_probe**다. 인과/성능은 계속 미확인이고 실제 제한된 개입과 독립 확인이 필요하다. 저수준 write 도구를 모든 용도에서 봉쇄하는 기능은 아니다.

Linear는 `[out,in]`, GPT-2 Conv1D는 `[in,out]`이다. describe_parameter는 실제 모듈을 확인해 축과 forward 식을 반환한다. 의미 역할은 matched context와 국소 반증 개입을 통해 확인해야 한다.

## 직접 숫자 지정

write_parameter_values는 checkpoint, source_sha256, tensor, rows, columns, expected_values, values, hypothesis, limits, output을 받는다. expected_values는 읽은 원본 값, values는 LLM이 이유를 설명해 정한 새 값이다. create_candidate에도 아래 plan을 전달할 수 있다. 아래 숫자/이름은 형식 예시이며 실제 모델에서 읽은 값으로 작성해야 한다.

```json
{
  "checkpoint":"model/model.safetensors",
  "source_sha256":"ACTUAL_SOURCE_SHA256",
  "hypothesis":"관측한 실패와 이 좌표의 구조상 기여에 근거한 반증 가능한 가설",
  "limits":{"max_changed_fraction":0.002,"max_relative_l2":0.05},
  "operations":[{
    "kind":"set_values","tensor":"ACTUAL_TENSOR_NAME",
    "rows":[0],"columns":[1,2],
    "expected_values":[[0.12,-0.03]],"values":[[0.11,-0.025]]
  }]
}
```

rows×columns 순서로 최대4096개 값을 지정한다. hash/원래 값이 다르거나 새 값이 비유한 수, shape 불일치, 변경 예산 초과이면 거부한다. 저장 dtype으로 반올림된 실제 변경 원소 수와 상대L2를 검사한다. source는 그대로 유지된다. 후보의 replacements.safetensors는 변경 텐서의 전체 대체값이며 완전한 HF checkpoint가 아니다. 후보 디렉터리에는 이 파일과 manifest.json만 두고 보고서는 밖에 저장한다.

scale은 선택 block에 `(1-strength)`를 곱하고 project_out은 명시적 방향의 투영을 제거한다. rank_one은 `strength*outer(left,right)`를 더한다. scale은 strength [0,1], project_out/rank_one은 유한한 비음수 계수를 받는다. tensor당 연산 하나, plan당 최대32개다. revise_candidate의 sequence는 같은 tensor의 명시적 후속 연산을 합치며 각 단계마다 저장 dtype으로 반올림한다. set_values에는 axis/strength가 필요 없다. create_candidate는 원본 분기이고, revise_candidate는 기존 parent 변경을 유지하며 이번 명시적 operations를 합친다.

## CLI

```bash
.venv/bin/forwardedit --workspace /absolute/experiment inspect model/model.safetensors
.venv/bin/forwardedit --workspace /absolute/experiment describe model TENSOR --rows 0 1 --columns 0 1
.venv/bin/forwardedit --workspace /absolute/experiment probe model development.json --device cuda --max-examples 2 --max-new-tokens 16
.venv/bin/forwardedit --workspace /absolute/experiment diagnose model development.json TENSOR --device cuda
.venv/bin/forwardedit --workspace /absolute/experiment analyze-parameter analysis-spec.json analysis.json
.venv/bin/forwardedit --workspace /absolute/experiment prepare-proposal analysis.json basis-request.json proposal.json
.venv/bin/forwardedit --workspace /absolute/experiment preview plan.json
.venv/bin/forwardedit --workspace /absolute/experiment apply plan.json candidate-01
.venv/bin/forwardedit --workspace /absolute/experiment describe model TENSOR --rows 0 1 --columns 0 1 --candidate-dir candidate-01
.venv/bin/forwardedit --workspace /absolute/experiment score model candidate-01 development.json --device cuda
.venv/bin/forwardedit --workspace /absolute/experiment probe model development.json --candidate-dir candidate-01 --device cuda
```

CUDA는 probe/diagnose/score의 추론에만 사용한다. 최종 paired gate 평가는 CPU FP32로 수행한다. 로딩한 model은 eval/frozen이고 forward는 inference_mode다. include-gradients CLI/MCP 옵션은 제거했다. Python의 과거 gradient 요청과 gradient_assisted mode도 거부한다.

## 데이터와 독립 확인

```json
{"name":"local-probe","version":"1","splits":{
 "discovery":[{"id":"d0","tokens":[1,2,3]}],
 "validation":[{"id":"v0","tokens":[2,3,4]}],
 "test":[{"id":"t0","tokens":[3,4,5]}]}}
```

token ID는 동일 모델/tokenizer로 준비한다. ID는 split 전체에서 고유하고 문장은 두 token 이상이다. 편집 중에는 development 데이터만 worker에 제공하고 score/probe는 discovery/validation으로 제한한다. 후보 수·forward/token 예산·최소 개선·허용 회귀·생성 조건은 미리 정한다. 자동 loss fitting이나 strength/좌표 sweep은 금지다. 메커니즘 가설을 반증하는 제한된 개입·대조군·dose-response만 수행한다.

후보를 고정한 뒤 evaluate_candidate를 사용한다. 이 함수는 모든 split을 평가하므로 탐색에 사용하지 않는다. gate options의 min_nll_gain은 양수여야 하고, selection/heldout gain·heldout CI·accuracy·보존 split·누출 검사를 통과해야 한다. 같은 test로 차점 후보를 재선택하지 않는다. NLL 감소만으로 수학·자유 생성 능력 향상을 주장하지 않는다.

run_experiment config는 model_dir, dataset_path, plans(최대12개), min_nll_gain, selection_split, heldout_split, preserve_splits, max_nll_regression, max_accuracy_drop, seed, bootstrap, max_examples, max_tokens를 받는다. design_mode는 direct_edit만 허용한다. baseline/identity/selection evidence/frozen-choice/confirmation을 저장하고 원본을 자동 교체하지 않는다. 파일 접근 봉인·전체 실험 예산은 호스트가 관리하며 이 MCP는 OS sandbox가 아니다.

생성 반복 평가에는 `probe_model(stop_at_eos=true,use_cache=true)`와 기록 결과의 `summarize_generations`/`compare_generations`를 사용한다. EOS 이후 토큰은 제외하며 reference NLL/accuracy·출력 길이·짧은 출력 비율도 보존해야 한다. validation 선택을 고정한 뒤 `confirm_generation`으로 초기 봉인 데이터와 동일 decoding에서 원본/선택 후보만 test 평가한다. 이 생성 기준과 evaluate_candidate의 NLL 개선 기준은 서로 다른 판정이다.
