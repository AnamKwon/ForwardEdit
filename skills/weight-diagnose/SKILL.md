---
name: weight-diagnose
description: Inspect actual parameter values, structural roles and forward-only failure evidence before LLM-directed edits; never collect gradients or train.
---

[실행 계약](../../docs/INFERENCE_ONLY_CONTRACT.md)을 따른다. backward/autograd/optimizer/numerical gradient/자동 loss fitting은 금지다. GPU는 MCP forward inference에만 사용한다.

inspect_checkpoint로 hash·shape를 고정하고 read_tensor_slice와 describe_parameter로 작은 실제 숫자 묶음·module class·입출력 축을 읽는다. candidate_dir로 원본과 후보를 구분한다. 전체 weight를 context에 출력하지 않는다.

probe_model로 생성/next-token 분포를 보고 diagnose_model로 동일 입력의 원본/후보 activation mean·RMS를 비교한다. device=cpu|cuda는 추론 장치다. 큰 norm이나 activation만으로 원인을 확정하지 않는다.

Linear [out,in]와 GPT-2 Conv1D [in,out]의 실제 모듈을 확인한다. 구조상 MLP output이라는 사실과 ‘반복 뉴런’이라는 의미 가설을 구분한다. 실패/성공 matched context와 작은 반증 개입으로 역할을 확인한다. 도구가 제공하지 않는 attention/토큰 영역 분석은 미확인으로 남긴다.

관측·가설·대상 좌표·새 값의 이유·예상 forward 변화·반대 대조를 기록한다. discovery/validation만 탐색에 사용하고 test로 방향을 찾지 않는다.

수정 전에는 [위치별 분석](../../docs/PARAMETER_ANALYSIS.md)에 따라 analyze_parameter_contexts로 discovery의 명시적 reference/failure id와 token 위치를 비교한다. 부호 있는 input, 선택 weight의 실제 기여, 비교 차이, 지정한 새 값의 고정 입력 로컬 효과를 저장한다. 전체 평균으로 토큰별 부호가 가려지지 않게 한다. label은 호출자 주장이고 올바름/실패를 도구가 검증한 것은 아니다. shared prefix와 문맥 차이를 보고 pairing의 교란을 설명한다.

분석은 숫자를 고르지 않는다. proposed_values 없이 관측만 할 수 있다. 실제 역할이 불명확하면 근거를 더 모으거나 편집을 보류한다. candidate_dir로 같은 위치를 다시 분석하면 실제 개입 전후 module 출력과 target log-probability를 비교할 수 있다. 고정 입력의 affine 계산을 전체 모델의 인과적 기능으로 확대하지 않는다.

반복 생성의 늦은 위치와 짧은 정상 문장의 이른 위치를 비교하면 위치·길이·내용이 함께 바뀐다. 가능한 한 같은 위치·비슷한 길이의 정상 continuation 대조를 마련하고, 불가능하면 이 교란을 보존한다. failure-minus-reference 차이가 커도 failure의 절대 투영이 작거나 반대 부호일 수 있으므로 양쪽의 signed projection과 실제 delta를 따로 확인한다.

모델의 계산 구조가 불분명하면 [통합 분석 스킬](../weight-space-engineer/SKILL.md)의 필요한 부분을 사용한다. describe_model_structure/inspect_module_computation으로 실제 소유 계층과 forward를 읽고, trace_token_flow로 여러 경계의 같은 token 위치를 관측한다. live tensor identity 연결만 실제 연결로 표시하며 함수 연산의 미관측 edge나 semantic circuit을 지어내지 않는다. weight-only SVD/합성은 analyze_matrix_geometry로 계산할 수 있지만 실제 네트워크의 nonlinearity·norm·gate를 생략한다.

실제 생성 반복을 고치려는 경우 정상 teacher-forced 문장의 단어 재등장을 실패 표본으로 대신 쓰지 않는다. `build_repetition_contexts(reference_path, probe_paths, example_ids, output, ngram_size, min_occurrences, max_positions)`는 저장된 원본 discovery 생성에서 실제 n-gram 재발을 확인하고, EOS 이후를 제외하여 동일 프롬프트·동일 절대 위치의 참조/생성 문맥을 만든다. min_occurrences는 최소3이다. 선택 ID 중 제외된 항목과 이유도 기록한다. 출력 dataset의 `pairs`를 위치 분석에 사용할 수 있다. 이 도구의 label은 토큰 재발만 증명하며 내용의 오류나 의미 동등성은 증명하지 않는다. 일반 반복과 병적 반복의 의미 구분, 프롬프트 밖 내용 차이는 별도 검토한다.
