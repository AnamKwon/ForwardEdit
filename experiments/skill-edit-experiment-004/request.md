사용자 요청: 실험을 해보고 제대로 동작하는지, 실제 파라미터 업데이트와 성능 개선이 되는지 확인한다.
기존 명시 조건: LLM이 실제 weight와 추론 관측을 읽고 대상/방향/수정량을 판단한다. GPU는 추론에만 쓴다. backward/autograd/optimizer/학습/numerical gradient/자동 loss fitting과 blind sweep은 금지한다.
새 스킬과 MCP의 구조/계산/토큰 흐름/누적 편집을 실제로 사용한다. 고정 원본·parent·child 실제 값과 추론 결과를 연결한다. source/config/decoding/data는 유지하며 독립 test로 재선택하지 않는다.
운영 계약: protocol.json의 데이터·예산·기준은 baseline/편집 전 고정했다. 새 후보는 최대6개, primary편집 최대4개, worker추론요청 최대55회. 실패로 되돌아갈 수 있지만 실제 parent를 관측한 뒤 revise_candidate로 최소2개 후속 개입을 확인한다. 근거 없는 개입은 강제하지 않고 미완료를 보고한다.
성공을 보장하라는 요구가 아니다. 준비한 하나의 후보를 동결하고 host가 별도64개 test로 딱 한번 평가한다. validation 부적격 후보도 diagnostic_only로 고정해 독립 확인할 수 있으나 채택에는 validation과 test 모두 기준 통과가 필요하다.
고정 verification의 update/method/evaluation 사례는 실제 실행/수치 일관성만 검사하며 semantic role이나 독립 성능 개선은 별도 결론이다. baseline data는 공식 TinyStories validation의 고정 revision에서 가져왔다. pretrained model이 과거에 그 문장을 보지 않았다는 보증은 없다.
