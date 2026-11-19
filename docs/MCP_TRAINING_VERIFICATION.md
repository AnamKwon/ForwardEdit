> 역사 기록: 아래 optimizer/gradient 및 fixture 훈련 결과는 현재 사용자의 무경사 직접 편집 요청을 충족하지 않는다. 실행하지 않으며 성공 근거에서 제외한다. 현재 계약은 [INFERENCE_ONLY_CONTRACT.md](INFERENCE_ONLY_CONTRACT.md)다.

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

# MCP 파라미터 수정과 수정 후 학습 검증

2026-11-04, 실제 stdio MCP 서버와 클라이언트를 연결해 검사했다. **파라미터 수정과 수정 후 역전파·optimizer 업데이트는 정상이다. 사전학습 모델에서는 이번 학습 설정의 과적합도 확인했다.**

## 실제로 검사한 경로

`inspect_checkpoint → read_tensor_slice → preview_edit → create_candidate → evaluate_candidate`를 실제 MCP로 호출했다. NumPy로 독립 계산한 수정값, 저장한 safetensors, 로딩한 모델의 실제 `nn.Parameter` 값을 원소별로 대조했다. 세 값은 정확히 일치했다. 수정 대상 밖의 가중치와 원본 체크포인트는 그대로 유지되었다.

| 모델 | scale 변경 원소 | project_out 변경 원소 | rank_one 변경 원소 |
|---|---:|---:|---:|
| 소형 GPT-2 합성 과제 | 32 | 2 | 32 |
| 사전학습 TinyStories | 512 | 2 | 512 |

무수정 대조군은 변경 원소 0개, logits 차이 0이었다. 세 실제 편집은 모두 logits를 바꾸었다. TinyStories의 최대 logits 절대 차이는 scale 0.62807, project_out 0.00747, rank_one 0.51464였다. MCP의 후보 NLL과 별도 로컬 추론 NLL도 일치했다. 따라서 저장만 되고 실제 모델에 반영되지 않는 문제는 관측되지 않았다. 이 편집은 동작 확인용이며 세 연산 자체의 성능 개선을 주장하지 않는다.

## 수정된 좌표도 실제 학습되는가

rank_one 후보를 로딩한 뒤 편집한 행렬만 trainable로 두고 나머지 parameter를 고정했다. **MCP에는 학습 도구가 없으므로 AdamW 학습은 별도 로컬 검증 코드에서 실행했다.** discovery 데이터로 token-normalized NLL을 계산하고 lr=0.003, weight decay=0으로 30회 업데이트했다. 이는 `gradient_training`이며 직접 편집 자체와 구분한다.

- `nn.Parameter`와 `requires_grad`가 유지되었다.
- 모든 step에서 gradient가 유한하고 0이 아니었다. MCP가 바꾼 좌표만 추적해도 0이 아닌 gradient가 확인되었다.
- 합성 모델의 편집된 32개 좌표, TinyStories의 편집된 512개 좌표가 모두 optimizer에 의해 다시 업데이트되었다. 학습은 해당 좌표만이 아니라 **해당 행렬 전체**를 대상으로 했다.
- 다른 parameter는 바뀌지 않았고, optimizer를 실행하지 않은 별도 후보 대조군의 지표는 그대로였다.
- 학습 모델을 `save_pretrained`로 저장하고 재로딩한 parameter와 logits가 정확히 일치했다. 저장된 학습 모델을 MCP `evaluate_baseline`으로 다시 평가한 지표도 일치했다.

| 모델 | 학습 NLL 전 → 후 | validation NLL 전 → 후 | 해석 |
|---|---|---|---|
| 소형 GPT-2 합성 과제 | 1.41656 → 1.39888 | 1.44678 → 1.42812 | 이 검증 과제에서는 두 손실 모두 감소 |
| 사전학습 TinyStories | 2.45478 → 0.47971 | 2.92535 → 4.54941 | 학습은 진행되지만 validation 악화, 과적합 |

TinyStories validation 토큰 정확도도 38.56% → 24.51%로 내려갔다. 319개 학습 target token에 대해 16,384개 행렬 원소를 30회 조정한 작은 데이터의 학습 검사다. 현재 설정을 좋은 파인튜닝 설정으로 채택하면 안 된다. 이 결과는 MCP가 학습을 방해한다는 증거가 아니라, 정상적인 gradient 업데이트와 일반화 성능이 별개의 검증 대상임을 보여 준다. test 데이터는 이 검사에서 추론하지 않았다. 학습한 모델은 별도 디렉터리에 남겼으며 원본이나 서비스에 적용하지 않았다.

## 재현과 증거

과거 실행 명령은 제거했다. 보관된 코드는 비실행 .txt 기록이며 재실행하지 않는다.


새 디렉터리가 필요하다. `--pretrained-root`를 생략하면 다운로드 없이 합성 소형 모델만 검사한다. 재현 스크립트는 [verify_mcp_training.py](../archive/gradient_experiments/verify_mcp_training.py.txt), 회귀 테스트는 [test_mcp_training.py](../archive/gradient_experiments/test_mcp_training.py.txt)다.

[원시 검증 보고서](evidence/mcp-training-verification.json)와 `runs/mcp-training-check-003/`에 원본·MCP 패치·학습 모델을 보존했다. 보고서의 `passed`는 파라미터 수정·학습 가능성 검사 통과를 의미하며 일반화 성능 통과가 아니다. 최종 전체 테스트는 **43개 통과**했다.
