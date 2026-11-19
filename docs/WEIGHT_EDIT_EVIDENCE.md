# 실제 가중치 편집에서 확인한 것

> 과거 실행·조사 기록입니다. 본문의 테스트 수·버전·지원 범위는 당시 상태이며 현재 릴리스와 다를 수 있습니다. [공개 사례와 발췌 로그](../experiments/README.md), [현재 구조](../DESIGN.md)를 함께 참고하세요. 공개하지 않은 `runs/` 자료는 로컬 전용으로 표시했습니다. Gradient/optimizer 이력은 현재 방식의 성공 근거에서 제외합니다.

2026-11-04에 원문과 공개 실행 기록을 읽고 정리했다. 외부 연구 결과와 이 저장소에서 직접 실행한 결과는 구분한다. 업데이트 방식 비교는 [UPDATE_METHODS.md](UPDATE_METHODS.md), 실험 결과는 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)를 참조한다.

## 원문의 편집은 무엇을 바꾸었나

[원문](https://rikiyat.github.io/blog/weight-space-programming/)의 Experiment 8과 [공개 기록](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math)을 확인했다. 특히 `runs/p50c/tool_calls.jsonl`, `runs/p50c/final_answer.md`, `code/apply_edit.py`, `code/make_edit_diff.py`를 읽었다. 전체 논문의 독립 재현을 수행한 것은 아니다.

Qwen의 MLP 출력 행렬은 `W[out, hidden]`이다. 선택한 hidden neuron의 열 집합을 `J`, 관측한 실패 방향의 단위벡터를 `u`라고 하면 편집은 다음과 같다.

```text
c = uᵀ W[:, J]
ΔW[:, J] = -α · outer(u, c)
W'[:, J] = W[:, J] + ΔW[:, J]
```

새 optimizer나 학습 루프가 가중치를 결정한 것이 아니다. LLM이 입력의 실패 양상, matched context의 hidden state/MLP activation, 개입 결과를 보고 방향·위치·세기를 선택하고 대체 텐서를 저장했다. 공개 최종 기록에는 layers 15·19의 선택 열에 α=6을 적용한 식이 남아 있다. **α>1은 단순한 직교 투영 제거를 넘어 해당 방향 성분을 반전시킬 수 있다.** 초기 도구는 strength≤1이었으나 현재 `project_out`/`rank_one`은 유한한 비음수 계수를 명시하도록 확장했다. 수치 표현을 지원하는 것이 원문의 가중치·BF16 평가나 성능 재현을 뜻하지 않는다. 자체 예산과 독립 평가가 필요하다.

로그에서 중요한 점은 특정 레이어 번호보다 절차다. 실패·성공에 가까운 문맥을 비교해 방향 가설을 만들고, 작은 뉴런 개입·레이어 분리·세기 변화·크기를 맞춘 무작위 방향으로 반증을 시도한 다음, 편집 해시를 고정하고 별도 문제에서 확인했다. 큰 activation이나 큰 가중치 norm 자체는 실패의 원인이라는 증거가 아니다.

## 저장 파일과 추론 모델 사이의 확인

[복원 코드](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py)는 원본 텐서 해시를 확인하고 저장된 변경을 적용한 후 최종 텐서 해시도 확인한다. 원 연구는 평가 dtype의 반올림까지 다루었다. 우리 도구의 현재 추론 범위는 **CPU/CUDA FP32, 로컬 단일 safetensors 체크포인트**이며 BF16 원문 재현과 구분한다.

우리의 확인 경로는 `plan → core.prepare → replacements.safetensors → verify_candidate → apply_candidate → 실제 forward → token NLL/accuracy`다. plan을 재계산한 값과 저장 패치가 일치해야 하고, 모델 안의 실제 파라미터에도 같은 값이 들어가야 한다. 테스트는 tensor 일치뿐 아니라 같은 입력의 logits 변화도 확인한다. tied embedding/head 편집은 명시적으로 거부한다. 로딩 중 누락된 파라미터를 무작위로 초기화한 모델은 비교 대상으로 받아들이지 않는다.

## 파인튜닝 구조에서 가져올 것

이 저장소에는 기존 fine-tuning trainer가 없었다. 기반 Transformer의 실제 module class, forward 입출력, checkpoint tensor를 확인하는 방식으로 구조를 조사한다. `torch.nn.Linear`는 `[out,in]`, GPT-2 `Conv1D`는 `[in,out]`이므로 같은 `c_proj` 이름에서도 뉴런 선택 축이 달라질 수 있다. bias와 residual 경로도 남아 있다.

AdamW는 손실 gradient와 optimizer 상태로 업데이트를 계산한다. LoRA는 base를 고정하고 낮은 rank의 두 행렬을 학습한다. 직접 편집은 LLM이 명시한 delta/연산을 적용한다. 행렬 delta가 낮은 rank라는 이유만으로 LoRA 학습이라고 부르지 않는다. 현재 사용자 지시와 원문 프롬프트에 따라 gradient 진단도 금지한다. [현재 실행 계약](INFERENCE_ONLY_CONTRACT.md)을 따른다. 과거 gradient 결과는 이 방식의 성공 근거에서 제외한다.

## 다음 편집의 판단 기준

- 관측한 실패와 정상 사례, 사용한 토큰 영역, module layout, direction의 출처를 함께 기록한다.
- 추론 진단을 읽고 가설을 세운 뒤 기존 원본에서 작은 후보를 분기한다. identity·무작위/반대 개입으로 단순한 감쇠 효과인지 확인한다.
- validation까지만 사용해 후보를 고르고 해시를 고정한다. 최종 확인 결과를 본 뒤에는 같은 test로 재선택하지 않는다.
- 현재 NLL 평가는 teacher forcing이다. 생성 반복 감소·수학 정답률·장문 추론 능력 향상은 별도 generation benchmark 없이는 주장하지 않는다.
- 개선이 없으면 결과를 남기고 중단한다. 가중치 편집이 없는 능력을 만들어 준다고 가정하지 않는다.
