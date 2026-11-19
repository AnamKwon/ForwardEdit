# ForwardEdit

**추론만으로 신경망 가중치를 관측하고 직접 편집·검증하는 MCP·CLI 도구입니다.**

[English](README.md) · [주요 실험과 로그](experiments/README.md) · [현재 구조](DESIGN.md) · [상세 사용법](docs/USAGE.ko.md)

외부 LLM이 실제 가중치와 forward 관측을 읽고 실패 메커니즘을 추론한 뒤, 대상·좌표·계수·새 값을 명시합니다. 도구는 지정된 연산을 CPU에서 계산하고 후보와 근거를 저장합니다. GPU는 추론에만 사용하며 학습이나 손실 자동 피팅을 수행하지 않습니다.

[Weight-Space Programming(WSP)](https://rikiyat.github.io/blog/weight-space-programming/)에 기반한 구현입니다. 현재 기여는 재사용할 도구·편집 절차·검토 가능한 기록입니다. 새로운 편집 알고리즘이나 기존 WSP 대비 성능 우위는 입증하지 않았습니다.

시각적인 설명은 [104초 한국어 워크플로 영상](videos/forwardedit-explainer/README.ko.md)과 공개용 기본 버전인 [영어판](videos/forwardedit-explainer/README.md)에서 미리볼 수 있습니다. 두 버전 모두 자막을 포함한 편집 가능한 Remotion 구성입니다. 화면의 수치와 호출은 설명용이며 실제 모델 실험 녹화가 아닙니다.

## 제공하는 기능

- 실제 parameter 값·축·소유 모듈과 로딩된 forward 소스 관측
- 토큰별 부호 있는 활성·기여와 여러 모듈의 흐름 추적
- 명시적 값 교체, 국소 감쇠·투영·rank-one 편집과 변경 예산 검사
- 원본·부모·후보 비교, 피드백에 근거한 재수정과 값 복원
- 개발 평가, 후보·설정 해시 고정과 별도 독립 확인
- Codex·Claude 등 외부 클라이언트에서 사용할 MCP와 편집 스킬

가중치 파일이 바뀐 사실, 예상한 행동이 바뀐 사실, 독립 성능이 개선된 사실을 구분합니다. 설명이 그럴듯하거나 소프트웨어 테스트가 통과했다는 이유로 모델 개선을 주장하지 않습니다.

## 실제 사례부터 보기

아래는 종료된 로컬 실험의 주요 기록을 공개한 것입니다. 새 실험을 실행한 결과가 아닙니다. 수록한 모든 실험은 사전 기준 중 하나 이상을 통과하지 못했으며, 전체 합격한 모델 개선 사례로 소개하지 않습니다.

| 사례 | 기록된 결과 | 참고할 내용 |
|---|---|---|
| [직접 값 수정 001](experiments/direct-values-inference-only-001/README.md) | 후보별 6개 값 수정, 두 후보 모두 거부 | 실제 old/new 숫자와 MCP·CPU/CUDA 확인 |
| [TinyStories 004](experiments/skill-edit-experiment-004/README.md) | 독립 반복률 7.279% → 7.081%, 신뢰구간이 0 포함 | 개발 결과와 독립 결과의 차이 |
| [TinyStories 005](experiments/failure-repair-005/README.md) | 독립 반복률 6.060% → 6.512%, 채택 안 함 | 누적 재수정·반대 대조·정상 사례 손상 |
| [원문 참고 Qwen 006](experiments/extended-discovery-006/README.md) | 독립 정답 18 → 103 / 192, 보존 기준 실패 | 큰 점수 증가와 전체 채택 실패가 함께 존재 |
| [MCP 전용 Qwen 009–014](experiments/independent-mcp-014/README.md) | 개발 5 → 10 / 64, 독립 25 → 22 / 192 | 연결·승인 실패, 실제 피드백 재수정 1회와 음성 결과 |

006은 원문의 개입을 참고한 호스트 분기이며 독립적인 재발견 실험이 아닙니다. 정답 증가는 기록된 원본·디코딩 설정 대비 결과입니다. 최강 무편집 설정 대비 우위는 확인하지 않았습니다. 실험 간 모집단·조건이 달라 직접 성능 비교나 합산도 할 수 없습니다. 각 사례의 **Unaddressed criticisms / 미대응 비판**을 함께 읽어주세요.

[공개 기록 안내](experiments/README.md)에 선정 기준, 가린 정보와 제외한 파일을 적었습니다. 모델·GPU 없이 공개 사본의 해시와 JSON 형식만 확인할 수 있습니다.

```bash
python3 scripts/verify_public_evidence.py
```

이 검사는 공개 파일의 일관성을 확인합니다. 추론 재현, 공개하지 않은 원본의 진위, 성능·인과 가설을 인증하지 않습니다.

## 설치

현재 제공 형태는 **로컬 stdio MCP 서버와 저장소에 포함된 스킬**입니다. 이 저장소에는 Codex·Claude 마켓플레이스에서 설치할 플러그인 묶음이나 호스팅된 MCP URL이 없습니다. Python 소스를 설치한 뒤 사용 중인 클라이언트에 서버를 등록합니다. 스킬은 MCP의 `read_guidance`로 읽을 수 있어 별도 설치가 필수는 아닙니다.

Python 3.11 이상과 소스 checkout이 필요합니다. MCP가 저장소의 스킬과 계약 문서를 읽으므로 editable 설치를 사용합니다.

```bash
git clone https://github.com/AnamKwon/ForwardEdit.git
cd ForwardEdit
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[lm,mcp,test]'
.venv/bin/forwardedit --help
.venv/bin/forwardedit-mcp --help
```

이미 소스를 받은 경우 해당 디렉터리에서 가상환경 생성부터 진행합니다. 위 명령은 GitHub 소스 설치이며 PyPI 배포를 전제로 하지 않습니다.

이 명령은 Transformers 4.x 소형 모델 경로를 설치합니다. Qwen/research 경로는 별도 환경에서 `pip install -e '.[qwen]'`을 사용하고, 추론 장치에 맞는 PyTorch를 선택합니다. `lm`과 `qwen`은 Transformers 버전 제약이 달라 같은 환경에 함께 설치하지 않습니다. 선택적 vLLM은 별도의 호환 실행 환경을 명시합니다. [환경과 workspace 준비](docs/STANDALONE_MCP.ko.md)를 참고하세요.

배포 이름은 `forwardedit`, Python import와 모듈 경로는 `weight_space_lab`입니다. 기존 `weight-space`·`weight-space-mcp` 명령은 호환 별칭으로 유지합니다.

## LLM 연결과 실행

받은 `ForwardEdit` 디렉터리에서 소스 경로와 workspace를 지정합니다. workspace 예시 경로는 실제로 쓸 수 있는 경로로 바꿉니다.

```bash
FORWARDEDIT_ROOT="$(pwd)"
FORWARDEDIT_WORKSPACE="/absolute/path/to/experiment"
mkdir -p "$FORWARDEDIT_WORKSPACE"
```

**Codex CLI 등록:**

```bash
codex mcp add forwardedit --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 -- \
  "$FORWARDEDIT_ROOT/.venv/bin/forwardedit-mcp" --workspace "$FORWARDEDIT_WORKSPACE"
codex mcp get forwardedit
```

**Claude Code 등록:**

```bash
claude mcp add --transport stdio --scope user forwardedit -- \
  "$FORWARDEDIT_ROOT/.venv/bin/forwardedit-mcp" --workspace "$FORWARDEDIT_WORKSPACE"
claude mcp get forwardedit
```

사용하는 클라이언트의 명령을 선택합니다. `mcp get`은 저장된 등록 정보를 확인합니다. 새 클라이언트 세션에서 `edit_api_schema`와 `read_guidance(path="skills/weight-space-mcp-only/SKILL.md")`를 호출하도록 요청하면 실제 연결과 스킬 읽기를 확인할 수 있습니다. 이 두 호출은 모델 추론을 하지 않습니다.

연결 확인에는 비어 있는 쓰기 가능한 디렉터리로 충분합니다. 실제 모델 실험은 로컬 `model/`, 개발 데이터와 [workspace 설정](docs/STANDALONE_MCP.ko.md)이 필요합니다. 모델과 데이터는 자동 다운로드하지 않습니다. `experiment_info`는 research workspace 설정까지 준비한 뒤 호출합니다.

파일로 설정하는 클라이언트는 [Claude 형식 JSON](examples/mcp/claude.json)과 [Codex TOML](examples/mcp/codex.toml)의 경로를 실제 환경에 맞게 바꿉니다. [MCP 실행 안내](docs/STANDALONE_MCP.ko.md), [Codex 연결 안내](docs/CODEX_MCP_CLIENT.ko.md)에 상세 조건이 있습니다. stdio 서버를 터미널에서 직접 실행하면 클라이언트 요청을 기다리며, 웹 화면은 표시하지 않습니다.

편집기는 `read_guidance`로 `skills/weight-space-mcp-only/SKILL.md`를 읽고 `edit_api_schema`, `experiment_info`부터 확인합니다. 실제 관측으로 수정값을 판단하고 개입 전에 예측을 적습니다. 독립 데이터는 후보 고정 후에만 사용합니다. 사례집의 해법을 편집기에게 제공했다면 원문 참고 실험으로 표시해야 합니다.

CLI의 기본 흐름은 다음과 같습니다.

```bash
.venv/bin/forwardedit --workspace /absolute/path/to/experiment inspect model/model.safetensors
.venv/bin/forwardedit --workspace /absolute/path/to/experiment preview plan.json
.venv/bin/forwardedit --workspace /absolute/path/to/experiment apply plan.json candidate-01
```

`plan.json`은 호출자가 작성한 명시적 편집입니다. 자동 최적화 요청이 아닙니다. [수치 예제와 평가 API](docs/USAGE.ko.md)를 참고하세요.

## 지원 범위와 실행 계약

기본 경로는 로컬 `config.json`과 단일 `model.safetensors`, CPU/CUDA FP32 관측과 CPU FP32 paired NLL 확인입니다. 별도 research 경로는 모델·데이터·디코딩 해시와 예산으로 묶인 BF16 추론을 지원합니다. 정밀도·지원 모델·평가 범위를 구분해야 합니다.

양자화·sharding·원격 모델 코드·자동 다운로드는 지원하지 않습니다. 공유/tied weight 편집에는 제한이 있습니다. 후보는 replacement tensor와 manifest이며 완전한 Hugging Face checkpoint가 아닙니다.

[실행 계약](docs/INFERENCE_ONLY_CONTRACT.md)은 backward·autograd·gradient 수집·수치 미분·optimizer/LoRA 학습·손실 자동 피팅·무작위/좌표/세기 sweep을 금지합니다. LLM이 문항·층·계수·산식을 명시한 1회 다중 오차 CPU 갱신과 제한된 반증 개입은 허용하지만 least-squares/pinv solve는 금지합니다.

모델은 eval/frozen/inference 상태로 실행합니다. 런타임과 테스트에 훈련 API 차단 검사가 있지만 OS sandbox나 임의 외부 코드의 통제를 의미하지 않습니다.

## 파일과 코딩 에이전트

- `weight_space_lab/`, `tests/`: 구현과 검증
- `skills/`: 실제 가중치 편집 LLM의 제품 사용 지침
- `AGENTS.md`, `DESIGN.md`: 저장소 유지보수 지침과 현재 구현 구조
- `docs/`: 계약·사용법·리뷰 기준과 과거 보고서
- `experiments/`: 선정한 공개 실험 기록
- `examples/mcp/`: 개인 정보 없는 공유 설정 예제
- `archive/gradient_experiments/`: 현재 방식의 성공 근거에서 제외한 과거 비실행 이력

개인 에이전트 설정, 인증 정보, 가상환경, 로그, 체크포인트와 전체 `runs/`는 Git에서 제외합니다. 공개 사례는 별도로 선정한 사본이며, 원래 외부 저장소 경로에 있던 로컬 실험도 포함합니다. 모델·대형 활성·전체 데이터셋은 포함하지 않습니다. 공개된 평가 문항은 이후 새로운 독립 test로 사용할 수 없습니다.

## 검증과 출처

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider
python3 scripts/verify_public_evidence.py
```

테스트에서도 fixture를 학습하지 않으며 skip을 통과와 구분합니다. 문서·증거 정리를 위해 모델 실험을 재실행할 필요는 없습니다.

원래 WSP 연구와 [공개 로그](https://github.com/RikiyaT/lm_understand-logs)를 방법론 출처로 명시합니다. [원문 조사와 구현 구분](docs/WEIGHT_SPACE_SKILL_DESIGN.md)을 참고하세요. 외부 모델·데이터의 이용 조건은 해당 원출처를 따르며, 이 기록의 공개가 checkpoint나 전체 데이터셋의 재배포·재라이선스를 뜻하지 않습니다.
