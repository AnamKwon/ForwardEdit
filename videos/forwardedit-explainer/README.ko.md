# ForwardEdit 워크플로 설명 영상

[English](README.md) · 한국어

공개용 기본 버전은 **영어판**이며 한국어판도 함께 제공합니다. 두 버전 모두 136초, 1920×1080, 30fps입니다. LLM·MCP 워크플로, Qwen 006의 기록된 점수 상승과 채택 실패, 충분한 데이터로 학습한 파인튜닝과의 비교가 필요함을 설명합니다. 화면 자막이 포함돼 있으며 녹음된 내레이션과 음악은 포함하지 않습니다.

## 미리보기와 편집

```bash
cd videos/forwardedit-explainer
npm ci
npm run dev -- --port=3333
```

공개용 영어판은 **ForwardEdit-English**, 한국어판은 **ForwardEdit-Korean**을 선택합니다. 각 언어의 개별 장면도 별도 폴더에서 편집할 수 있습니다. 텍스트·이미지·장면은 Studio에서 편집할 수 있고, 행렬 셀·측정값 그룹·진행 표시줄은 코드와 데이터에서 계산됩니다. 결과 수치는 `src/recorded-evidence.json`에 묶이며 개별 주장을 따로 편집하지 않습니다.

MP4가 필요하면 Studio의 **Render** 버튼이나 다음 명령을 사용합니다.

```bash
npx remotion render ForwardEdit-English out/forwardedit-en-comparison.mp4
npx remotion render ForwardEdit-Korean out/forwardedit-ko-comparison.mp4
```

이전 `*-results.mp4`·`*-context.mp4`는 교체 전 비교 장면을 포함합니다. 이번 수정본의 파일명은 `*-comparison.mp4`입니다.

사용 형태에 맞는 [Remotion 라이선스](https://www.remotion.dev/docs/license)를 확인하세요. 포함된 글꼴의 고지는 `licenses/`에 있습니다.

## 장면 구성

| 시간        | 장면                                         |
| ----------- | -------------------------------------------- |
| 00:00–00:10 | 프로젝트 소개                                |
| 00:10–00:24 | 외부 LLM, MCP 도구, 대상 모델의 역할         |
| 00:24–00:40 | 실제 가중치·구조·forward 관측 읽기           |
| 00:40–00:54 | 수치 근거, 대안 설명, 반증 조건              |
| 00:54–01:10 | CPU에서 명시적인 값 쓰기와 후보 저장         |
| 01:10–01:22 | 개발 피드백, 후보 이력, 재수정 또는 복원     |
| 01:22–01:34 | 후보 선택 고정과 독립 확인                   |
| 01:34–01:50 | Qwen 006의 기록된 점수 상승과 보존 기준 실패 |
| 01:50–02:06 | 충분한 데이터의 파인튜닝 비교와 아직 필요한 근거 |
| 02:06–02:16 | 정확한 쓰기, 행동 변화, 독립 성능과 한계     |

## 영상이 설명하는 범위

- 행렬, `0.120 → 0.110`, 입력 `+0.50`, API 화면은 **설명용 수치와 인자를 생략한 모의 호출**입니다. 실제 모델의 관측 기록이나 새 실험이 아니며, 이 영상을 위해 모델을 실행하거나 수정하지 않았습니다.
- Qwen 장면은 모의 예시와 구분한 **과거 실제 측정치**입니다. 수치 사본은 `scripts/build_readme_figures.py`로 추출합니다. 파인튜닝 장면은 **아직 측정하지 않은 비교 요건**입니다. 12문장 AdamW 동작 검사는 방법 비교에서 제외하고 [과거 기록](../../docs/RESULTS.ko.md#파인튜닝-참고-비교)으로 보존합니다. [비교 설계](../../docs/FINETUNING_COMPARISON.ko.md)에 충분한 과제 데이터·검증 선택·공통 평가를 명시했습니다. 보관 코드는 실행하지 않았습니다.
- 연결 도구는 `standalone.py`, 기본 FP32 관측·수정 예시는 `server.py`와 `docs/USAGE.md`를 참고했습니다. 실행 환경별 도구와 평가 경로가 다르므로 완전한 실행 기록으로 해석하지 않습니다.
- 고정 입력의 국소 산식만으로 모델 전체 행동을 예측할 수 없습니다. `ready_for_causal_probe`는 제한된 개입 준비 상태이며 인과나 성능의 합격 판정이 아닙니다.
- 명시적 CPU 연산과 추론 전용 모델 실행을 설명합니다. backward, gradient, optimizer, 자동 loss fitting, 좌표·세기 sweep, GPU 학습은 사용하지 않습니다.
- 일반적인 모델 성능 개선이나 WSP 대비 우위를 주장하지 않습니다. 공개된 모든 사례가 하나 이상의 채택 기준을 통과하지 못했다는 한계도 유지합니다.
- AI로 생성한 두 이미지는 개념 일러스트이며 실제 레이어 수, 가중치 배치나 완전한 계산 그래프를 나타내지 않습니다.

출처: [영어 README](../../README.md), [한국어 README](../../README.ko.md), [구조](../../DESIGN.md), [API 사용법](../../docs/USAGE.ko.md), [실행 계약](../../docs/INFERENCE_ONLY_CONTRACT.md), [검토 지침](../../docs/REVIEW_GUIDE.md), 실제 도구 정의. 제작 스토리보드에는 참고 파일의 SHA-256을 기록합니다.

## 결과 장면의 해석

Qwen 006은 새 정답 86개·손실 1개로 순증 85개였고 기존 정답 17/18개를 유지했습니다. 고정 5% 규칙은 해당 실험의 선택이며 분모 18개에서는 사실상 손실 0개만 허용합니다. 규칙 미충족이 정답률 상승을 지우지는 않으며, 원래 프로토콜의 거부 판정은 유지합니다.

파인튜닝도 기존 능력을 망각할 수 있습니다. 양쪽에 같은 목표 성능·기존 정답 보존 평가와 별도의 비대상 과제 확인이 필요합니다. 과거 토큰 정확도 표는 영상에서 제외했습니다. 충분한 데이터로 학습한 파인튜닝과의 성능·보존·비용 비교는 아직 없습니다. [근거와 연구 출처](../../docs/RESULTS.ko.md#채택-판정과-망각을-해석하는-법)를 참고하세요.

## 에셋과 제작 파일

- `public/generated/`: Codex 내장 imagegen 결과를 복사한 이미지. 관측용 일러스트는 알파 채널을 유지합니다.
- `production/asset-log.json`: 생성 프롬프트, 모드, 에셋 경로와 이미지 해시.
- `production/captions.en.srt`, `production/captions.ko.srt`: 각 언어의 자막.
- `production/narration.en.txt`, `production/narration.ko.txt`: 추후 녹음을 위한 내레이션 원고.
- `production/storyboard.en.json`, `production/storyboard.ko.json`: 장면·자막 타이밍과 출처 해시.

장면 타이밍이나 자막을 수정한 뒤 `npm run captions`를 실행합니다. 내보내기 도구는 소스의 정적 JSX를 읽으며 모델이나 생성 코드를 실행하지 않습니다.

## 확인 명령

```bash
npm run lint
npm run build
npm run captions
npm run check:preview
```

미리보기 확인에는 실행 중인 Studio와 설치된 Chrome/Chromium이 필요합니다. 필요한 경우 `FORWARDEDIT_STUDIO_URL`이나 `FORWARDEDIT_CHROME`을 지정합니다. 실제 Studio WebMCP 콜백을 통해 언어별 11개 위치를 확인하고, 브라우저 오류와 스크린샷을 `out/qa/`에 저장합니다. 영상 내보내기나 모델 실험은 실행하지 않습니다.
