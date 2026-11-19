# MCP만으로 직접 가중치 편집하기

이 경로는 loop-engineering 또는 다른 MCP에 의존하지 않는다. 준비된 모델·개발 데이터가 있는 workspace를 ForwardEdit MCP에 연결하고, 새 Codex/Claude 세션이 가중치 분석·수정값 결정·후보 저장·추론·피드백 재수정을 진행한다. gradient/optimizer 학습은 제공하지 않는다. 성능 개선은 실제 독립 평가로 따로 확인해야 한다.

프로젝트의 MCP 실행 명령은 다음과 같다. 환경에 패키지를 설치한 경우 `forwardedit-mcp --workspace /absolute/workspace`도 같은 진입점을 사용한다. 기존 `weight-space-mcp` 명령은 호환 별칭이다.

```
/path/to/model-environment/python -m weight_space_lab.standalone --workspace /absolute/workspace
```

`python` 환경은 프로젝트 패키지와 MCP, 실제 대상 모델을 지원하는 PyTorch/Transformers를 포함해야 한다. Qwen3.5는 기존 Transformers 4.x 소형 모델 환경과 구분한다. vLLM을 사용할 경우 workspace의 `research.json`에 해당 별도 실행 환경을 고정한다. 서버의 GPU 접근은 MCP 클라이언트가 아닌 서버 실행 환경을 따른다.

Claude의 사용자 MCP 설정 예시:

```json
{
  "mcpServers": {
    "forwardedit": {
      "command": "/path/to/model-environment/python",
      "args": ["-m", "weight_space_lab.standalone", "--workspace", "/absolute/workspace"],
      "env": {"CUDA_VISIBLE_DEVICES": "0"},
      "timeout": 7200000
    }
  }
}
```

워크스페이스에는 원본 `model/`, `practice.json`의 `discovery`/`validation`, 원본·데이터·정밀도·추론 정책·예산을 해시로 고정한 `research.json`이 필요하다. 선택적인 `protocol.json`은 검증 ID·편집 한도·선택 기준을 명시한다. 독립 test는 편집자 MCP workspace 밖에 둔다. 이 실행 경로는 자동 다운로드나 데이터 준비를 대신하지 않는다.

세션에는 다음처럼 요청할 수 있다.

> ForwardEdit MCP만 사용하세요. `read_guidance`로 `skills/weight-space-mcp-only/SKILL.md`를 읽고, 이 workspace 모델의 수학 성능 개선을 시도하세요. 실제 가중치와 forward를 분석해 대상·좌표·수정량을 직접 판단하고, 근거와 반증 조건을 기록하세요. gradient, optimizer, finite differences, 자동 fitting이나 좌표/계수 sweep은 금지합니다. 정상 정답 손실도 검사하고, 후보를 고정하기 전에는 독립 test를 사용하지 마세요.

셸 없는 클라이언트도 `array_compute`, `prepare_edit_from_arrays`, `create_candidate_from_arrays`, `record_observation`, `forward_and_wait`로 연산·초기 분석·편집·기록·추론을 수행할 수 있다. 원래 프로젝트의 후보 작성기와 research inference를 호출하며 자동으로 편집값을 고르지 않는다. 명시적 수치 계획, 해시, 호출·오류, 실제 추론 결과가 workspace에 남는다. MCP 경로 제한은 등록된 도구에만 적용되므로, MCP 단독 실행을 검증하려면 클라이언트 셸/파일 도구와 다른 MCP를 별도로 끈다.

스킬은 MCP 등록만으로 자동 주입되지 않는다. 클라이언트에 로컬 스킬을 설치하거나, 위와 같이 `read_guidance`로 본문을 실제로 읽는다. `freeze_candidate`는 후보 선택과 해시만 기록하며 성능 합격을 승인하지 않는다. 독립 평가에서 정답률, 기존 정답 손실, 불확실성까지 통과했을 때만 개선을 보고한다.

MCP SDK 1.x/2.x는 같은 프로젝트 API를 제공한다. Qwen용 모델/분석 환경은 `pip install -e ".[qwen]"`로 소스 설치할 수 있다. 먼저 GPU 드라이버에 맞는 PyTorch를 설치하고 버전을 고정한다. vLLM은 자체 PyTorch/Transformers 제약이 있으므로 별도 환경을 사용한다. 스킬은 이 프로젝트 소스에서 읽으므로 소스 설치 경로를 유지한다. 최신 vLLM 릴리스가 현재 드라이버의 CUDA 계열을 지원하는지 확인하고, 실제 MCP 원본 추론 제어를 통과한 호환 버전을 기록한다.

편집을 작성하기 전에 `edit_api_schema`로 실제 JSON 필수 필드와 수식을 읽는다. 도구 카탈로그에도 plan의 구조가 포함된다. `scale.strength`는 제거 비율이므로 `W_new=(1-strength)*W`다. 숫자 참조 계획은 `preview_edit_from_arrays`로 확인한다. 오류 응답은 필요한 필드·인자 경로와 실제 원인을 반환한다.

중단된 같은 실험을 이어갈 때는 `workspace_status`와 `list_workspace_artifacts`로 기존 결과와 노트를 찾는다. 성공한 추론 결과를 다시 뽑아 선택하지 않는다. `research.json`·ledger·snapshot은 유지해야 누적 예산과 provenance가 보존된다. SDK 내부 턴, 실제 모델 응답 ID 수, 도구 호출 수는 다른 지표이므로 모두 측정한다.
