# 일반 Codex 사용자의 MCP 실행 경로

`ForwardEdit`는 loop-engineering 없이 실행하는 stdio MCP 서버다. 새 Codex 세션이 프로젝트 스킬을 읽고 직접 수정값을 결정한다. 서버는 명시된 연산을 실행하고 실제 추론 결과를 반환한다. 가중치 변경 자체가 성능 개선을 보장하지는 않는다.

모델·개발 데이터·추론 정책이 준비된 workspace와 프로젝트를 설치한 Python 환경이 필요하다. 준비 항목은 [STANDALONE_MCP.md](STANDALONE_MCP.md)에 설명되어 있다. 독립 test는 이 workspace 밖에 보관한다.

Codex MCP 설정 예시:

```toml
[mcp_servers.forwardedit]
command = "/absolute/model-environment/bin/python"
args = ["-B", "-m", "weight_space_lab.standalone", "--workspace", "/absolute/workspace"]
startup_timeout_sec = 60
tool_timeout_sec = 7200
required = true
tool_input_schema_max_bytes = 50000

# 사용자가 이 workspace의 읽기·직접 편집·추론을 허용한 경우,
# 이 서버에 한해서 사전에 승인한다. 다른 MCP의 정책은 변경하지 않는다.
default_tools_approval_mode = "approve"

[mcp_servers.forwardedit.env]
CUDA_VISIBLE_DEVICES = "0"
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
PYTHONDONTWRITEBYTECODE = "1"
```

위 설정의 Python에 프로젝트가 설치되어 있어야 한다. 소스 설치 대신 모듈 검색 경로를 지정한다면 같은 `env`에 `PYTHONPATH = "/absolute/forwardedit"`을 넣는다. vLLM 사용 시 해당 환경·드라이버·샘플러 설정도 workspace의 고정 정책과 일치시킨다.

비대화형 `approval_policy = "never"`에서는 서버의 기본 승인 정책에 따라 읽기 도구도 거절될 수 있다. 이미 사용자가 승인한 이 MCP의 범위는 위 `default_tools_approval_mode`로 지정한다. 전역 sandbox/인증/관리자 정책을 해제할 필요는 없다. [공식 Codex MCP 설정](https://developers.openai.com/codex/mcp)에서 서버별 승인 항목을 확인할 수 있다.

다른 MCP와 과거 대화가 없는 새 세션에 다음처럼 요청한다.

> forwardedit MCP와 프로젝트 스킬만 사용해 이 모델의 수학 개선을 실험하세요. `read_guidance`로 `skills/weight-space-mcp-only/SKILL.md`를 읽고 `edit_api_schema`, `experiment_info`, `workspace_status`를 확인하세요. 실제 파라미터·계산 구조·forward 관측을 근거로 대상과 수정량을 직접 판단하세요. GPU는 추론에만 사용하고 gradient, optimizer, finite differences, 자동 fitting, 무작위·좌표·세기 sweep은 금지합니다. 수정 전 제안과 반증 조건을 기록하고, 실제 GPU 변경과 같은 입력의 정답률·기존 정답 손실·정답 참조 NLL을 검사하세요. 피드백에 근거한 후속 수정 후 후보 하나를 고정하세요. 독립 test는 고정 이후에만 평가합니다. 셸·파일 편집·다른 MCP는 사용하지 마세요.

Codex가 `functions.exec`를 도구 호출 래퍼로 제공하면 `ALL_TOOLS`에서 `forwardedit` 도구를 찾고 그 MCP만 호출한다. 이 클라이언트에서는 code-mode host를 끄면 등록된 MCP 호출도 불가능할 수 있다. 래퍼 사용 자체와 래퍼 안에서 호출한 실제 MCP를 구분하여 감사한다.

독립 실행 확인에는 클라이언트의 실제 MCP 호출 이벤트와 workspace `mcp-audit`의 요청·응답을 대조한다. 등록 서버가 하나인지, 다른 MCP·셸·파일 편집을 실제로 호출하지 않았는지, 편집값이 LLM이 제출한 계획과 일치하는지 확인한다. 이는 새 세션과 MCP 사용 경로의 검증이며 별도 OS 컨테이너 격리를 뜻하지 않는다.

사용량은 실제 provider `response_id`, 입력·캐시·출력 토큰, 사용자 턴, 도구 호출, 추론 배치를 각각 집계한다. Codex의 저장된 세션에 있는 `token_usage_record`는 실제 응답별 토큰을 제공한다. 출력 토큰에 포함된 reasoning 토큰을 비용에 다시 더하지 않는다. API 상당 금액 계산과 구독 계정의 실제 청구액은 구분한다.

## 검증 시간 관리

소수의 실패·정상 사례에서 먼저 진단하고, 유망한 후보만 고정 validation 전체 예산으로 평가한다. 기존 정답 보존에 실패하면 분석·재수정을 진행하거나 실패로 종료한다. 큰 독립 test는 개발의 정확도·보존 기준을 통과한 후보를 고정한 뒤 실행하는 것이 기본이다. `diagnostic_only` 고정은 큰 평가를 자동 승인하는 뜻이 아니다. 짧은 생성 진단으로 최종 성능을 입증했다고 보고하지 않는다. 최대 토큰 수, 문항 수×모델 수, 실제 생성 토큰과 예상 소요 시간을 평가 전에 명시한다.
