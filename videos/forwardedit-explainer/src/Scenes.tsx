import {
  CanvasImage,
  Img,
  Interactive,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Arrow, Caption, Header, Matrix, Panel, ToolLine } from "./Visuals";

const IntroInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <CanvasImage
        name="가중치 구조 일러스트"
        src={staticFile("generated/weight-sculpture.png")}
        style={{
          position: "absolute",
          width: 1920,
          height: 1080,
          objectFit: "cover",
          opacity: interpolate(frame, [0, fps], [0, 1], {
            extrapolateRight: "clamp",
          }),
          scale: interpolate(frame, [0, 10 * fps], [1.08, 1], {
            extrapolateRight: "clamp",
          }),
        }}
      />
      <Header step="00 / INTRODUCTION" />
      <Interactive.Div
        name="소개 분류"
        style={{
          position: "absolute",
          left: 110,
          top: 194,
          fontSize: 26,
          letterSpacing: 5,
          color: "#3158D5",
          fontWeight: 700,
        }}
      >
        A LOOK INSIDE MODEL EDITING
      </Interactive.Div>
      <Interactive.H1
        name="프로젝트 이름"
        style={{
          position: "absolute",
          left: 103,
          top: 220,
          margin: 0,
          fontSize: 148,
          fontWeight: 700,
          letterSpacing: -7,
          color: "#182638",
          opacity: interpolate(frame, [0, fps], [0, 1], {
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [0, fps], ["0px 32px", "0px 0px"], {
            extrapolateRight: "clamp",
          }),
        }}
      >
        ForwardEdit
      </Interactive.H1>
      <Interactive.Div
        name="소개 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 414,
          fontSize: 82,
          lineHeight: 1.35,
          fontWeight: 700,
          letterSpacing: -4,
          color: "#182638",
          opacity: interpolate(frame, [fps, 2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        가중치를 읽고,
        <br />
        직접 고치고,
        <br />
        다시 확인하다.
      </Interactive.Div>
      <Interactive.Div
        name="소개 설명"
        style={{
          position: "absolute",
          left: 115,
          top: 795,
          fontSize: 29,
          color: "#68717A",
        }}
      >
        LLM × MCP TOOLS × FORWARD OBSERVATIONS
      </Interactive.Div>
      <Caption name="소개 자막 1" from={fps} durationInFrames={4 * fps}>
        LLM은 모델 안의 숫자와 실제 추론 결과를 읽습니다.
      </Caption>
      <Caption name="소개 자막 2" from={5 * fps} durationInFrames={5 * fps}>
        ForwardEdit는 그 판단을 명시적인 수정과 검증으로 연결합니다.
      </Caption>
    </>
  );
};
export const IntroScene = Interactive.create({
  Component: IntroInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const RolesInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="01 / ROLES" />
      <Interactive.H1
        name="역할 구분 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 96,
          letterSpacing: -4,
          color: "#182638",
        }}
      >
        판단은 LLM이. 실행은 도구가.
      </Interactive.H1>
      <Interactive.Div
        name="역할 구분 부제"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        가중치를 편집하는 모델과, 편집 대상인 모델을 구분합니다.
      </Interactive.Div>
      <Interactive.Div
        name="외부 LLM 역할"
        style={{
          position: "absolute",
          left: 110,
          top: 405,
          width: 500,
          height: 345,
          borderRadius: 20,
          backgroundColor: "#3158D5",
          color: "white",
          padding: 38,
          boxSizing: "border-box",
          opacity: interpolate(frame, [0, fps], [0, 1], {
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ fontSize: 25, opacity: 0.65, letterSpacing: 3 }}>
          01 / DECIDE
        </div>
        <div style={{ fontSize: 63, marginTop: 24, fontWeight: 700 }}>
          외부 LLM
        </div>
        <div style={{ fontSize: 32, marginTop: 23, lineHeight: 1.6 }}>
          관측 해석 · 원인 가설
          <br />
          좌표 · 계수 · 새 값 지정
        </div>
      </Interactive.Div>
      <Arrow x={630} y={540} width={90} />
      <Interactive.Div
        name="MCP 도구 역할"
        style={{
          position: "absolute",
          left: 740,
          top: 405,
          width: 490,
          height: 345,
          borderRadius: 20,
          backgroundColor: "#E7E9E3",
          color: "#182638",
          padding: 38,
          boxSizing: "border-box",
          opacity: interpolate(frame, [fps, 2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ fontSize: 25, color: "#68717A", letterSpacing: 3 }}>
          02 / EXECUTE
        </div>
        <div style={{ fontSize: 63, marginTop: 24, fontWeight: 700 }}>
          MCP 도구
        </div>
        <div style={{ fontSize: 32, marginTop: 23, lineHeight: 1.6 }}>
          정해진 CPU 연산
          <br />
          후보 저장 · 기록 보존
        </div>
      </Interactive.Div>
      <Arrow x={1250} y={540} width={90} />
      <Interactive.Div
        name="로컬 모델 역할"
        style={{
          position: "absolute",
          left: 1360,
          top: 405,
          width: 450,
          height: 345,
          borderRadius: 20,
          backgroundColor: "#DCEBE5",
          color: "#182638",
          padding: 38,
          boxSizing: "border-box",
          opacity: interpolate(frame, [2 * fps, 3 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ fontSize: 25, color: "#3D9B92", letterSpacing: 3 }}>
          03 / OBSERVE
        </div>
        <div style={{ fontSize: 61, marginTop: 24, fontWeight: 700 }}>
          대상 모델
        </div>
        <div style={{ fontSize: 32, marginTop: 23, lineHeight: 1.6 }}>
          로컬 가중치 + 후보
          <br />
          실제 forward 추론
        </div>
      </Interactive.Div>
      <Interactive.Div
        name="피드백 되돌림"
        from={4 * fps}
        style={{
          position: "absolute",
          left: 120,
          top: 797,
          fontSize: 29,
          color: "#3158D5",
        }}
      >
        ← 도구가 돌려준 관측을 읽고, LLM이 다음 판단을 내립니다.
      </Interactive.Div>
      <Caption name="역할 자막 1" durationInFrames={5 * fps}>
        외부 LLM이 무엇을 왜 바꿀지 결정하고, MCP로 도구를 호출합니다.
      </Caption>
      <Caption name="역할 자막 2" from={5 * fps} durationInFrames={5 * fps}>
        도구는 지정된 연산을 계산합니다. GPU는 모델의 추론에만 사용합니다.
      </Caption>
      <Caption name="역할 자막 3" from={10 * fps} durationInFrames={4 * fps}>
        도구 안의 자동 최적화기가 수정값을 선택하는 구조가 아닙니다.
      </Caption>
    </>
  );
};
export const RolesScene = Interactive.create({
  Component: RolesInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const ObserveInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="02 / READ & OBSERVE" dark />
      <Interactive.H1
        name="관측 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 96,
          letterSpacing: -4,
          color: "#F4F2ED",
        }}
      >
        먼저, 실제 모델을 읽습니다.
      </Interactive.H1>
      <Interactive.Div
        name="관측 부제"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#A9B9CB",
        }}
      >
        추측한 구조 대신 실제 값 · 축 · forward를 확인합니다.
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 110,
          top: 398,
          width: 1000,
          padding: 34,
          boxSizing: "border-box",
          background: "#111D2D",
          border: "1px solid #526174",
          borderRadius: 18,
        }}
      >
        <div
          style={{
            fontSize: 23,
            letterSpacing: 3,
            color: "#8596AD",
            marginBottom: 20,
          }}
        >
          LLM → MCP / SELECTED CALLS
        </div>
        <Interactive.Div
          name="접속 도구"
          style={{
            opacity: interpolate(frame, [0, fps], [0, 1], {
              extrapolateRight: "clamp",
            }),
          }}
        >
          <ToolLine color="#9FB6FF">
            read_guidance() · edit_api_schema()
            <br />
            experiment_info()
          </ToolLine>
        </Interactive.Div>
        <Interactive.Div
          name="구조 관측 도구"
          style={{
            opacity: interpolate(frame, [2 * fps, 3 * fps], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <ToolLine color="#9FB6FF">
            describe_model_structure()
            <br />
            inspect_module_computation()
          </ToolLine>
        </Interactive.Div>
        <Interactive.Div
          name="값과 추론 관측 도구"
          style={{
            opacity: interpolate(frame, [4 * fps, 5 * fps], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <ToolLine color="#8BCABB">
            describe_parameter() · probe_model()
            <br />
            analyze_parameter_contexts()
          </ToolLine>
        </Interactive.Div>
      </div>
      <Img
        name="관측용 돋보기 일러스트"
        src={staticFile("generated/inspection-cutout.png")}
        style={{
          position: "absolute",
          left: 1135,
          top: 370,
          width: 670,
          height: 520,
          objectFit: "contain",
          opacity: interpolate(frame, [fps, 2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(
            frame,
            [fps, 3 * fps],
            ["40px 25px", "0px 0px"],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
          ),
        }}
      />
      <Caption name="관측 자막 1" durationInFrames={5 * fps}>
        처음에는 계약, API 형식, 모델과 개발 데이터의 설정을 확인합니다.
      </Caption>
      <Caption name="관측 자막 2" from={5 * fps} durationInFrames={5 * fps}>
        그다음 실제 모듈과 가중치 값, 토큰 위치별 추론 관측을 읽습니다.
      </Caption>
      <Caption name="관측 자막 3" from={10 * fps} durationInFrames={6 * fps}>
        실패 문맥과 정상 문맥의 부호 있는 기여를 비교해 수정 근거를 찾습니다.
      </Caption>
    </>
  );
};
export const ObserveScene = Interactive.create({
  Component: ObserveInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const HypothesisInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="03 / FORM A HYPOTHESIS" />
      <Interactive.H1
        name="가설 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 96,
          letterSpacing: -4,
          color: "#182638",
        }}
      >
        숫자에서, 검증할 가설로.
      </Interactive.H1>
      <Interactive.Div
        name="가설 부제"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        큰 활성 하나만으로 의미나 원인을 단정하지 않습니다.
      </Interactive.Div>
      <Panel
        style={{
          position: "absolute",
          left: 110,
          top: 400,
          width: 830,
          height: 380,
          boxSizing: "border-box",
        }}
      >
        <Interactive.Div
          name="국소 산식 분류"
          style={{ fontSize: 24, color: "#3158D5", letterSpacing: 2 }}
        >
          ILLUSTRATIVE LOCAL CONTRIBUTION
        </Interactive.Div>
        <Interactive.Div
          name="기여 산식"
          style={{
            fontFamily: "Code",
            fontSize: 49,
            marginTop: 28,
            color: "#182638",
          }}
        >
          +0.50 × 0.120 = +0.060
        </Interactive.Div>
        <Interactive.Div
          name="변경량 산식"
          from={2 * fps}
          style={{
            fontFamily: "Code",
            fontSize: 45,
            marginTop: 30,
            color: "#E77751",
          }}
        >
          Δw = −0.010 → Δ = −0.005
        </Interactive.Div>
        <Interactive.Div
          name="가설 수치 한계"
          style={{
            fontSize: 28,
            color: "#68717A",
            marginTop: 28,
            lineHeight: 1.5,
          }}
        >
          입력이 고정된 국소 산식 예시입니다.
          <br />이 계산만으로 모델의 행동을 예측할 수는 없습니다.
        </Interactive.Div>
      </Panel>
      <Arrow x={970} y={560} width={90} />
      <Panel
        style={{
          position: "absolute",
          left: 1095,
          top: 400,
          width: 715,
          height: 380,
          boxSizing: "border-box",
          borderColor: "#3158D5",
          opacity: interpolate(frame, [3 * fps, 4 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <Interactive.Div
          name="가설 검토 항목"
          style={{
            fontSize: 36,
            color: "#182638",
            fontWeight: 700,
            lineHeight: 1.85,
          }}
        >
          왜 이 좌표와 새 값인가?
          <br />
          다른 설명은 없는가?
          <br />
          무엇이 나오면 가설이 틀리는가?
        </Interactive.Div>
        <Interactive.Div
          name="제안 준비 도구"
          style={{
            fontFamily: "Code",
            fontSize: 26,
            color: "#3158D5",
            marginTop: 20,
          }}
        >
          prepare_edit_proposal()
        </Interactive.Div>
      </Panel>
      <Caption name="가설 자막 1" durationInFrames={5 * fps}>
        LLM은 좌표와 변경량을 고른 이유, 대안 설명, 반증 조건을 명시합니다.
      </Caption>
      <Caption name="가설 자막 2" from={5 * fps} durationInFrames={5 * fps}>
        근거가 부족하면 defer. 준비가 되면 제한된 개입으로 확인합니다.
      </Caption>
      <Caption name="가설 자막 3" from={10 * fps} durationInFrames={4 * fps}>
        ready_for_causal_probe는 개입 준비 상태이며, 인과나 성능의 합격 판정은
        아닙니다.
      </Caption>
    </>
  );
};
export const HypothesisScene = Interactive.create({
  Component: HypothesisInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const WriteInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="04 / EXPLICIT CPU EDIT" />
      <Interactive.H1
        name="수정 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 96,
          letterSpacing: -4,
          color: "#182638",
        }}
      >
        LLM이 지정한 값만, 정확히.
      </Interactive.H1>
      <Interactive.Div
        name="수정 부제"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        원본을 유지하고, 명시적인 후보를 별도로 만듭니다.
      </Interactive.Div>
      <Matrix
        name="설명용 가중치 행렬"
        edited
        style={{ position: "absolute", left: 110, top: 400, width: 650 }}
      />
      <Interactive.Div
        name="수정 좌표"
        style={{
          position: "absolute",
          left: 110,
          top: 875,
          fontFamily: "Code",
          fontSize: 26,
          color: "#3158D5",
        }}
      >
        illustrative tensor W · coordinate [0, 1]
      </Interactive.Div>
      <Arrow x={805} y={560} width={120} />
      <div
        style={{
          position: "absolute",
          left: 980,
          top: 400,
          width: 830,
          height: 438,
          background: "#182638",
          padding: 36,
          borderRadius: 18,
          boxSizing: "border-box",
        }}
      >
        <Interactive.Pre
          name="직접 값 수정 호출 요약"
          style={{
            margin: 0,
            color: "#DCE6F2",
            fontFamily: "Code",
            fontSize: 27,
            lineHeight: 1.8,
            whiteSpace: "pre-wrap",
          }}
        >
          {
            "write_parameter_values(\n  rows=[0], columns=[1],\n  expected_values=[[0.120]],\n  values=[[0.110]],\n  ...\n)"
          }
        </Interactive.Pre>
        <Interactive.Div
          name="수정 전 검사"
          from={6 * fps}
          style={{ color: "#8BCABB", fontSize: 26, marginTop: 14 }}
        >
          source hash · old value · 변경 예산 검사
        </Interactive.Div>
      </div>
      <Interactive.Div
        name="새 값 강조"
        style={{
          position: "absolute",
          left: 980,
          top: 850,
          fontSize: 32,
          color: "#3158D5",
          opacity: interpolate(frame, [5 * fps, 6 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        0.120 → 0.110 / CPU 연산 · 후보 저장
      </Interactive.Div>
      <Caption name="수정 자막 1" durationInFrames={5 * fps}>
        예를 들어 LLM이 좌표 [0, 1]의 값을 0.120에서 0.110으로 지정합니다.
      </Caption>
      <Caption name="수정 자막 2" from={5 * fps} durationInFrames={5 * fps}>
        도구는 원본 해시와 현재 값, 변경 예산을 검사한 뒤 후보를 저장합니다.
      </Caption>
      <Caption name="수정 자막 3" from={10 * fps} durationInFrames={6 * fps}>
        화면의 호출은 일부 인자를 생략한 설명 예시입니다. 실제 가중치를 수정하지
        않습니다.
      </Caption>
    </>
  );
};
export const WriteScene = Interactive.create({
  Component: WriteInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const FeedbackInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="05 / FEEDBACK & REVISION" dark />
      <Interactive.H1
        name="피드백 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 96,
          letterSpacing: -4,
          color: "#F4F2ED",
        }}
      >
        수정한 다음이, 더 중요합니다.
      </Interactive.H1>
      <Interactive.Div
        name="피드백 부제"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#A9B9CB",
        }}
      >
        실제 추론 피드백을 읽고 가설과 후보를 재검토합니다.
      </Interactive.Div>
      <Interactive.Div
        name="원본 노드"
        style={{
          position: "absolute",
          left: 110,
          top: 440,
          padding: "35px 45px",
          width: 340,
          boxSizing: "border-box",
          border: "1px solid #738196",
          borderRadius: 18,
          color: "#F4F2ED",
          fontSize: 48,
          fontWeight: 700,
        }}
      >
        원본
        <span
          style={{
            display: "block",
            marginTop: 18,
            color: "#A9B9CB",
            fontSize: 26,
          }}
        >
          변경하지 않음
        </span>
      </Interactive.Div>
      <Arrow x={490} y={500} width={135} color="#9FB6FF" />
      <Interactive.Div
        name="부모 후보 노드"
        style={{
          position: "absolute",
          left: 665,
          top: 440,
          padding: "35px 45px",
          width: 430,
          boxSizing: "border-box",
          backgroundColor: "#3158D5",
          borderRadius: 18,
          color: "#F4F2ED",
          fontSize: 48,
          fontWeight: 700,
        }}
      >
        후보 A
        <span
          style={{
            display: "block",
            marginTop: 18,
            color: "#DCE6F2",
            fontSize: 26,
          }}
        >
          probe / score
        </span>
      </Interactive.Div>
      <Arrow x={1135} y={500} width={135} color="#9FB6FF" />
      <Interactive.Div
        name="자식 후보 노드"
        style={{
          position: "absolute",
          left: 1310,
          top: 440,
          padding: "35px 45px",
          width: 500,
          boxSizing: "border-box",
          border: "2px solid #8BCABB",
          borderRadius: 18,
          color: "#8BCABB",
          fontSize: 48,
          fontWeight: 700,
          opacity: interpolate(frame, [3 * fps, 4 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        후보 B 또는 복원
        <span
          style={{
            display: "block",
            marginTop: 18,
            color: "#A9B9CB",
            fontSize: 26,
          }}
        >
          revise_candidate()
        </span>
      </Interactive.Div>
      <Interactive.Div
        name="피드백 판단 항목"
        from={2 * fps}
        style={{
          position: "absolute",
          left: 110,
          top: 708,
          fontSize: 37,
          lineHeight: 1.7,
          color: "#F4F2ED",
        }}
      >
        예측한 행동이 바뀌었는가? 정상 사례가 손상되지는 않았는가?
      </Interactive.Div>
      <Interactive.Div
        name="후보 값 비교 도구"
        from={4 * fps}
        style={{
          position: "absolute",
          left: 110,
          top: 802,
          fontSize: 27,
          fontFamily: "Code",
          color: "#9FB6FF",
        }}
      >
        compare_candidate_parameters() → 원본 / 부모 / 자식의 실제 값 비교
      </Interactive.Div>
      <Caption name="피드백 자막 1" durationInFrames={4 * fps}>
        개발 데이터에서 원본과 후보의 추론 결과, 정상 사례 손상을 확인합니다.
      </Caption>
      <Caption name="피드백 자막 2" from={4 * fps} durationInFrames={4 * fps}>
        LLM이 실제 부모 후보의 피드백을 읽고, 근거를 붙여 재수정하거나
        복원합니다.
      </Caption>
      <Caption name="피드백 자막 3" from={8 * fps} durationInFrames={4 * fps}>
        이력과 old/new 값을 보존합니다. 손실 자동 피팅이나 좌표·세기 탐색은 하지
        않습니다.
      </Caption>
    </>
  );
};
export const FeedbackScene = Interactive.create({
  Component: FeedbackInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const ConfirmationInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="06 / FREEZE & CONFIRM" />
      <Interactive.H1
        name="독립 확인 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 96,
          letterSpacing: -4,
          color: "#182638",
        }}
      >
        선택을 고정한 뒤, 독립 확인.
      </Interactive.H1>
      <Interactive.Div
        name="독립 확인 부제"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        개발 중 판단과 최종 확인을 분리합니다.
      </Interactive.Div>
      <Panel
        style={{
          position: "absolute",
          left: 110,
          top: 420,
          width: 650,
          height: 340,
          boxSizing: "border-box",
        }}
      >
        <Interactive.Div
          name="개발 선택 제목"
          style={{ fontSize: 49, fontWeight: 700, color: "#182638" }}
        >
          개발 / 검증
        </Interactive.Div>
        <Interactive.Div
          name="개발 선택 설명"
          style={{
            fontSize: 33,
            lineHeight: 1.7,
            marginTop: 30,
            color: "#68717A",
          }}
        >
          후보를 비교하고 선택
          <br />
          미리 정한 개선·보존 기준
        </Interactive.Div>
        <Interactive.Div
          name="개발 데이터 이름"
          style={{
            fontFamily: "Code",
            fontSize: 25,
            color: "#3158D5",
            marginTop: 20,
          }}
        >
          discovery / validation
        </Interactive.Div>
      </Panel>
      <Interactive.Div
        name="고정 경계"
        style={{
          position: "absolute",
          left: 820,
          top: 402,
          width: 275,
          height: 380,
          backgroundColor: "#3158D5",
          borderRadius: 140,
          textAlign: "center",
          paddingTop: 100,
          boxSizing: "border-box",
          color: "white",
          opacity: interpolate(frame, [2 * fps, 3 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ fontSize: 67, fontWeight: 700 }}>고정</div>
        <div style={{ fontSize: 27, marginTop: 25, lineHeight: 1.7 }}>
          후보 · 데이터
          <br />
          설정 · 해시
        </div>
      </Interactive.Div>
      <Arrow x={1120} y={560} width={90} />
      <Panel
        style={{
          position: "absolute",
          left: 1250,
          top: 420,
          width: 560,
          height: 340,
          boxSizing: "border-box",
          borderColor: "#3D9B92",
        }}
      >
        <Interactive.Div
          name="독립 데이터 제목"
          style={{ fontSize: 49, fontWeight: 700, color: "#182638" }}
        >
          독립 데이터
        </Interactive.Div>
        <Interactive.Div
          name="독립 데이터 설명"
          style={{
            fontSize: 33,
            lineHeight: 1.7,
            marginTop: 30,
            color: "#68717A",
          }}
        >
          개선 · 불확실성 · 보존
          <br />
          실패한 기준도 함께 기록
        </Interactive.Div>
        <Interactive.Div
          name="독립 데이터 이름"
          style={{
            fontFamily: "Code",
            fontSize: 25,
            color: "#3D9B92",
            marginTop: 20,
          }}
        >
          independent test
        </Interactive.Div>
      </Panel>
      <Interactive.Div
        name="테스트 재선택 제한"
        style={{
          position: "absolute",
          left: 110,
          top: 816,
          fontSize: 33,
          color: "#E77751",
        }}
      >
        독립 test 결과를 보고 후보를 다시 선택하거나 수정하지 않습니다.
      </Interactive.Div>
      <Caption name="독립 확인 자막 1" durationInFrames={4 * fps}>
        개발·검증 데이터에서 선택을 끝낸 뒤 후보와 평가 조건을 고정합니다.
      </Caption>
      <Caption
        name="독립 확인 자막 2"
        from={4 * fps}
        durationInFrames={4 * fps}
      >
        그다음 독립 데이터에서 효과와 보존 기준, 불확실성을 확인합니다.
      </Caption>
      <Caption
        name="독립 확인 자막 3"
        from={8 * fps}
        durationInFrames={4 * fps}
      >
        값을 정확히 썼다는 사실과 독립 성능이 개선됐다는 결론은 다릅니다.
      </Caption>
    </>
  );
};
export const ConfirmationScene = Interactive.create({
  Component: ConfirmationInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const RecapInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="09 / WHAT THIS ESTABLISHES" dark />
      <Interactive.H1
        name="마무리 제목"
        style={{
          position: "absolute",
          left: 110,
          top: 165,
          margin: 0,
          fontSize: 110,
          letterSpacing: -5,
          lineHeight: 1.28,
          color: "#F4F2ED",
        }}
      >
        관측 → 가설 → 명시적 수정
        <br />
        <span style={{ color: "#8BCABB" }}>→ 피드백 → 독립 확인</span>
      </Interactive.H1>
      <Interactive.Div
        name="판정 1"
        style={{
          position: "absolute",
          left: 110,
          top: 505,
          fontSize: 44,
          color: "#F4F2ED",
        }}
      >
        01 숫자를 정확히 썼는가?
      </Interactive.Div>
      <Interactive.Div
        name="판정 2"
        style={{
          position: "absolute",
          left: 110,
          top: 593,
          fontSize: 44,
          color: "#F4F2ED",
          opacity: interpolate(frame, [fps, 2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        02 예측한 행동이 바뀌었는가?
      </Interactive.Div>
      <Interactive.Div
        name="판정 3"
        style={{
          position: "absolute",
          left: 110,
          top: 681,
          fontSize: 44,
          color: "#8BCABB",
          opacity: interpolate(frame, [2 * fps, 3 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        03 독립 성능과 보존 기준을 통과했는가?
      </Interactive.Div>
      <Interactive.Div
        name="프로젝트 한계"
        style={{
          position: "absolute",
          left: 110,
          top: 812,
          fontSize: 29,
          color: "#A9B9CB",
        }}
      >
        WSP 기반의 도구·절차·검토 가능한 기록. 일반적인 모델 개선이나 WSP 대비
        우위는 입증되지 않았습니다.
      </Interactive.Div>
      <Caption name="마무리 자막 1" durationInFrames={5 * fps}>
        ForwardEdit는 판단, 실행, 검증을 구분하고 그 근거를 남기는 도구입니다.
      </Caption>
      <Caption name="마무리 자막 2" from={5 * fps} durationInFrames={5 * fps}>
        관측한 성능 상승과 원래 규칙의 판정을 구분해서 함께 보고합니다.
      </Caption>
    </>
  );
};
export const RecapScene = Interactive.create({
  Component: RecapInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});
