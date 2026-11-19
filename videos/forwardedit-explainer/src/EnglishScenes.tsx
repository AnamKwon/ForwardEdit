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
        name="Weight sculpture"
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
        name="Opening eyebrow"
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
        name="Project name"
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
        name="Opening title"
        style={{
          position: "absolute",
          left: 110,
          top: 414,
          fontSize: 74,
          lineHeight: 1.35,
          fontWeight: 700,
          letterSpacing: -2,
          color: "#182638",
          opacity: interpolate(frame, [fps, 2 * fps], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Read the weights.
        <br />
        Edit with intent.
        <br />
        Verify the result.
      </Interactive.Div>
      <Interactive.Div
        name="Opening description"
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
      <Caption name="Opening caption 1" from={fps} durationInFrames={4 * fps}>
        The editor LLM reads real model weights and forward observations.
      </Caption>
      <Caption
        name="Opening caption 2"
        from={5 * fps}
        durationInFrames={5 * fps}
      >
        ForwardEdit turns its explicit choices into local edits and inspectable
        evidence.
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
        name="Roles title"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 88,
          letterSpacing: -2,
          color: "#182638",
        }}
      >
        The LLM decides. Tools execute.
      </Interactive.H1>
      <Interactive.Div
        name="Roles description"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        The editor LLM and the model being edited have different roles.
      </Interactive.Div>
      <Interactive.Div
        name="External LLM role"
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
          External LLM
        </div>
        <div style={{ fontSize: 32, marginTop: 23, lineHeight: 1.6 }}>
          Observe · hypothesize
          <br />
          Set coordinates &amp; values
        </div>
      </Interactive.Div>
      <Arrow x={630} y={540} width={90} />
      <Interactive.Div
        name="MCP tools role"
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
          MCP tools
        </div>
        <div style={{ fontSize: 32, marginTop: 23, lineHeight: 1.6 }}>
          Explicit CPU arithmetic
          <br />
          Candidates &amp; audit trail
        </div>
      </Interactive.Div>
      <Arrow x={1250} y={540} width={90} />
      <Interactive.Div
        name="Target model role"
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
        <div style={{ fontSize: 52, marginTop: 24, fontWeight: 700 }}>
          Target model
        </div>
        <div style={{ fontSize: 32, marginTop: 23, lineHeight: 1.6 }}>
          Local weights + patch
          <br />
          Forward inference
        </div>
      </Interactive.Div>
      <Interactive.Div
        name="Return observations"
        from={4 * fps}
        style={{
          position: "absolute",
          left: 120,
          top: 797,
          fontSize: 29,
          color: "#3158D5",
        }}
      >
        ← The LLM reads returned observations before making its next decision.
      </Interactive.Div>
      <Caption name="Roles caption 1" durationInFrames={5 * fps}>
        An external LLM chooses what to change and why, then calls tools through
        MCP.
      </Caption>
      <Caption name="Roles caption 2" from={5 * fps} durationInFrames={5 * fps}>
        Tools execute the specified arithmetic on CPU. GPU use is inference
        only.
      </Caption>
      <Caption
        name="Roles caption 3"
        from={10 * fps}
        durationInFrames={4 * fps}
      >
        No optimizer inside the tools selects or fits the new weights.
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
        name="Observation title"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 88,
          letterSpacing: -2,
          color: "#F4F2ED",
        }}
      >
        Inspect the model as it is.
      </Interactive.H1>
      <Interactive.Div
        name="Observation description"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#A9B9CB",
        }}
      >
        Read actual values, axes and forward code before choosing an edit.
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
          name="Connection tools"
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
          name="Structure inspection tools"
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
          name="Weight and forward observations"
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
        name="Inspection illustration"
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
      <Caption name="Observation caption 1" durationInFrames={5 * fps}>
        Start with the execution contract, API schema, model and
        development-data settings.
      </Caption>
      <Caption
        name="Observation caption 2"
        from={5 * fps}
        durationInFrames={5 * fps}
      >
        Inspect actual modules, weight values and token-position observations.
      </Caption>
      <Caption
        name="Observation caption 3"
        from={10 * fps}
        durationInFrames={6 * fps}
      >
        Compare signed contributions in failure and reference contexts before
        proposing a change.
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
        name="Hypothesis title"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 88,
          letterSpacing: -2,
          color: "#182638",
        }}
      >
        From numbers to a testable hypothesis.
      </Interactive.H1>
      <Interactive.Div
        name="Hypothesis description"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        A large activation alone does not identify a semantic role or a cause.
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
          name="Local arithmetic label"
          style={{ fontSize: 24, color: "#3158D5", letterSpacing: 2 }}
        >
          ILLUSTRATIVE LOCAL CONTRIBUTION
        </Interactive.Div>
        <Interactive.Div
          name="Contribution arithmetic"
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
          name="Change arithmetic"
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
          name="Local arithmetic limitation"
          style={{
            fontSize: 28,
            color: "#68717A",
            marginTop: 28,
            lineHeight: 1.5,
          }}
        >
          Fixed-input local arithmetic only.
          <br />
          This alone does not predict model behavior.
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
          name="Hypothesis requirements"
          style={{
            fontSize: 36,
            color: "#182638",
            fontWeight: 700,
            lineHeight: 1.85,
          }}
        >
          Why this coordinate and value?
          <br />
          What else could explain it?
          <br />
          What would falsify the hypothesis?
        </Interactive.Div>
        <Interactive.Div
          name="Proposal preparation tool"
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
      <Caption name="Hypothesis caption 1" durationInFrames={5 * fps}>
        State the numerical derivation, alternative explanations and a
        falsification condition.
      </Caption>
      <Caption
        name="Hypothesis caption 2"
        from={5 * fps}
        durationInFrames={5 * fps}
      >
        Missing evidence means defer. A prepared hypothesis permits a bounded
        causal probe.
      </Caption>
      <Caption
        name="Hypothesis caption 3"
        from={10 * fps}
        durationInFrames={4 * fps}
      >
        ready_for_causal_probe certifies neither causality nor performance.
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
        name="Explicit edit title"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 88,
          letterSpacing: -2,
          color: "#182638",
        }}
      >
        Write exactly what the LLM specifies.
      </Interactive.H1>
      <Interactive.Div
        name="Explicit edit description"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        Keep the source unchanged. Save an explicit candidate separately.
      </Interactive.Div>
      <Matrix
        name="Illustrative weight matrix"
        edited
        style={{ position: "absolute", left: 110, top: 400, width: 650 }}
      />
      <Interactive.Div
        name="Edit coordinate"
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
          name="Abbreviated explicit-value call"
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
          name="Pre-write checks"
          from={6 * fps}
          style={{ color: "#8BCABB", fontSize: 26, marginTop: 14 }}
        >
          source hash · old value · change budget
        </Interactive.Div>
      </div>
      <Interactive.Div
        name="Explicit value change"
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
        0.120 → 0.110 / CPU edit · saved candidate
      </Interactive.Div>
      <Caption name="Edit caption 1" durationInFrames={5 * fps}>
        Here, the LLM explicitly chooses 0.120 → 0.110 at coordinate [0, 1].
      </Caption>
      <Caption name="Edit caption 2" from={5 * fps} durationInFrames={5 * fps}>
        The tool checks the source hash, current value and change budget before
        saving a candidate.
      </Caption>
      <Caption name="Edit caption 3" from={10 * fps} durationInFrames={6 * fps}>
        This abbreviated call and these numbers are illustrative. No real model
        is edited.
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
        name="Feedback title"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 88,
          letterSpacing: -2,
          color: "#F4F2ED",
        }}
      >
        An edit is the start of verification.
      </Interactive.H1>
      <Interactive.Div
        name="Feedback description"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#A9B9CB",
        }}
      >
        Use actual forward feedback to reconsider the hypothesis and candidate.
      </Interactive.Div>
      <Interactive.Div
        name="Original checkpoint"
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
        Original
        <span
          style={{
            display: "block",
            marginTop: 18,
            color: "#A9B9CB",
            fontSize: 26,
          }}
        >
          Unchanged source
        </span>
      </Interactive.Div>
      <Arrow x={490} y={500} width={135} color="#9FB6FF" />
      <Interactive.Div
        name="Parent candidate"
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
        Candidate A
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
        name="Child candidate or restoration"
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
        Revise or restore
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
        name="Feedback questions"
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
        Did the predicted behavior change? Were normal cases preserved?
      </Interactive.Div>
      <Interactive.Div
        name="Candidate parameter comparison"
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
        compare_candidate_parameters() → actual source / parent / child values
      </Interactive.Div>
      <Caption name="Feedback caption 1" durationInFrames={4 * fps}>
        Use development data to check the intended effect and damage to normal
        cases.
      </Caption>
      <Caption
        name="Feedback caption 2"
        from={4 * fps}
        durationInFrames={4 * fps}
      >
        Read actual parent-candidate feedback before revising or restoring
        values.
      </Caption>
      <Caption
        name="Feedback caption 3"
        from={8 * fps}
        durationInFrames={4 * fps}
      >
        Preserve lineage and old/new values. No loss fitting or
        coordinate/strength sweeps.
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
        name="Independent confirmation title"
        style={{
          position: "absolute",
          left: 110,
          top: 150,
          margin: 0,
          fontSize: 88,
          letterSpacing: -2,
          color: "#182638",
        }}
      >
        Freeze the choice. Then confirm.
      </Interactive.H1>
      <Interactive.Div
        name="Independent confirmation description"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        Separate development selection from independent confirmation.
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
          name="Development selection title"
          style={{ fontSize: 49, fontWeight: 700, color: "#182638" }}
        >
          Development
        </Interactive.Div>
        <Interactive.Div
          name="Development selection description"
          style={{
            fontSize: 33,
            lineHeight: 1.7,
            marginTop: 30,
            color: "#68717A",
          }}
        >
          Compare and select candidates
          <br />
          Declared gain &amp; preservation rules
        </Interactive.Div>
        <Interactive.Div
          name="Development split names"
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
        name="Freeze boundary"
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
        <div style={{ fontSize: 67, fontWeight: 700 }}>Freeze</div>
        <div style={{ fontSize: 27, marginTop: 25, lineHeight: 1.7 }}>
          Candidate · data
          <br />
          Settings · hashes
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
          name="Independent data title"
          style={{ fontSize: 49, fontWeight: 700, color: "#182638" }}
        >
          Independent test
        </Interactive.Div>
        <Interactive.Div
          name="Independent data description"
          style={{
            fontSize: 33,
            lineHeight: 1.7,
            marginTop: 30,
            color: "#68717A",
          }}
        >
          Effect · uncertainty
          <br />
          Preservation · failed criteria
        </Interactive.Div>
        <Interactive.Div
          name="Independent split name"
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
        name="No test-driven reselection"
        style={{
          position: "absolute",
          left: 110,
          top: 816,
          fontSize: 33,
          color: "#E77751",
        }}
      >
        Do not retune or reselect candidates after inspecting the independent
        test.
      </Interactive.Div>
      <Caption name="Confirmation caption 1" durationInFrames={4 * fps}>
        Finish development selection, then freeze the candidate and evaluation
        settings.
      </Caption>
      <Caption
        name="Confirmation caption 2"
        from={4 * fps}
        durationInFrames={4 * fps}
      >
        Check effect, preservation and uncertainty on independent data.
      </Caption>
      <Caption
        name="Confirmation caption 3"
        from={8 * fps}
        durationInFrames={4 * fps}
      >
        A correct weight write does not establish an independent task
        improvement.
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
      <Header step="07 / WHAT THIS ESTABLISHES" dark />
      <Interactive.H1
        name="Closing title"
        style={{
          position: "absolute",
          left: 110,
          top: 165,
          margin: 0,
          fontSize: 92,
          letterSpacing: -2,
          lineHeight: 1.28,
          color: "#F4F2ED",
        }}
      >
        Observe → Hypothesize → Edit
        <br />
        <span style={{ color: "#8BCABB" }}>→ Read feedback → Confirm</span>
      </Interactive.H1>
      <Interactive.Div
        name="Write correctness question"
        style={{
          position: "absolute",
          left: 110,
          top: 505,
          fontSize: 44,
          color: "#F4F2ED",
        }}
      >
        01 Were the intended values written correctly?
      </Interactive.Div>
      <Interactive.Div
        name="Behavioral effect question"
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
        02 Did the predicted behavior change?
      </Interactive.Div>
      <Interactive.Div
        name="Independent improvement question"
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
        03 Did independent improvement and preservation pass?
      </Interactive.Div>
      <Interactive.Div
        name="Scope and limitations"
        style={{
          position: "absolute",
          left: 110,
          top: 812,
          fontSize: 29,
          color: "#A9B9CB",
        }}
      >
        WSP-inspired tools, workflow and inspectable evidence. General model
        improvement or superiority over WSP is not established.
      </Interactive.Div>
      <Caption name="Closing caption 1" durationInFrames={5 * fps}>
        ForwardEdit separates judgment, execution and verification, and retains
        the evidence.
      </Caption>
      <Caption
        name="Closing caption 2"
        from={5 * fps}
        durationInFrames={5 * fps}
      >
        Every published study failed at least one declared acceptance criterion.
      </Caption>
    </>
  );
};
export const RecapScene = Interactive.create({
  Component: RecapInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});
