import { Interactive, useVideoConfig } from "remotion";
import evidence from "./recorded-evidence.json";
import { Caption, Header } from "./Visuals";

const qwen = evidence.qwen006.independent;
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

// Measured values form a data-driven group, not individually editable copy.
const QwenMetrics = ({ ko = false }: { ko?: boolean }) => (
  <>
    <div
      style={{
        position: "absolute",
        left: 110,
        top: 410,
        width: 970,
        height: 350,
        padding: 38,
        boxSizing: "border-box",
        background: "#FFFFFF",
        border: "1px solid #D9DEDF",
        borderRadius: 20,
      }}
    >
      <div style={{ color: "#68717A", fontSize: 29 }}>
        {ko
          ? "독립 평가 선정 부분집합 · 192문항"
          : "Selected independent subsets · 192 problems"}
      </div>
      <div
        style={{
          fontSize: 84,
          fontWeight: 700,
          color: "#3158D5",
          marginTop: 32,
        }}
      >
        {qwen.original_correct}/192 → {qwen.edited_correct}/192
      </div>
      <div style={{ fontSize: 36, color: "#182638", marginTop: 24 }}>
        +{(qwen.statistics.gain * 100).toFixed(2)} pp · 97.5% interval
      </div>
      <div
        style={{
          fontFamily: "Code",
          fontSize: 28,
          color: "#68717A",
          marginTop: 16,
        }}
      >
        [+{(qwen.statistics.paired_bootstrap_interval[0] * 100).toFixed(2)}, +
        {(qwen.statistics.paired_bootstrap_interval[1] * 100).toFixed(2)}] pp
      </div>
    </div>
    <div
      style={{
        position: "absolute",
        left: 1130,
        top: 410,
        width: 680,
        height: 350,
        padding: 38,
        boxSizing: "border-box",
        background: "#F3E4DA",
        borderRadius: 20,
      }}
    >
      <div style={{ color: "#A54732", fontSize: 43, fontWeight: 700 }}>
        {ko ? "사전 보존 규칙" : "Prespecified rule"}
      </div>
      <div style={{ color: "#182638", fontSize: 34, marginTop: 36 }}>
        {ko ? "기존 정답 손실" : "Old-correct answers lost"}: 1/18
      </div>
      <div
        style={{
          color: "#A54732",
          fontSize: 53,
          fontWeight: 700,
          marginTop: 24,
        }}
      >
        {percent(qwen.statistics.fraction_of_originally_correct_lost)} &gt;{" "}
        {percent(evidence.qwen006.preservation_limit)}
      </div>
      <div style={{ color: "#68717A", fontSize: 27, marginTop: 20 }}>
        {ko
          ? "17/18개 유지 · 해당 5% 규칙 미충족"
          : "17/18 retained; study's 5% rule missed"}
      </div>
    </div>
  </>
);

const ComparisonRequirements = ({ ko = false }: { ko?: boolean }) => (
  <div
    style={{
      position: "absolute",
      left: 110,
      top: 405,
      width: 1700,
      color: "#182638",
    }}
  >
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1320px 328px",
        padding: "0 26px 23px",
        boxSizing: "border-box",
        fontSize: 29,
        color: "#68717A",
      }}
    >
      <span>{ko ? "비교에 필요한 근거" : "Evidence required for comparison"}</span>
      <span>{ko ? "현재 상태" : "Current status"}</span>
    </div>
    {(ko
      ? [
          "충분한 과제 데이터로 학습하고 검증으로 선택한 파인튜닝",
          "같은 원본과 공통 test · 목표 및 비대상 과제 보존",
          "데이터 · 조정 · 추론 · 학습을 포함한 전체 비용",
        ]
      : [
          "Fine-tuning on substantial task data, selected on validation",
          "Same original and common test; target and unrelated retention",
          "Total resources: data, tuning, inference and training",
        ]
    ).map((requirement) => (
      <div
        key={requirement}
        style={{
          display: "grid",
          gridTemplateColumns: "1320px 328px",
          alignItems: "center",
          padding: "25px 26px",
          boxSizing: "border-box",
          height: 108,
          marginBottom: 12,
          borderRadius: 14,
          background: "#FFFFFF",
          fontSize: 32,
        }}
      >
        <span>{requirement}</span>
        <span style={{ color: "#A54732" }}>
          {ko ? "미측정" : "Not measured"}
        </span>
      </div>
    ))}
  </div>
);

const EnglishResultsInner = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="07 / RETAINED RESULTS" />
      <Interactive.H1
        name="Recorded results title"
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
        Accuracy gain and retention trade-off.
      </Interactive.H1>
      <Interactive.Div
        name="Recorded results scope"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        Qwen 006 · source-informed host branch · fixed-original comparison
      </Interactive.Div>
      <QwenMetrics />
      <Interactive.Div
        name="Unparsed answer limitation"
        style={{
          position: "absolute",
          left: 110,
          top: 815,
          fontSize: 30,
          color: "#68717A",
        }}
      >
        All 86 rescues started from unparsed answers. New mathematical knowledge
        is not established.
      </Interactive.Div>
      <Caption name="Recorded results caption 1" durationInFrames={4 * fps}>
        This subset gained 85 net correct answers: 86 rescued and one lost.
      </Caption>
      <Caption
        name="Recorded results caption 2"
        from={4 * fps}
        durationInFrames={4 * fps}
      >
        Seventeen of 18 originally correct answers remained correct.
      </Caption>
      <Caption
        name="Recorded results caption 3"
        from={8 * fps}
        durationInFrames={4 * fps}
      >
        The prespecified 5% rule allows zero losses when the denominator is 18.
      </Caption>
      <Caption
        name="Recorded results caption 4"
        from={12 * fps}
        durationInFrames={4 * fps}
      >
        Missing that study rule does not erase the measured accuracy gain.
      </Caption>
    </>
  );
};
export const EnglishResultsScene = Interactive.create({
  Component: EnglishResultsInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const KoreanResultsInner = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="07 / RETAINED RESULTS" />
      <Interactive.H1
        name="기록된 결과 제목"
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
        점수 상승과 보존의 절충.
      </Interactive.H1>
      <Interactive.Div
        name="기록된 결과 범위"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#68717A",
        }}
      >
        Qwen 006 · 원문 참고 호스트 분기 · 고정 원본 대비
      </Interactive.Div>
      <QwenMetrics ko />
      <Interactive.Div
        name="미파싱 답변 구제 한계"
        style={{
          position: "absolute",
          left: 110,
          top: 815,
          fontSize: 30,
          color: "#68717A",
        }}
      >
        구제된 86개는 모두 이전 미파싱 답변입니다. 새 수학 지식 습득을 입증하지
        않습니다.
      </Interactive.Div>
      <Caption name="기록된 결과 자막 1" durationInFrames={4 * fps}>
        새 정답 86개와 손실 1개로, 이 선정 집합의 정답은 순증 85개였습니다.
      </Caption>
      <Caption
        name="기록된 결과 자막 2"
        from={4 * fps}
        durationInFrames={4 * fps}
      >
        기존에 맞힌 18개 중 17개, 즉 94.44%는 정답을 유지했습니다.
      </Caption>
      <Caption
        name="기록된 결과 자막 3"
        from={8 * fps}
        durationInFrames={4 * fps}
      >
        18개에서 사전 5% 규칙은 사실상 손실 0개만 허용하는 엄격한 규칙입니다.
      </Caption>
      <Caption
        name="기록된 결과 자막 4"
        from={12 * fps}
        durationInFrames={4 * fps}
      >
        해당 실험의 규칙 미충족이 측정된 정답률 상승을 부정하지는 않습니다.
      </Caption>
    </>
  );
};
export const KoreanResultsScene = Interactive.create({
  Component: KoreanResultsInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const EnglishComparisonInner = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="08 / FAIR COMPARISON" />
      <Interactive.H1
        name="Fine-tuning comparison title"
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
        Compare with properly trained baselines.
      </Interactive.H1>
      <Interactive.Div
        name="Reference comparison scope"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#A54732",
        }}
      >
        The 12-story trainability check is excluded from method comparisons.
      </Interactive.Div>
      <ComparisonRequirements />
      <Interactive.Div
        name="Reference comparison limitation"
        style={{
          position: "absolute",
          left: 110,
          top: 830,
          fontSize: 28,
          color: "#68717A",
        }}
      >
        Report common-data and larger-data comparisons separately. No winner
        has been measured.
      </Interactive.Div>
      <Caption name="Comparison caption 1" durationInFrames={5 * fps}>
        Twelve training stories cannot represent properly tuned fine-tuning.
      </Caption>
      <Caption
        name="Comparison caption 2"
        from={5 * fps}
        durationInFrames={5 * fps}
      >
        Use substantial task data, validation selection and common quality and
        retention checks.
      </Caption>
      <Caption
        name="Comparison caption 3"
        from={10 * fps}
        durationInFrames={6 * fps}
      >
        Report the data and total costs. This comparison has not yet been run.
      </Caption>
    </>
  );
};
export const EnglishComparisonScene = Interactive.create({
  Component: EnglishComparisonInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});

const KoreanComparisonInner = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Header step="08 / FAIR COMPARISON" />
      <Interactive.H1
        name="파인튜닝 비교 제목"
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
        충분한 데이터의 파인튜닝과 비교.
      </Interactive.H1>
      <Interactive.Div
        name="참고 비교 범위"
        style={{
          position: "absolute",
          left: 115,
          top: 282,
          fontSize: 35,
          color: "#A54732",
        }}
      >
        12문장 학습 동작 검사는 방법 비교에서 제외합니다.
      </Interactive.Div>
      <ComparisonRequirements ko />
      <Interactive.Div
        name="참고 비교 한계"
        style={{
          position: "absolute",
          left: 110,
          top: 830,
          fontSize: 28,
          color: "#68717A",
        }}
      >
        공통 데이터 비교와 더 큰 데이터의 실용 비교를 따로 보고합니다. 우열은
        아직 측정하지 않았습니다.
      </Interactive.Div>
      <Caption name="파인튜닝 비교 자막 1" durationInFrames={5 * fps}>
        학습 12문장으로 제대로 조정한 파인튜닝을 대표할 수는 없습니다.
      </Caption>
      <Caption
        name="파인튜닝 비교 자막 2"
        from={5 * fps}
        durationInFrames={5 * fps}
      >
        충분한 과제 데이터와 검증 선택, 공통 성능·보존 평가가 필요합니다.
      </Caption>
      <Caption
        name="파인튜닝 비교 자막 3"
        from={10 * fps}
        durationInFrames={6 * fps}
      >
        데이터와 전체 비용도 보고해야 합니다. 이 비교는 아직 실행하지
        않았습니다.
      </Caption>
    </>
  );
};
export const KoreanComparisonScene = Interactive.create({
  Component: KoreanComparisonInner,
  layout: "absolute-fill",
  schema: { ...Interactive.backgroundSchema },
});
