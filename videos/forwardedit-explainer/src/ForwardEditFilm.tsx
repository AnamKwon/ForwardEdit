import { AbsoluteFill, Track, useVideoConfig } from "remotion";
import {
  ConfirmationScene,
  FeedbackScene,
  HypothesisScene,
  IntroScene,
  ObserveScene,
  RecapScene,
  RolesScene,
  WriteScene,
} from "./Scenes";
import { Progress } from "./Visuals";
import { KoreanResultsScene, KoreanComparisonScene } from "./ResultsScenes";

export const ForwardEditFilm = () => {
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill
      data-testid="forwardedit-film"
      style={{
        fontFamily: "NanumSquare, sans-serif",
        backgroundColor: "#F4F2ED",
      }}
    >
      <Track name="설명 장면">
        <IntroScene
          name="00 프로젝트 소개"
          durationInFrames={10 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <RolesScene
          name="01 LLM과 도구의 역할"
          from={10 * fps}
          durationInFrames={14 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <ObserveScene
          name="02 실제 값과 추론 관측"
          from={24 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#182638" }}
        />
        <HypothesisScene
          name="03 가설과 수정 근거"
          from={40 * fps}
          durationInFrames={14 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <WriteScene
          name="04 명시적인 CPU 수정"
          from={54 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <FeedbackScene
          name="05 피드백과 후보 재수정"
          from={70 * fps}
          durationInFrames={12 * fps}
          style={{ backgroundColor: "#182638" }}
        />
        <ConfirmationScene
          name="06 후보 고정과 독립 확인"
          from={82 * fps}
          durationInFrames={12 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <KoreanResultsScene
          name="07 기록된 Qwen 결과"
          from={94 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <KoreanComparisonScene
          name="08 충분한 데이터의 파인튜닝 비교 필요"
          from={110 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <RecapScene
          name="07 판정과 한계"
          from={126 * fps}
          durationInFrames={10 * fps}
          style={{ backgroundColor: "#182638" }}
        />
      </Track>
      <Progress locale="ko" />
    </AbsoluteFill>
  );
};
