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
} from "./EnglishScenes";
import { Progress } from "./Visuals";
import { EnglishResultsScene, EnglishComparisonScene } from "./ResultsScenes";

export const EnglishFilm = () => {
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill
      data-testid="forwardedit-film"
      style={{
        fontFamily: "NanumSquare, sans-serif",
        backgroundColor: "#F4F2ED",
      }}
    >
      <Track name="Explainer scenes">
        <IntroScene
          name="00 Introduction"
          durationInFrames={10 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <RolesScene
          name="01 Roles"
          from={10 * fps}
          durationInFrames={14 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <ObserveScene
          name="02 Read and observe"
          from={24 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#182638" }}
        />
        <HypothesisScene
          name="03 Form a hypothesis"
          from={40 * fps}
          durationInFrames={14 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <WriteScene
          name="04 Explicit CPU edit"
          from={54 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <FeedbackScene
          name="05 Feedback and revision"
          from={70 * fps}
          durationInFrames={12 * fps}
          style={{ backgroundColor: "#182638" }}
        />
        <ConfirmationScene
          name="06 Freeze and confirm"
          from={82 * fps}
          durationInFrames={12 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <EnglishResultsScene
          name="07 Retained Qwen results"
          from={94 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <EnglishComparisonScene
          name="08 Fine-tuning comparison with sufficient data"
          from={110 * fps}
          durationInFrames={16 * fps}
          style={{ backgroundColor: "#F4F2ED" }}
        />
        <RecapScene
          name="07 Claims and limitations"
          from={126 * fps}
          durationInFrames={10 * fps}
          style={{ backgroundColor: "#182638" }}
        />
      </Track>
      <Progress />
    </AbsoluteFill>
  );
};
