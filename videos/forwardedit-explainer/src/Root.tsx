import { Composition, Folder } from "remotion";
import { EnglishFilm } from "./EnglishFilm";
import { ForwardEditFilm } from "./ForwardEditFilm";
import * as EN from "./EnglishScenes";
import * as KO from "./Scenes";

export const RemotionRoot = () => (
  <>
    <Composition
      id="ForwardEdit-English"
      component={EnglishFilm}
      width={1920}
      height={1080}
      fps={30}
      durationInFrames={3120}
    />
    <Composition
      id="ForwardEdit-Korean"
      component={ForwardEditFilm}
      width={1920}
      height={1080}
      fps={30}
      durationInFrames={3120}
    />
    <Folder name="EN-Scenes">
      <Composition
        id="EN-Intro"
        component={EN.IntroScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={300}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Roles"
        component={EN.RolesScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={420}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Observe"
        component={EN.ObserveScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={480}
        defaultProps={{
          style: { backgroundColor: "#182638", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Hypothesis"
        component={EN.HypothesisScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={420}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Explicit-Write"
        component={EN.WriteScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={480}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Feedback"
        component={EN.FeedbackScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={360}
        defaultProps={{
          style: { backgroundColor: "#182638", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Confirmation"
        component={EN.ConfirmationScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={360}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="EN-Recap"
        component={EN.RecapScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={300}
        defaultProps={{
          style: { backgroundColor: "#182638", fontFamily: "NanumSquare" },
        }}
      />
    </Folder>
    <Folder name="KO-Scenes">
      <Composition
        id="KO-Intro"
        component={KO.IntroScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={300}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Roles"
        component={KO.RolesScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={420}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Observe"
        component={KO.ObserveScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={480}
        defaultProps={{
          style: { backgroundColor: "#182638", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Hypothesis"
        component={KO.HypothesisScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={420}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Explicit-Write"
        component={KO.WriteScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={480}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Feedback"
        component={KO.FeedbackScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={360}
        defaultProps={{
          style: { backgroundColor: "#182638", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Confirmation"
        component={KO.ConfirmationScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={360}
        defaultProps={{
          style: { backgroundColor: "#F4F2ED", fontFamily: "NanumSquare" },
        }}
      />
      <Composition
        id="KO-Recap"
        component={KO.RecapScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={300}
        defaultProps={{
          style: { backgroundColor: "#182638", fontFamily: "NanumSquare" },
        }}
      />
    </Folder>
  </>
);
