import type { CSSProperties, ReactNode } from "react";
import { loadFont } from "@remotion/fonts";
import {
  Interactive,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

await Promise.all([
  loadFont({
    family: "NanumSquare",
    url: staticFile("fonts/NanumSquare-Regular.ttf"),
    weight: "400",
  }),
  loadFont({
    family: "NanumSquare",
    url: staticFile("fonts/NanumSquare-Bold.ttf"),
    weight: "700",
  }),
  loadFont({
    family: "Code",
    url: staticFile("fonts/DejaVuSansMono.ttf"),
    weight: "400",
  }),
]);

export const palette = {
  paper: "#F4F2ED",
  ink: "#182638",
  blue: "#3158D5",
  teal: "#3D9B92",
  coral: "#E77751",
};

export const Header = ({
  step,
  dark = false,
}: {
  step: string;
  dark?: boolean;
}) => (
  <div
    style={{
      position: "absolute",
      top: 60,
      left: 110,
      right: 110,
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      color: dark ? "#A9B9CB" : "#68717A",
      fontSize: 23,
      letterSpacing: 3,
      fontWeight: 700,
    }}
  >
    <span>
      <span
        style={{
          display: "inline-block",
          width: 12,
          height: 12,
          background: "#E77751",
          marginRight: 14,
        }}
      />
      FORWARDEDIT <span style={{ opacity: 0.45, margin: "0 18px" }}>/</span> HOW
      IT WORKS
    </span>
    <span>{step}</span>
  </div>
);

const CaptionInner = ({
  children,
  style,
}: {
  children: ReactNode;
  style?: CSSProperties;
}) => (
  <div
    style={{
      position: "absolute",
      left: 110,
      right: 110,
      bottom: 88,
      color: "#182638",
      fontSize: 35,
      lineHeight: 1.5,
      fontWeight: 700,
      textAlign: "center",
      padding: "14px 28px",
      backgroundColor: "#F4F2ED",
      borderRadius: 6,
      ...style,
    }}
  >
    {children}
  </div>
);
export const Caption = Interactive.create({
  Component: CaptionInner,
  layout: "none",
  schema: {
    ...Interactive.childrenSchema,
    ...Interactive.textSchema,
    ...Interactive.backgroundSchema,
  },
});

export const Panel = ({
  children,
  style,
}: {
  children: ReactNode;
  style?: CSSProperties;
}) => (
  <div
    style={{
      border: "1px solid #DCDDD9",
      background: "#FFFDF8",
      borderRadius: 20,
      padding: 36,
      ...style,
    }}
  >
    {children}
  </div>
);

export const ToolLine = ({
  children,
  color = "#3158D5",
}: {
  children: ReactNode;
  color?: string;
}) => (
  <div
    style={{
      fontFamily: "Code, monospace",
      fontSize: 24,
      color,
      padding: "16px 0",
      borderBottom: "1px solid #B5BEC526",
      whiteSpace: "pre-wrap",
      lineHeight: 1.6,
    }}
  >
    {children}
  </div>
);

export const Arrow = ({
  x,
  y,
  width = 140,
  color = "#3158D5",
}: {
  x: number;
  y: number;
  width?: number;
  color?: string;
}) => (
  <svg
    width={width}
    height={40}
    viewBox={`0 0 ${width} 40`}
    style={{ position: "absolute", left: x, top: y, overflow: "visible" }}
  >
    <path
      d={`M 0 20 H ${width - 6} M ${width - 24} 3 L ${width - 5} 20 L ${width - 24} 37`}
      stroke={color}
      strokeWidth={4}
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const MatrixInner = ({
  edited,
  style,
}: {
  edited: boolean;
  style: CSSProperties;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const change = edited
    ? interpolate(frame, [4 * fps, 5 * fps], [0, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      })
    : 0;
  const values = [
    0.04, 0.12, -0.03, 0.02, 0.01, -0.06, 0.08, 0.03, -0.02, 0.05, 0.11, -0.01,
    0.07, 0.02, -0.04, 0.09,
  ];
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 12,
        width: 600,
        ...style,
      }}
    >
      {values.map((value, i) => (
        <div
          key={i}
          style={{
            height: 100,
            borderRadius: 12,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            fontFamily: "Code",
            fontSize: 31,
            background:
              i === 1 ? (change > 0 ? "#3158D5" : "#F4D6C7") : "#E9EAE5",
            color: i === 1 && change > 0 ? "white" : "#354253",
            border: i === 1 ? "3px solid #E77751" : "3px solid transparent",
          }}
        >
          {(i === 1 ? value - 0.01 * change : value).toFixed(3)}
        </div>
      ))}
    </div>
  );
};
export const Matrix = Interactive.create({
  Component: MatrixInner,
  layout: "none",
  schema: {
    edited: {
      type: "boolean",
      default: false,
      description: "Show illustrative value change",
    },
  },
});

export const Progress = ({ locale = "en" }: { locale?: "en" | "ko" }) => {
  const frame = useCurrentFrame();
  const { durationInFrames, fps } = useVideoConfig();
  return (
    <div
      style={{
        position: "absolute",
        bottom: 30,
        left: 110,
        right: 110,
        fontFamily: "NanumSquare",
        color: "#8A9199",
        fontSize: 18,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 13,
        }}
      >
        <span>
          {locale === "en"
            ? "WORKFLOW DEMO + RETAINED RESULTS · NO NEW TARGET-MODEL RUN"
            : "설명용 시연과 과거 결과 · 새 대상 모델 실험 없음"}
        </span>
        <span>FORWARDEDIT / {durationInFrames / fps} SEC</span>
      </div>
      <div style={{ height: 3, background: "#8793A329" }}>
        <div
          style={{
            height: 3,
            background: "#E77751",
            width: `${Math.min(100, (frame / (durationInFrames - 1)) * 100)}%`,
          }}
        />
      </div>
    </div>
  );
};
