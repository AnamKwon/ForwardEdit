import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import ts from "typescript";

const root = fileURLToPath(new URL("../", import.meta.url));
const fps = 30;
const parse = async (name) =>
  ts.createSourceFile(
    name,
    await readFile(path.join(root, name), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
const collect = (node, test) => {
  const found = [];
  const visit = (child) => {
    if (test(child)) found.push(child);
    ts.forEachChild(child, visit);
  };
  visit(node);
  return found;
};
const number = (node) => {
  if (ts.isNumericLiteral(node)) return Number(node.text);
  if (ts.isIdentifier(node) && node.text === "fps") return fps;
  if (
    ts.isBinaryExpression(node) &&
    node.operatorToken.kind === ts.SyntaxKind.AsteriskToken
  )
    return number(node.left) * number(node.right);
  throw new Error(`Expected static frame timing: ${node.getText()}`);
};
const attr = (node, name, fallback) => {
  const a = node.attributes.properties.find(
    (p) => ts.isJsxAttribute(p) && p.name.getText() === name,
  );
  if (!a) return fallback;
  if (ts.isStringLiteral(a.initializer)) return a.initializer.text;
  if (ts.isJsxExpression(a.initializer) && a.initializer.expression)
    return number(a.initializer.expression);
  throw new Error(`Unsupported ${name}: ${a.getText()}`);
};
const clock = (frame) => {
  const ms = Math.round((frame * 1000) / fps);
  return `${String(Math.floor(ms / 3600000)).padStart(2, "0")}:${String(Math.floor(ms / 60000) % 60).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")},${String(ms % 1000).padStart(3, "0")}`;
};
const sources = [];
for (const name of [
  "README.md",
  "README.ko.md",
  "DESIGN.md",
  "docs/USAGE.md",
  "docs/INFERENCE_ONLY_CONTRACT.md",
  "docs/REVIEW_GUIDE.md",
  "docs/RESULTS.md",
  "docs/FINETUNING_COMPARISON.md",
  "docs/FINETUNING_COMPARISON.ko.md",
  "docs/assets/results/figure-data.json",
  "experiments/extended-discovery-006/study-outcome.json",
  "experiments/direct-values-inference-only-001/host-audit.json",
  "docs/evidence/mcp-training-verification.json",
  "weight_space_lab/server.py",
  "weight_space_lab/standalone.py",
]) {
  const bytes = await readFile(path.resolve(root, "../..", name));
  sources.push({
    path: name,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  });
}
await mkdir(path.join(root, "production"), { recursive: true });
for (const [locale, film, sceneFile] of [
  ["en", "EnglishFilm", "EnglishScenes"],
  ["ko", "ForwardEditFilm", "Scenes"],
]) {
  const parent = await parse(`src/${film}.tsx`);
  const source = await parse(`src/${sceneFile}.tsx`);
  const results = await parse("src/ResultsScenes.tsx");
  const declarations = new Map(
    [
      ...collect(source, ts.isVariableDeclaration),
      ...collect(results, ts.isVariableDeclaration),
    ].map((d) => [d.name.getText(), d.initializer]),
  );
  const scenes = collect(
    parent,
    (n) =>
      ts.isJsxSelfClosingElement(n) && n.tagName.getText().endsWith("Scene"),
  ).map((node) => {
    const id = node.tagName.getText();
    const definition = declarations.get(id);
    const component = definition.arguments[0].properties
      .find(
        (p) => ts.isPropertyAssignment(p) && p.name.getText() === "Component",
      )
      .initializer.getText();
    const from = attr(node, "from", 0);
    const duration = attr(node, "durationInFrames");
    const captions = collect(
      declarations.get(component),
      (n) =>
        ts.isJsxElement(n) && n.openingElement.tagName.getText() === "Caption",
    ).map((n) => {
      const start = from + attr(n.openingElement, "from", 0);
      const end = start + attr(n.openingElement, "durationInFrames");
      if (end > from + duration)
        throw new Error(`Caption exceeds scene: ${id}`);
      return {
        startFrame: start,
        endFrame: end,
        text: n.children
          .map((c) => c.getText())
          .join(" ")
          .replace(/\s+/g, " ")
          .trim(),
      };
    });
    return {
      id,
      title: attr(node, "name"),
      startFrame: from,
      durationInFrames: duration,
      captions,
    };
  });
  scenes.sort((a, b) => a.startFrame - b.startFrame);
  for (let i = 1; i < scenes.length; i++) {
    if (
      scenes[i].startFrame !==
      scenes[i - 1].startFrame + scenes[i - 1].durationInFrames
    )
      throw new Error("Scene gap or overlap");
  }
  const durationInFrames =
    scenes.at(-1).startFrame + scenes.at(-1).durationInFrames;
  const captions = scenes.flatMap((s) => s.captions);
  const srt = captions
    .map(
      (c, i) =>
        `${i + 1}\n${clock(c.startFrame)} --> ${clock(c.endFrame)}\n${c.text}\n`,
    )
    .join("\n");
  const narration = scenes
    .map(
      (s) =>
        `${s.title} (${s.startFrame / fps}–${(s.startFrame + s.durationInFrames) / fps}s)\n${s.captions.map((c) => c.text).join("\n")}`,
    )
    .join("\n\n");
  await writeFile(path.join(root, `production/captions.${locale}.srt`), srt);
  await writeFile(
    path.join(root, `production/narration.${locale}.txt`),
    narration + "\n",
  );
  await writeFile(
    path.join(root, `production/storyboard.${locale}.json`),
    JSON.stringify(
      {
        locale,
        primaryPublicVersion: locale === "en",
        fps,
        durationInFrames,
        durationSeconds: durationInFrames / fps,
        audio: "Captions only; no recorded narration or music.",
        disclosure:
          "Workflow demonstrations use illustrative values and abbreviated calls. The Qwen scene uses retained measurements. The fine-tuning scene states missing evidence and the comparison design; the 12-story trainability check is excluded from method comparisons. No new target-model run or training.",
        sources,
        scenes,
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    `${locale}: ${scenes.length} scenes, ${captions.length} captions, ${durationInFrames / fps} seconds.`,
  );
}
