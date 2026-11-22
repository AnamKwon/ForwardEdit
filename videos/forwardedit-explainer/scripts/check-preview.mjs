import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { chromium } from "playwright-core";

let executablePath = process.env.FORWARDEDIT_CHROME;
if (!executablePath)
  for (const command of ["google-chrome", "chromium", "chromium-browser"]) {
    try {
      executablePath = execFileSync("which", [command], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }).trim();
      break;
    } catch {
      /* Try the next installed browser. */
    }
  }
if (!executablePath)
  throw new Error(
    "Set FORWARDEDIT_CHROME to an installed Chrome executable for preview checks.",
  );
await mkdir("out/qa", { recursive: true });
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ["--no-sandbox"],
});
const errors = [];
const checks = [];
try {
  const page = await browser.newPage({
    viewport: { width: 2560, height: 1600 },
    deviceScaleFactor: 1,
  });
  // Use the real Studio WebMCP callbacks in a headless browser without a WebMCP host.
  // Only preview seeking and read-only playback inspection are invoked.
  await page.addInitScript(() => {
    const tools = new Map();
    window.__previewTools = tools;
    Object.defineProperty(document, "modelContext", {
      configurable: true,
      value: {
        registerTool(tool, options) {
          tools.set(tool.name, tool);
          options?.signal?.addEventListener("abort", () => {
            if (tools.get(tool.name) === tool) tools.delete(tool.name);
          });
          return Promise.resolve();
        },
      },
    });
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  for (const [locale, composition] of [
    ["en", "ForwardEdit-English"],
    ["ko", "ForwardEdit-Korean"],
  ]) {
    await mkdir(`out/qa/${locale}`, { recursive: true });
    await page.goto(
      `${process.env.FORWARDEDIT_STUDIO_URL || "http://localhost:3333"}/${composition}`,
      { waitUntil: "networkidle", timeout: 60000 },
    );
    await page.waitForFunction(() =>
      window.__previewTools.has("seek_to_frame"),
    );
    for (const [name, second, en, ko] of [
      ["intro", 3, "Read the weights.", "가중치를 읽고,"],
      ["roles", 17, "External LLM", "외부 LLM"],
      ["observe", 33, "describe_parameter()", "describe_parameter()"],
      ["hypothesis", 48, "prepare_edit_proposal()", "prepare_edit_proposal()"],
      ["write-before", 57, "0.120", "0.120"],
      ["write-after", 62, "0.110", "0.110"],
      ["feedback", 77, "Revise or restore", "후보 B 또는 복원"],
      ["confirmation", 89, "Independent test", "독립 데이터"],
      ["recorded-results", 101, "103/192", "103/192"],
      ["fine-tuning-comparison-needed", 118, "Not measured", "미측정"],
      ["recap", 132, "original protocol decisions", "원래 규칙의 판정"],
    ]) {
      const frame = second * 30;
      await page.evaluate(async (f) => {
        await window.__previewTools.get("seek_to_frame").execute({ frame: f });
      }, frame);
      await page.waitForFunction(
        (f) =>
          window.__previewTools
            .get("get_playback_state")
            .execute({})
            .then((r) => r.currentFrame === f),
        frame,
      );
      await page.evaluate(() => document.fonts.ready);
      await page.waitForLoadState("networkidle");
      const film = page.getByTestId("forwardedit-film");
      const text = await film.innerText();
      if (!text.includes(locale === "en" ? en : ko))
        throw new Error(`Missing scene content: ${locale}/${name}`);
      if (locale === "en" && /[가-힣]/.test(text))
        throw new Error(`Korean text in public English edition: ${name}`);
      if (name === "observe") {
        if ((await film.locator("img").count()) !== 1)
          throw new Error(`Missing inspection asset: ${locale}`);
        await film
          .locator("img")
          .evaluateAll((images) =>
            Promise.all(images.map((image) => image.decode())),
          );
      }
      if (
        name === "write-before" &&
        (await film.getByText("0.120", { exact: true }).count()) !== 1
      )
        throw new Error("Missing original matrix value");
      if (
        name === "write-after" &&
        (await film.getByText("0.120", { exact: true }).count()) !== 0
      )
        throw new Error("Matrix value did not change");
      await film.screenshot({ path: `out/qa/${locale}/${name}.png` });
      checks.push({ locale, name, frame, expectedTextFound: true });
    }
    await page.screenshot({
      path: `out/qa/${locale}/studio.png`,
      fullPage: true,
    });
  }
} finally {
  await browser.close();
}
await writeFile(
  "out/qa/report.json",
  JSON.stringify({ checks, browserErrors: errors }, null, 2) + "\n",
);
if (errors.length) throw new Error(errors.join("\n"));
console.log(
  `Checked ${checks.length} bilingual preview positions; no browser errors. Screenshots: out/qa/`,
);
