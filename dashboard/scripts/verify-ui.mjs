/**
 * Verify the dashboard in a real browser, and screenshot it.
 *
 * Everything about this dashboard had until now been checked indirectly — HTTP status,
 * TypeScript types, a successful build, endpoint payloads. None of that shows whether the
 * page actually *renders*: a component that throws on a null field still type-checks and
 * still builds, and the server still returns 200 for the shell.
 *
 * This drives a real Chromium, so it catches the whole class of failures the other checks
 * cannot see — a runtime error in a panel, a fetch blocked by CORS, an empty gauge, a
 * sparkline that never draws.
 *
 * Run against a server that is already up:
 *
 *     node scripts/verify-ui.mjs                  # http://localhost:3000
 *     node scripts/verify-ui.mjs http://host:3000
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "../.ui-check");
const baseUrl = process.argv[2] ?? "http://localhost:3000";

/**
 * How long to let the page settle before judging it.
 *
 * The dashboard fetches on mount and then samples telemetry once a second, so a shot
 * taken immediately would catch the "Loading…" state and prove nothing. Six seconds is
 * enough for the first load plus several telemetry ticks, which is what the sparklines
 * need before they have two points to draw a line between.
 */
const SETTLE_MS = 6000;

async function main() {
  mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch();
  // A tall viewport so the whole dashboard is in one shot without scrolling, and a fixed
  // size so screenshots are comparable between runs.
  const page = await browser.newPage({ viewport: { width: 2500, height: 1700 } });

  // Collect the things a human would only notice by opening devtools. A page can look
  // fine and still be logging a failed fetch on every tick.
  const consoleErrors = [];
  const pageErrors = [];
  const failedRequests = [];

  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  // "pageerror" is an uncaught exception in page code — the signal that a component threw.
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) =>
    failedRequests.push(`${request.method()} ${request.url()} — ${request.failure()?.errorText}`),
  );

  console.log(`opening ${baseUrl}`);
  await page.goto(baseUrl, { waitUntil: "networkidle", timeout: 45000 });
  await page.waitForTimeout(SETTLE_MS);

  // ── what must be on the page ──
  //
  // Chosen to prove data actually arrived rather than that the shell rendered. "GPU load"
  // only appears once a telemetry reading has populated the panel, so its presence is
  // evidence the API call succeeded and the component rendered it.
  const checks = [
    ["heading", "h1"],
    ["GPU panel", "text=GPU load"],
    ["VRAM figure", "text=VRAM"],
    ["Host panel", "text=Host"],
    ["Models panel", "text=Models"],
    ["Claude usage panel", "text=Claude usage"],
    ["Local totals panel", "text=Local totals"],
    ["Call history panel", "text=Call history"],
    ["throughput metric", "text=throughput"],
  ];

  const results = [];
  for (const [label, selector] of checks) {
    const found = (await page.locator(selector).count()) > 0;
    results.push([label, found]);
    console.log(`  ${found ? "OK  " : "MISS"} ${label}`);
  }

  // Sparklines are the specific thing added for liveness, and the specific thing that
  // could silently draw nothing. Count the SVG polylines rather than the containers: an
  // empty <svg class="spark"> renders as a box and would pass a naive check.
  const sparkLines = await page.locator("svg.spark polyline").count();
  console.log(`  ${sparkLines >= 3 ? "OK  " : "MISS"} sparklines drawn: ${sparkLines}`);

  // Read the live values back out of the DOM, which is the only way to confirm the
  // browser is really receiving telemetry rather than showing placeholders.
  const gpuText = await page.locator(".spark-label").allInnerTexts();
  console.log(`  live labels: ${JSON.stringify(gpuText)}`);

  // The freshness counter is the liveness indicator. If it says anything older than a few
  // seconds, the telemetry poll is not running in the browser even if the API works.
  const freshness = await page.locator("text=/just now|\\d+s ago/").allInnerTexts();
  console.log(`  freshness: ${JSON.stringify(freshness.slice(0, 3))}`);

  await page.screenshot({ path: resolve(outDir, "dashboard.png"), fullPage: true });
  console.log(`\nscreenshot -> ${resolve(outDir, "dashboard.png")}`);

  // ── report problems ──
  if (pageErrors.length) {
    console.log("\nUNCAUGHT ERRORS IN PAGE CODE:");
    for (const error of pageErrors) console.log(`  ${error}`);
  }
  if (consoleErrors.length) {
    console.log("\nCONSOLE ERRORS:");
    for (const error of consoleErrors.slice(0, 8)) console.log(`  ${error}`);
  }
  if (failedRequests.length) {
    console.log("\nFAILED REQUESTS:");
    for (const request of failedRequests.slice(0, 8)) console.log(`  ${request}`);
  }

  await browser.close();

  const missing = results.filter(([, found]) => !found).map(([label]) => label);
  const ok =
    missing.length === 0 && pageErrors.length === 0 && sparkLines >= 3;

  console.log(
    ok
      ? "\nUI VERIFIED: every panel rendered, sparklines drawing, no page errors."
      : `\nUI PROBLEMS: missing=${JSON.stringify(missing)} pageErrors=${pageErrors.length} sparklines=${sparkLines}`,
  );
  // Non-zero exit so this can gate a script rather than needing to be read.
  process.exit(ok ? 0 : 1);
}

main().catch((error) => {
  console.error("verify-ui failed:", error);
  process.exit(1);
});
