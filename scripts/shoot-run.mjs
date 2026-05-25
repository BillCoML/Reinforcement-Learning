// Screenshot after running a component: node scripts/shoot-run.mjs <url> <out> <selector> <playMs> [stepClicks]
import { chromium } from "playwright";
const [, , url, outfile, selector, playMs, stepClicks] = process.argv;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1240, height: 1000 }, deviceScaleFactor: 2 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));

await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(600);

const el = await page.$(selector);
if (!el) { console.error("not found", selector); process.exit(2); }
await el.scrollIntoViewIfNeeded();

if (stepClicks) {
  const step = await page.$(`${selector} >> text=Step`);
  for (let i = 0; i < +stepClicks; i++) { await step?.click(); await page.waitForTimeout(350); }
} else {
  const run = await page.$(`${selector} button.primary`);
  await run?.click();
  await page.waitForTimeout(+playMs || 2500);
  const pause = await page.$(`${selector} button.primary`);
  await pause?.click(); // pause for a clean shot
}
await page.waitForTimeout(400);
await el.screenshot({ path: outfile });
console.log(errors.length ? "CONSOLE ERRORS:\n" + errors.join("\n") : "no console errors");
console.log("wrote", outfile);
await browser.close();
