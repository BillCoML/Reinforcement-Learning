import { chromium } from "playwright";
const browser = await chromium.launch();

// 1) V8 re-shoot
{
  const page = await browser.newPage({ viewport: { width: 1240, height: 900 }, deviceScaleFactor: 2 });
  await page.goto("http://localhost:5191/", { waitUntil: "networkidle" });
  await page.waitForTimeout(700);
  const el = await page.$("#bandits-forward-links roadmap-mini");
  await el.scrollIntoViewIfNeeded();
  await el.screenshot({ path: "/tmp/shots/v8b.png" });
  await page.close();
}

// 2) reduced-motion: V4 should still advance via Run/Step without errors
{
  const page = await browser.newPage({ viewport: { width: 1000, height: 800 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("http://localhost:5191/", { waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  const sel = "#epsilon-greedy epsilon-greedy-explorer";
  await page.click(`${sel} button.primary`);
  await page.waitForTimeout(1500);
  const status = await page.$eval(`${sel} .panel-status`, (e) => e.textContent);
  console.log("reduced-motion V4 advanced:", status, errors.length ? "ERRORS:" + errors : "(no errors)");
  await page.close();
}

// 3) print media: controls + header hidden inside a heavy panel
{
  const page = await browser.newPage({ viewport: { width: 1000, height: 800 } });
  await page.goto("http://localhost:5191/", { waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  await page.emulateMedia({ media: "print" });
  const headerVisible = await page.$eval("#ucb1 .rl-panel__header", (e) => getComputedStyle(e).display);
  const controlsVisible = await page.$eval("#ucb1 .rl-controls", (e) => getComputedStyle(e).display);
  console.log("print: header display =", headerVisible, "| controls display =", controlsVisible);
  await page.close();
}

// 4) keyboard: V1 arm cards are focusable
{
  const page = await browser.newPage({ viewport: { width: 1000, height: 800 } });
  await page.goto("http://localhost:5191/", { waitUntil: "networkidle" });
  await page.waitForTimeout(400);
  const tab = await page.$eval("#the-bandit-problem .bandit-arm", (e) => e.tabIndex);
  console.log("V1 arm card tabIndex:", tab);
  await page.close();
}

await browser.close();
