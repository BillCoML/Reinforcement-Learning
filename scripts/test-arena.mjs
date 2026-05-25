import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1100 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("PE:" + e.message));
const sel = "#bandits-empirical-comparison algorithm-battle-arena";

// 1) URL share: only UCB + Thompson
await page.goto("http://localhost:5191/?seed=7&algos=ucb,ts", { waitUntil: "networkidle" });
await page.waitForTimeout(500);
const checks = await page.$$eval(`${sel} .arena-algo-row input`, (els) => els.map((e) => e.checked));
console.log("algos enabled (rand,eps001,eps01,ucb,ts,custom):", checks.join(","));

// 2) K=8 run
await page.$eval(`${sel} .arena-rail input[type=range]`, (el) => {
  el.value = "8";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
await page.waitForTimeout(200);
await page.click(`${sel} button.primary`);
await page.waitForTimeout(3000);
const status = await page.$eval(`${sel} .panel-status`, (e) => e.textContent);
console.log("K=8 run status:", status);

// 3) Export PNG triggers a download
await page.goto("http://localhost:5191/", { waitUntil: "networkidle" });
await page.waitForTimeout(400);
await page.click(`${sel} button.primary`);
await page.waitForTimeout(2500);
const dl = page.waitForEvent("download", { timeout: 4000 }).catch(() => null);
await page.click(`${sel} >> text=Export PNG`);
const download = await dl;
console.log("PNG download:", download ? download.suggestedFilename() : "FAILED");

// 4) URL updated after toggling an algorithm
await page.click(`${sel} .arena-algo-row input >> nth=0`); // toggle Random off
await page.waitForTimeout(200);
console.log("URL after toggle:", new URL(page.url()).search);

console.log(errors.length ? "CONSOLE ERRORS:\n" + errors.join("\n") : "no console errors");
await browser.close();
