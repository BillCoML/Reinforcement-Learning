import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1100 }, deviceScaleFactor: 2 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("PE:" + e.message));
await page.goto("http://localhost:5191/", { waitUntil: "networkidle" });
await page.waitForTimeout(600);
const sel = "#bandits-empirical-comparison .rl-panel.arena";
const el = await page.$(sel);
await el.scrollIntoViewIfNeeded();
await page.click(`${sel} button.primary`); // Run
await page.waitForTimeout(3500);
await el.screenshot({ path: process.argv[2] || "/tmp/shots/v7-done.png" });
console.log(errors.length ? "ERR:\n" + errors.join("\n") : "no console errors");
await browser.close();
