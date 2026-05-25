// Screenshot helper: node scripts/shoot.mjs <url> <outfile> [selector] [waitMs]
import { chromium } from "playwright";

const [, , url, outfile, selector, waitMs] = process.argv;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1240, height: 900 }, deviceScaleFactor: 2 });

const errors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
page.on("pageerror", (err) => errors.push("PAGEERROR: " + err.message));

await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(waitMs ? +waitMs : 700);

if (selector) {
  const el = await page.$(selector);
  if (!el) {
    console.error("selector not found:", selector);
    process.exit(2);
  }
  await el.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await el.screenshot({ path: outfile });
} else {
  await page.screenshot({ path: outfile, fullPage: true });
}

if (errors.length) {
  console.error("CONSOLE ERRORS:\n" + errors.join("\n"));
} else {
  console.log("no console errors");
}
console.log("wrote", outfile);
await browser.close();
