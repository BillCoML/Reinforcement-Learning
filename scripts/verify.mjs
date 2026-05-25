import { chromium } from "playwright";
const URL = process.argv[2] || "http://localhost:4173/";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1000 }, deviceScaleFactor: 1 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));

await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForTimeout(800);

// exercise each interactive briefly
const runs = [
  "#the-bandit-problem .bandit-arm",
  "#regret-definition regret-decomposition select",
];
for (const sel of runs) {
  const el = await page.$(sel);
  if (el) { await el.scrollIntoViewIfNeeded(); await el.click().catch(() => {}); await page.waitForTimeout(150); }
}
// run the live sims a moment
for (const sel of ["#epsilon-greedy", "#ucb1", "#thompson-sampling", "#bandits-empirical-comparison"]) {
  const btn = await page.$(`${sel} button.primary`);
  if (btn) { await btn.scrollIntoViewIfNeeded(); await btn.click(); await page.waitForTimeout(900); await btn.click().catch(()=>{}); }
}

// full-page screenshot
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(300);
await page.screenshot({ path: "/tmp/shots/fullpage.png", fullPage: true });

// minimal a11y sanity: headings, lang, labels, img alts
const a11y = await page.evaluate(() => {
  const lang = document.documentElement.lang;
  const h1 = document.querySelectorAll("h1").length;
  const h2 = document.querySelectorAll("h2").length;
  const unlabeledInputs = [...document.querySelectorAll("input,select")].filter((el) => {
    if (el.type === "range" || el.type === "checkbox") return false;
    const id = el.id;
    const lbl = id && document.querySelector(`label[for="${id}"]`);
    const wrapped = el.closest("label");
    return !lbl && !wrapped && !el.getAttribute("aria-label");
  }).length;
  const title = document.title;
  return { lang, h1, h2, unlabeledInputs, title };
});
console.log("a11y:", JSON.stringify(a11y));
console.log(errors.length ? "CONSOLE ERRORS:\n" + errors.join("\n") : "NO CONSOLE ERRORS");
await browser.close();
