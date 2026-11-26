// Loads a store product page in headless Chromium and checks the 3D viewer starts:
// no page errors, the loading message disappears, and the idle sway moves the picture.
// usage: node product_page.cjs <page url> <dir holding the three@0.160.0 npm package>
const { chromium } = require("playwright");
const path = require("path");
(async () => {
  const [url, threeDir] = process.argv.slice(2);
  const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  await page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**", r => {
    const p = new URL(r.request().url()).pathname.replace("/npm/three@0.160.0", "");
    r.fulfill({ path: path.join(threeDir, p), contentType: "text/javascript" });
  });
  await page.route("https://fonts.googleapis.com/**", r => r.abort());
  const errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  await page.goto(url);
  const loaded = await page.waitForFunction(() => document.getElementById("status").hidden, null, { timeout: 120000 })
    .then(() => true, () => false);
  let moved = false;
  if (loaded) {
    const a = await page.locator("#stage").screenshot();
    await page.waitForTimeout(2500);
    const b = await page.locator("#stage").screenshot();
    moved = !a.equals(b);
  }
  console.log(JSON.stringify({ errors, loaded, moved }));
  await browser.close();
})();
