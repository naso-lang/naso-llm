import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';

const appUrl = process.argv[2] ?? 'http://localhost:3000/';
const modelBase = process.argv[3] ?? 'http://127.0.0.1:8777';
const chromePath = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']
  .find((p) => existsSync(p));
if (!chromePath) { console.error('no chromium'); process.exit(2); }

const profile = '/var/tmp/probe-profile';
rmSync(profile, { recursive: true, force: true });
const browser = await puppeteer.launch({
  executablePath: chromePath, headless: true, userDataDir: profile,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

async function waitReady(page, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const s = await page.evaluate(() => {
      const d = document.getElementById('dot-model');
      return `${d?.className} | ${document.getElementById('model-state')?.textContent}`;
    }, { timeout: 5000 });
    if (/dot ok|dot err/.test(s)) return s;
    await new Promise((r) => setTimeout(r, 1000));
  }
  return 'TIMEOUT';
}

const page = await browser.newPage();
await page.setRequestInterception(true);
page.on('request', (req) => {
  const u = req.url();
  const m = u.match(/huggingface\.[^/]+\/[^/]+\/[^/]+\/resolve\/main\/(.+)$/);
  if (m) req.continue({ url: `${modelBase}/${m[1]}` });
  else req.continue();
});

await page.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
console.log('ready:', await waitReady(page, 150000));

// Same prompt as e2e test
await page.type('#input', 'What is the capital of France?');
await page.keyboard.press('Enter');

let done = false;
while (!done) {
  await new Promise(r => setTimeout(r, 500));
  done = await page.evaluate(() => !document.getElementById('send').disabled).catch(() => true);
}

const replies = await page.evaluate(() => [...document.querySelectorAll('#chat .msg.assistant .body')].map((b) => b.textContent));
console.log('Assistant replies (same prompt as e2e):');
replies.forEach((r, i) => console.log(`  [${i}]`, JSON.stringify(r)));

await browser.close();
rmSync(profile, { recursive: true, force: true });