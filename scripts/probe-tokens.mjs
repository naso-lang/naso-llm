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
page.on('console', msg => console.log('PAGE:', msg.text()));
await page.setRequestInterception(true);
page.on('request', (req) => {
  const u = req.url();
  const m = u.match(/huggingface\.com\/[^\/]+\/[^\/]+\/resolve\/main\/(.+)$/);
  if (m) req.continue({ url: `${modelBase}/${m[1]}` });
  else req.continue();
});

await page.goto(appUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
console.log('ready:', await waitReady(page, 150000));

// Patch the onToken callback to log token IDs
await page.evaluate(() => {
  const originalSend = window.send;
  window.send = async function(...args) {
    // We can't easily patch the internal onToken, but we can log from the page
    console.log('Send called');
  };
});

// Use a prompt that shows the issue
await page.type('#input', 'What is the capital of France?');
await page.keyboard.press('Enter');

// Wait for generation and log token IDs from the page
let done = false;
while (!done) {
  await new Promise(r => setTimeout(r, 500));
  done = await page.evaluate(() => !document.getElementById('send').disabled).catch(() => true);
}

// Get the raw text from the assistant message
const replies = await page.evaluate(() => [...document.querySelectorAll('#chat .msg.assistant .body')].map((b) => b.textContent));
console.log('Assistant replies:');
replies.forEach((r, i) => console.log(`  [${i}]`, JSON.stringify(r)));

// Also check the tokenizer's eosId and specials
const tokInfo = await page.evaluate(() => {
  // We can't access the tokenizer directly, but we can check if the special tokens appear in the text
  const body = document.querySelector('#chat .msg.assistant .body');
  return body ? body.textContent : '';
});
console.log('Raw body text:', JSON.stringify(tokInfo));

await browser.close();
rmSync(profile, { recursive: true, force: true });