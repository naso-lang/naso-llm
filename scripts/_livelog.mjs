#!/usr/bin/env node
import puppeteer from 'puppeteer-core';
import { existsSync, rmSync } from 'node:fs';
const appUrl = 'https://naso-lang.github.io/naso-llm/';
const chromePath = ['/usr/bin/chromium','/usr/bin/chromium-browser','/usr/bin/google-chrome'].find(p=>existsSync(p));
const profile='/var/tmp/livelog-profile'; rmSync(profile,{recursive:true,force:true});
const browser = await puppeteer.launch({executablePath:chromePath,headless:true,userDataDir:profile,args:['--no-sandbox','--disable-dev-shm-usage']});
const page = await browser.newPage();
const logs=[];
page.on('console',m=>logs.push(m.type()+': '+m.text()));
await page.goto(appUrl,{waitUntil:'domcontentloaded',timeout:60000});
const t0=Date.now(); let state='';
while(Date.now()-t0<120000){ try{state=await page.evaluate(()=>`${document.getElementById('dot-model')?.className} | ${document.getElementById('model-state')?.textContent}`);}catch{state='PF';} if(/dot ok|dot err/.test(state))break; await new Promise(r=>setTimeout(r,1000)); }
const applog = await page.evaluate(()=>[...document.querySelectorAll('#log div')].map(d=>d.textContent));
const cacheTry = await page.evaluate(async()=>{
  const U='https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors';
  const out={hasCaches:'caches' in window};
  try{ const r=await fetch(U,{method:'GET'}); out.fetchStatus=r.status; out.type=r.type;
    const c=await caches.open('naso-llm-models-v1');
    try{ await c.put(U, r.clone()); out.putOk=true; }catch(e){ out.putErr=String(e); }
    out.keys=(await c.keys()).map(k=>k.url.split('/').pop().slice(0,20));
  }catch(e){ out.err=String(e); }
  return out;
});
console.log('state:',state);
console.log('app log:'); for(const l of applog) console.log('  ',l);
console.log('cacheTry:',JSON.stringify(cacheTry));
console.log('console:'); for(const l of logs.filter(l=>/main|error|warn|fail/i.test(l)).slice(0,15)) console.log('  ',l);
await browser.close(); rmSync(profile,{recursive:true,force:true});
