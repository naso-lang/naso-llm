import { readFileSync } from 'node:fs';
import { BPETokenizer } from '/home/node/naso-llm/src/tokenizer.ts';

const tok = BPETokenizer.fromJSON(JSON.parse(readFileSync('/var/tmp/smol/tokenizer.json', 'utf8')));

// Test with filterSpecials=false (default)
const ids1 = [1, 2, 3]; // IM_START, IM_END, <repo_name>
const decoded1 = tok.decode(ids1, false);
console.log('filterSpecials=false:', JSON.stringify(decoded1));

// Test with filterSpecials=true
const decoded2 = tok.decode(ids1, true);
console.log('filterSpecials=true:', JSON.stringify(decoded2));

// Test eosId
console.log('eosId:', tok.eosId);
console.log('imStartId:', tok.imStartId);
console.log('specials:', tok.specials);