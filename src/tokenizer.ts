/**
 * Byte-level BPE tokenizer, ported from the Hugging Face `tokenizers` spec.
 *
 * This is a real tokenizer, not a vocabulary lookup: it reproduces the exact
 * pipeline declared in the model's `tokenizer.json` --
 *
 *   Sequence[
 *     Digits(individual_digits = true),   // "123" -> "1","2","3"
 *     ByteLevel(add_prefix_space=false, use_regex=true),
 *   ]
 *   BPE(ignore_merges=false)
 *
 * -- so the browser tokenises with no server round-trip, which is what lets the
 * chat demo work offline. `tools/tokenizer_oracle.py` produces reference ids
 * with the real Rust `tokenizers` library; `tools/verify_tokenizer.mjs` diffs
 * this port against them over a fixed corpus.
 */

/** GPT-2 pre-tokenization pattern (also used by Llama/Qwen/SmolLM). */
const GPT2_SPLIT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/**
 * GPT-2's reversible byte -> printable-unicode map. All 256 byte values map to a
 * single printable codepoint, so no token contains a character that survives
 * badly through JSON or HTML.
 */
function buildByteMaps(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i); // '!'..'~'
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i); // '¡'..'¬'
  for (let i = 0xae; i <= 0xff; i++) bs.push(i); // '®'..'ÿ'

  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const byteToChar: string[] = new Array(256);
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

export interface TokenizerJSON {
  model: { vocab: Record<string, number>; merges: string[] };
  added_tokens?: Array<{ id: number; content: string; special?: boolean }>;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * The ChatML control strings this tokenizer's model was trained on.
 *
 * Written as \u escapes rather than literals on purpose: these exact strings
 * must match entries in the checkpoint's `added_tokens` (`` = 1,
 * `` = 2). Getting them wrong is invisible -- `encode(s, true)` simply
 * fails to match and falls through to byte-level pieces -- and the consequence
 * is severe: a SmolLM2-Instruct model fed bare `user\n...\nassistant\n` text
 * with no control tokens does not answer, it loops and hallucinates fake turns.
 */
const IM_START = '\u003c\u007cim_start\u007c\u003e';
const IM_END = '\u003c\u007cim_end\u007c\u003e';

export class BPETokenizer {
  private readonly vocab: Map<string, number>;
  private readonly ranks: Map<string, number>;
  private readonly idToToken: string[];
  private readonly byteToChar: string[];
  private readonly charToByte: Map<string, number>;
  /** Special strings (e.g. `<|im_start|>`) matched literally, longest first. */
  private readonly specials: string[];
  /** Token closing an assistant turn; generation stops on it. */
  readonly eosId: number;
  readonly imStartId: number;

  private constructor(json: TokenizerJSON) {
    const { byteToChar, charToByte } = buildByteMaps();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;

    this.vocab = new Map(Object.entries(json.model.vocab));
    const maxId = Math.max(...Object.values(json.model.vocab));
    this.idToToken = new Array(maxId + 1);
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;

    this.ranks = new Map();
    json.model.merges.forEach((m, i) => {
      const sp = m.indexOf(' ');
      this.ranks.set(m.slice(0, sp) + ' ' + m.slice(sp + 1), i);
    });

    const added = json.added_tokens ?? [];
    for (const a of added) {
      this.vocab.set(a.content, a.id);
      this.idToToken[a.id] = a.content;
    }
    this.specials = added
      .filter((a) => a.special)
      .map((a) => a.content)
      .sort((a, b) => b.length - a.length);
    this.eosId = this.vocab.get(IM_END) ?? this.vocab.get('\u003c\u007cendoftext\u007c\u003e') ?? 2;
    this.imStartId = this.vocab.get(IM_START) ?? 0;
  }

  static fromJSON(json: unknown): BPETokenizer {
    return new BPETokenizer(json as TokenizerJSON);
  }

  /** Split into pre-tokens: Digits first, then GPT-2's byte-level rule. */
  private preTokenize(text: string): string[] {
    // Digits(individual_digits=true): each digit is its own pre-token.
    const digitPieces: string[] = [];
    let run = '';
    for (const ch of text) {
      if (ch >= '0' && ch <= '9') {
        if (run) {
          digitPieces.push(run);
          run = '';
        }
        digitPieces.push(ch);
      } else {
        run += ch;
      }
    }
    if (run) digitPieces.push(run);

    const out: string[] = [];
    for (const piece of digitPieces) {
      const words = piece.match(GPT2_SPLIT);
      if (words) out.push(...words);
      else if (piece) out.push(piece);
    }
    return out;
  }

  /** Greedy lowest-rank pair merging -- the reference BPE algorithm. */
  private bpe(token: string): string[] {
    let word = [...token];
    if (word.length < 2) return word;

    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(word[i] + ' ' + word[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex === -1) break;

      const pair = word[bestIndex] + ' ' + word[bestIndex + 1];
      const merged: string[] = [];
      for (let i = 0; i < word.length; ) {
        if (i < word.length - 1 && word[i] + ' ' + word[i + 1] === pair) {
          merged.push(word[i] + word[i + 1]);
          i += 2;
        } else {
          merged.push(word[i]);
          i += 1;
        }
      }
      word = merged;
    }
    return word;
  }

  /** UTF-8 bytes of `text`, each mapped to its printable byte-level char. */
  private toByteLevel(text: string): string {
    const bytes = new TextEncoder().encode(text);
    let out = '';
    for (const b of bytes) out += this.byteToChar[b];
    return out;
  }

  /**
   * Encode `text`. Special tokens (the chat control strings) are matched
   * literally when `allowSpecial` is set, so `<|im_start|>` becomes one id
   * rather than nine byte-level merges.
   */
  encode(text: string, allowSpecial = false): number[] {
    const ids: number[] = [];
    const chunks: Array<{ text: string; special: boolean }> = [];
    if (allowSpecial && this.specials.length) {
      let rest = text;
      while (rest) {
        let hit = -1;
        let hitTok = '';
        for (const s of this.specials) {
          const at = rest.indexOf(s);
          if (at !== -1 && (hit === -1 || at < hit)) {
            hit = at;
            hitTok = s;
          }
        }
        if (hit === -1) {
          chunks.push({ text: rest, special: false });
          break;
        }
        if (hit > 0) chunks.push({ text: rest.slice(0, hit), special: false });
        chunks.push({ text: hitTok, special: true });
        rest = rest.slice(hit + hitTok.length);
      }
    } else {
      chunks.push({ text, special: false });
    }

    for (const chunk of chunks) {
      if (chunk.special) {
        const id = this.vocab.get(chunk.text);
        if (id !== undefined) ids.push(id);
        continue;
      }
      for (const word of this.preTokenize(chunk.text)) {
        for (const sym of this.bpe(this.toByteLevel(word))) {
          const id = this.vocab.get(sym);
          if (id !== undefined) ids.push(id);
        }
      }
    }
    return ids;
  }

  /** Decode ids to text (byte-level chars -> bytes -> UTF-8). */
  decode(ids: number[]): string {
    let byteStr = '';
    for (const id of ids) {
      const tok = this.idToToken[id];
      if (tok === undefined) continue;
      if (this.specials.includes(tok)) {
        byteStr += tok;
        continue;
      }
      for (const ch of tok) {
        const b = this.charToByte.get(ch);
        if (b !== undefined) byteStr += String.fromCharCode(b);
      }
    }
    // Reassemble the byte string, then decode as UTF-8 with replacement so a
    // half-generated multi-byte character never throws mid-stream.
    const bytes = new Uint8Array(byteStr.length);
    for (let i = 0; i < byteStr.length; i++) bytes[i] = byteStr.charCodeAt(i) & 0xff;
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  }

  get size(): number {
    return this.idToToken.length;
  }

  idOf(token: string): number | undefined {
    return this.vocab.get(token);
  }

  /**
   * ChatML template (SmolLM2 / Qwen family), matching the model's
   * `tokenizer_config.json`. The `` / `` markers are required, not
   * cosmetic: without them the model does not recognise turn boundaries at all
   * and generates fake `user`/`assistant` lines inside its own reply.
   *
   * `addGenerationPrompt` opens the assistant turn so the model continues from it.
   */
  static chatTemplate(
    messages: ChatMessage[],
    addGenerationPrompt = true,
    systemPrompt?: string,
  ): string {
    let out = '';
    const sys = systemPrompt ?? messages.find((m) => m.role === 'system')?.content;
    if (sys) out += `${IM_START}system\n${sys}${IM_END}\n`;
    for (const m of messages) {
      if (m.role === 'system') continue;
      out += `${IM_START}${m.role}\n${m.content}${IM_END}\n`;
    }
    if (addGenerationPrompt) out += `${IM_START}assistant\n`;
    return out;
  }
}

/** Fetch and parse the tokenizer for a HF repo id. */
export async function loadTokenizer(repo: string): Promise<BPETokenizer> {
  const url = `https://huggingface.co/${repo}/resolve/main/tokenizer.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  return BPETokenizer.fromJSON(await res.json());
}
