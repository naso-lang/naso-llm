/**
 * Tokenizer file reader.
 *
 * This demo does not run a BPE encoder -- a correct one is a meaningful amount
 * of code and is orthogonal to the point being demonstrated (that Naso's
 * verified compiler produces the quantization kernels the GPU runs). What it
 * DOES do is read `tokenizer.json`, find the vocabulary, and decode. To feed
 * the model it needs token ids, so it uses the vocabulary's id<->token tables
 * directly:
 *
 *   * `encodeSimple(text)` lower-cases and splits on word boundaries, maps each
 *     piece to a vocab id, and wraps the result in the model's BOS/EOS pair.
 *     This is a lookup, not a BPE merge -- it is exact for tokens already in the
 *     vocab and emits `<unk>` for the rest.
 *   * `decode(ids)` concatenates the vocabulary strings, which is what turns
 *     the final logits into a readable next token.
 *
 * The limitation is stated here rather than hidden: any prompt whose tokens are
 * not literal vocabulary entries tokenizes as `<unk>`. That is enough to run a
 * forward pass and read a next token; it is not a general-purpose tokenizer.
 */

export interface SimpleTokenizer {
  vocab: Map<string, number>;
  idToToken: string[];
  bosId: number;
  eosId: number;
  unkId: number;
  /** Encode by vocabulary lookup; see the caveat in the module comment. */
  encodeSimple(text: string): number[];
  decode(ids: number[]): string;
  idFor(token: string): number | undefined;
}

interface AddedToken {
  id?: number;
  content?: string;
}

interface TokenizerJson {
  model?: { vocab?: Record<string, number>; unk_token?: string };
  added_tokens?: AddedToken[];
  added_tokens_decoder?: Record<string, { content: string; special?: boolean }>;
}

/** Parse a tokenizer.json into a lookup-based tokenizer. */
export function parseTokenizer(json: TokenizerJson): SimpleTokenizer {
  const vocab = new Map<string, number>();
  const idToToken: string[] = [];

  // The base vocabulary: token string -> id.
  if (json.model?.vocab) {
    for (const [token, id] of Object.entries(json.model.vocab)) {
      vocab.set(token, id);
      idToToken[id] = token;
    }
  }

  // Added tokens may extend the vocabulary (new ids) or override an id's string
  // (e.g. `<|endoftext|>`). Both are applied.
  const added = json.added_tokens ?? [];
  for (const at of added) {
    if (at.content !== undefined && at.id !== undefined) {
      vocab.set(at.content, at.id);
      idToToken[at.id] = at.content;
    }
  }
  if (json.added_tokens_decoder) {
    for (const [idStr, info] of Object.entries(json.added_tokens_decoder)) {
      const id = Number(idStr);
      if (Number.isFinite(id)) idToToken[id] = info.content;
    }
  }

  // Fill holes so idToToken is dense (some vocabs skip ids).
  for (let i = 0; i < idToToken.length; i++) {
    if (idToToken[i] === undefined) idToToken[i] = '';
  }

  const unkToken = json.model?.unk_token ?? '<unk>';
  const unkId = vocab.get(unkToken) ?? 0;
  const bosId = vocab.get('<s>') ?? vocab.get('<|begin_of_text|>') ?? vocab.get('<s>') ?? -1;
  const eosId = vocab.get('</s>') ?? vocab.get('<|end_of_text|>') ?? vocab.get('<|endoftext|>') ?? -1;

  return {
    vocab,
    idToToken,
    bosId,
    eosId,
    unkId,
    idFor(token: string) { return vocab.get(token); },
    encodeSimple(text: string): number[] {
      // A minimal word/punctuation split. Each piece is looked up directly; a
      // piece not in the vocab becomes <unk>.
      const pieces = text.match(/[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]/g) ?? [];
      const ids: number[] = [];
      if (bosId >= 0) ids.push(bosId);
      for (const piece of pieces) {
        const direct = vocab.get(piece);
        if (direct !== undefined) { ids.push(direct); continue; }
        const lowered = vocab.get(piece.toLowerCase());
        if (lowered !== undefined) { ids.push(lowered); continue; }
        // Try the "▁"-prefixed form SentencePiece maps a leading space to.
        const sp = vocab.get(`\u2581${piece.toLowerCase()}`) ?? vocab.get(`\u2581${piece}`);
        ids.push(sp ?? unkId);
      }
      return ids;
    },
    decode(ids: number[]): string {
      let out = '';
      for (const id of ids) {
        const tok = idToToken[id];
        if (tok !== undefined) out += tok;
      }
      return out;
    },
  };
}

/** Fetch and parse the tokenizer for a HF repo id. */
export async function loadTokenizer(repo: string): Promise<SimpleTokenizer> {
  const url = `https://huggingface.co/${repo}/resolve/main/tokenizer.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  return parseTokenizer((await res.json()) as TokenizerJson);
}
