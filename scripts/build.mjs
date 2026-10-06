// Programmatic production build (invoked by `npm run build`).
// Using the Vite JS API rather than the CLI keeps this runnable from any
// harness and gives a real exit code on failure.
import { build } from 'vite';

try {
  await build({ logLevel: 'warn' });
  console.log('build: ok');
  process.exit(0);
} catch (e) {
  console.error('build failed:', e);
  process.exit(1);
}
