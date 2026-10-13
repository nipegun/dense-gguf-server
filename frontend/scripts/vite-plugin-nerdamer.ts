import { build } from 'esbuild';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import type { Plugin } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));
const cVendorsDir = resolve(__dirname, '../src/lib/vendors');
const cVirtualId = 'virtual:nerdamer';
const cResolvedId = '\0' + cVirtualId;

/**
 * Bundle the vendored nerdamer-prime source into a minified IIFE string,
 * exposed as the `virtual:nerdamer` module. Flags mirror the upstream
 * build (esbuild --bundle --minify --format=iife --global-name=nerdamer),
 * so only human readable source lives in the repo and minification is a
 * build artifact. Vendored under src/lib/vendors/, upstream snapshot:
 * https://github.com/together-science/nerdamer-prime/commit/1936145f8af306ec0d883b9bfd7730aedd175c24
 */
export function fNerdamerPlugin(): Plugin {
  let vBundled: string | null = null;

  return {
    async load(pId) {
      if (pId !== cResolvedId) return undefined;

      if (vBundled === null) {
        const cResult = await build({
          alias: {
            'big-integer': resolve(cVendorsDir, 'big-integer/BigInteger.js'),
            'decimal.js': resolve(cVendorsDir, 'decimal.js/decimal.js')
          },
          bundle: true,
          entryPoints: [resolve(cVendorsDir, 'nerdamer-prime/all.js')],
          format: 'iife',
          globalName: 'nerdamer',
          logLevel: 'silent',
          minify: true,
          write: false
        });

        vBundled = cResult.outputFiles[0].text;
      }

      return `export default ${JSON.stringify(vBundled)};`;
    },
    name: 'dgscpp:nerdamer',
    resolveId(pId) {
      return pId === cVirtualId ? cResolvedId : undefined;
    }
  };
}
