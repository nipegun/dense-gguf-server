import { fBuildInfoPlugin } from './scripts/vite-plugin-build-info';
import { fNerdamerPlugin } from './scripts/vite-plugin-nerdamer';
import { fRelativizeBasePlugin } from './scripts/vite-plugin-relativize-base';
import { fSplashScreenPlugin } from './scripts/vite-plugin-splash-screen';
import { cSveltekitPwaOptions } from './src/lib/constants/pwa.constants';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { defineConfig, loadEnv, searchForWorkspaceRoot } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));
export default defineConfig(({ mode: vMode }) => {
  const dEnv = loadEnv(vMode, process.cwd(), 'VITE_PUBLIC_');
  const cServerOrigin = dEnv.VITE_PUBLIC_SERVER_ORIGIN || 'http://localhost:9931';

  return {
    build: {
      assetsInlineLimit: 32000,
      chunkSizeWarningLimit: 3072,
      minify: true
    },

    plugins: [
      tailwindcss(),
      sveltekit(),
      SvelteKitPWA(cSveltekitPwaOptions),
      fSplashScreenPlugin(),
      fBuildInfoPlugin(),
      fNerdamerPlugin(),
      fRelativizeBasePlugin()
    ],

    resolve: {
      alias: {
        'katex-fonts': resolve('node_modules/katex/dist/fonts')
      }
    },

    server: {
      fs: {
        allow: [searchForWorkspaceRoot(process.cwd())]
      },
      headers: {
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Opener-Policy': 'same-origin'
      },
      proxy: {
        '/api': cServerOrigin
      }
    }
  };
});
