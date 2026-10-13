import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { mdsvex } from 'mdsvex';

// CMake sets DGS_UI_OUT_DIR to the staging dir under the build tree; manual
// `npm run build` runs without the env var default to ./dist.
const cOutDir = process.env.DGS_UI_OUT_DIR ?? './dist';
/** @type {import('@sveltejs/kit').Config} */
const cConfig = {
  extensions: ['.svelte', '.svx'],

  kit: {
    adapter: adapter({
      assets: cOutDir,
      fallback: 'index.html',
      pages: cOutDir,
      precompress: false,
      strict: true
    }),
    output: {
      bundleStrategy: 'single'
    },
    paths: {
      relative: true
    },
    router: { type: 'hash' }
  },

  // Consult https://svelte.dev/docs/kit/integrations
  // for more information about preprocessors
  preprocess: [vitePreprocess(), mdsvex()]
};

export default cConfig;
