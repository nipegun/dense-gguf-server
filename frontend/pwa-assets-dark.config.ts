import { fWriteThemeFavicons } from './scripts/favicon-colorize';
import { cFaviconColors, cPwaAssetGenerator } from './src/lib/constants/pwa.constants';
import { defineConfig } from '@vite-pwa/assets-generator/config';

fWriteThemeFavicons(cFaviconColors.LIGHT, cFaviconColors.DARK, {
  padding: cPwaAssetGenerator.FAVICON_PADDING
});

export default defineConfig({
  headLinkOptions: {
    preset: '2023'
  },
  images: ['static/favicon-dark.svg'],
  preset: {
    apple: {
      sizes: []
    },
    maskable: {
      sizes: []
    },
    transparent: {
      favicons: [[48, 'favicon-dark.ico']],
      padding: cPwaAssetGenerator.FAVICON_PADDING,
      sizes: []
    }
  }
});
