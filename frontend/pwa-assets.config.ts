import { fWriteThemeFavicons } from './scripts/favicon-colorize';
import {
  cFaviconColors,
  cPwaAssetGenerator,
  aPwaGeneratorDevices,
  cThemeColors
} from './src/lib/constants/pwa.constants';
import { SplashOrientation } from './src/lib/enums/splash.enums';
import {
  combinePresetAndAppleSplashScreens,
  defineConfig,
  minimal2023Preset
} from '@vite-pwa/assets-generator/config';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

fWriteThemeFavicons(cFaviconColors.LIGHT, cFaviconColors.DARK, {
  padding: cPwaAssetGenerator.FAVICON_PADDING
});

export default defineConfig({
  headLinkOptions: {
    preset: cPwaAssetGenerator.LINK_PRESET
  },
  images: ['static/favicon.svg'],
  preset: combinePresetAndAppleSplashScreens(
    {
      ...minimal2023Preset,
      // tiny margin so favicon.ico / pwa-*.png breathe inside the canvas
      transparent: {
        ...minimal2023Preset.transparent,
        padding: cPwaAssetGenerator.FAVICON_PADDING
      }
    },
    {
      darkImageResolver: async (pImageName: string) => {
        if (pImageName.endsWith('favicon.svg')) {
          return readFileSync(resolve('static/favicon-dark.svg'));
        }
      },
      darkResizeOptions: {
        background: cThemeColors.BACKGROUND_DARK,
        fit: cPwaAssetGenerator.FIT_MODE
      },
      linkMediaOptions: {
        addMediaScreen: cPwaAssetGenerator.ADD_MEDIA_SCREEN,
        basePath: cPwaAssetGenerator.BASE_PATH,
        log: true,
        xhtml: cPwaAssetGenerator.XHTML
      },
      name: (pLandscape, pSize, pDark) => {
        const cOrientation = pLandscape
          ? SplashOrientation.cLandscape
          : SplashOrientation.cPortrait;
        const cDarkPrefix = pDark ? cPwaAssetGenerator.DARK_PREFIX : '';

        return `apple-splash-${cOrientation}-${cDarkPrefix}${pSize.width}x${pSize.height}.png`;
      },
      padding: cPwaAssetGenerator.SPLASH_PADDING,
      png: {
        compressionLevel: cPwaAssetGenerator.PNG_COMPRESSION_LEVEL,
        quality: cPwaAssetGenerator.PNG_QUALITY
      },
      resizeOptions: {
        background: cThemeColors.BACKGROUND_LIGHT,
        fit: cPwaAssetGenerator.FIT_MODE
      }
    },
    aPwaGeneratorDevices
  )
});
