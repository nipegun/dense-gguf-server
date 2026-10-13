import { cBuildConfig } from '../src/lib/constants/pwa.constants';
import { existsSync, writeFileSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';

let vProcessed = false;

const cOutputDir = process.env.DGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

/**
 * Write build.json with the dgs.cpp release build number.
 *
 * DGS_BUILD_NUMBER is passed from CMake -> npm -> vite via env var.
 * Used for display of the current dgs-server release (e.g. "b1234").
 */
export function fBuildInfoPlugin(): Plugin {
  return {
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (vProcessed) return;

          vProcessed = true;

          const cBuildNumber = process.env.DGS_BUILD_NUMBER || 'b0000';
          const cOutDir = resolve(cOutputDir);
          const cIndexPath = resolve(cOutDir, 'index.html');

          if (!existsSync(cIndexPath)) return;

          const cBuildJsonPath = resolve(cOutDir, 'build.json');

          writeFileSync(cBuildJsonPath, JSON.stringify({ version: cBuildNumber }), 'utf-8');
          console.log(`Created build.json (version: ${cBuildNumber})`);
        } catch (error) {
          console.error('Failed to write build.json:', error);
        }
      }, 100);
    },
    name: 'dgscpp:build-info'
  };
}
