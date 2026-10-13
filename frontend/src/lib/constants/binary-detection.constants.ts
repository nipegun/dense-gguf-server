import type { BinaryDetectionOptions } from '$lib/types';

export const cDefaultBinaryDetectionOptions: BinaryDetectionOptions = {
  maxAbsoluteNullBytes: 2,
  prefixLength: 1024 * 10, // Check the first 10KB of the string
  suspiciousCharThresholdRatio: 0.15 // Allow up to 15% suspicious chars
};
