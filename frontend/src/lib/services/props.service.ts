/**
 * PropsService - Fetches server properties from /props
 *
 * Returns global server settings and capabilities, including per-model
 * modalities in MODEL mode. No reactive state; consumed by serverStore and
 * the model props manager.
 */

import { fApiFetchWithParams } from '$lib/utils';

export class PropsService {
  /**
   * Fetches global server properties from the `/props` endpoint.
   * In MODEL mode, returns modalities for the single loaded model.
   * In ROUTER mode, returns server-wide settings without model-specific modalities.
   *
   * @param autoload - If false, prevents automatic model loading (default: false)
   * @returns Server properties including default generation settings and capabilities
   * @throws {Error} If the request fails or returns invalid data
   */
  static async mFetch(pAutoload = false): Promise<ApiDgsServerProps> {
    const dParams: Record<string, string> = {};

    if (!pAutoload) {
      dParams.autoload = 'false';
    }

    return fApiFetchWithParams<ApiDgsServerProps>('./props', dParams, { authOnly: true });
  }

  /**
   * Fetches server properties for a specific model (ROUTER mode only).
   * Required in ROUTER mode because global `/props` does not include per-model modalities.
   *
   * @param modelId - The model ID to fetch properties for
   * @param autoload - If false, prevents automatic model loading (default: false)
   * @returns Server properties specific to the requested model
   * @throws {Error} If the request fails, model not found, or model not loaded
   */
  static async mFetchForModel(
    pModelId: string,
    pAutoload = false
  ): Promise<ApiDgsServerProps> {
    const dParams: Record<string, string> = { model: pModelId };

    if (!pAutoload) {
      dParams.autoload = 'false';
    }

    return fApiFetchWithParams<ApiDgsServerProps>('./props', dParams, { authOnly: true });
  }
}
