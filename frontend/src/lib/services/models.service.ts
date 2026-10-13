/**
 * ModelsService - Stateless model management API layer
 *
 * Wraps the /models endpoints (list, load, unload) and the /models/sse
 * status feed in MODEL and ROUTER modes. No reactive state; consumed by
 * modelsStore and its status manager.
 */

import { base } from '$app/paths';
import { cApiModels, MODEL_ID, type ModelSidecar } from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import type { ParsedModelId } from '$lib/types/models';
import {
  fApiDelete,
  fApiFetch,
  fApiPost,
  fExtractSseDataPayload,
  fNormalizeModelName,
  fSidecarFromFileToken,
  fSidecarFromTag,
  fSplitSseRecords
} from '$lib/utils';
import { fGetAuthHeaders } from '$lib/utils/api-headers';

export class ModelsService {
  private static readonly vSseReconnectMs = 1000;

  /**
   * Build the `<repo>:<tag>` string POST /models expects, so callers don't need
   * to know the tag conventions.
   */
  static mBuildDownloadTag(
    pRepoId: string,
    pQuant: string | null,
    pSidecar: ModelSidecar | null
  ): string {
    if (!pQuant && !pSidecar) return pRepoId;

    if (!pQuant) return `${pRepoId}:${pSidecar}`;

    const cTag = pSidecar ? `${pQuant}-${pSidecar}` : pQuant;

    return `${pRepoId}:${cTag}`;
  }

  /**
   * Cancel an in-flight download, or remove a downloaded/failed entry from the
   * model cache (ROUTER mode only): DELETE /models?model=<repo:tag>.
   */
  static async mCancelDownload(pHfRepoWithTag: string): Promise<ApiModelsDownloadResponse> {
    return fApiDelete<ApiModelsDownloadResponse>(cApiModels.DELETE, {
      model: pHfRepoWithTag
    });
  }

  /**
   * Start a model download from HuggingFace (ROUTER mode only). The response
   * returns immediately; progress arrives over /models/sse. The server picks
   * the file matching the tag and also pulls the model's mmproj/draft sidecars.
   */
  static async mDownloadModel(pHfRepoWithTag: string): Promise<ApiModelsDownloadResponse> {
    const cPayload: ApiModelsDownloadRequest = { model: pHfRepoWithTag };

    return fApiPost<ApiModelsDownloadResponse>(cApiModels.DOWNLOAD, cPayload);
  }

  /**
   * Check if a model is loaded based on its metadata.
   *
   * @param model - Model data entry from the API response
   * @returns True if the model status is LOADED
   */
  static mIsModelLoaded(pModel: ApiModelDataEntry): boolean {
    return pModel.status.value === ServerModelStatus.cLoaded;
  }

  static mIsModelLoading(pModel: ApiModelDataEntry): boolean {
    return pModel.status.value === ServerModelStatus.LOADING;
  }

  /**
   *
   *
   * Load/Unload
   *
   *
   */

  /**
   * True when a router entry id marks a downloaded sidecar file, e.g.
   * `org/model:Q4_0-mtp` or `org/model:mmproj`, not a loadable model.
   */
  static mIsSidecarEntry(pModelId: string): boolean {
    const cIdx = pModelId.indexOf(MODEL_ID.QUANTIZATION_SEPARATOR);

    if (cIdx === MODEL_ID.NOT_FOUND) return false;

    return fSidecarFromTag(pModelId.slice(cIdx + 1)) !== null;
  }

  /**
   * Fetch list of models from OpenAI-compatible endpoint.
   * Works in both MODEL and ROUTER modes.
   *
   * @returns List of available models with basic metadata
   */
  static async mList(): Promise<ApiModelsListResponse> {
    return fApiFetch<ApiModelsListResponse>(cApiModels.LIST);
  }

  /**
   * Load a model (ROUTER mode only).
   * Sends POST request to `/models/load`. Note: the endpoint returns success
   * before loading completes — use polling to await actual load status.
   *
   * @param modelId - Model identifier to load
   * @param extraArgs - Optional additional arguments to pass to the model instance
   * @returns Load response from the server
   */
  static async load(pModelId: string, pExtraArgs?: string[]): Promise<ApiModelsLoadResponse> {
    const cPayload: { model: string; extra_args?: string[] } = { model: pModelId };

    if (pExtraArgs && pExtraArgs.length > 0) {
      cPayload.extra_args = pExtraArgs;
    }

    return fApiPost<ApiModelsLoadResponse>(cApiModels.LOAD, cPayload);
  }

  /**
   * Parse a model ID string into its structured components.
   *
   * Handles conventions like:
   *   `<org>/<ModelName>-<Parameters>(-<ActivatedParameters>)(-<Tags>)(-<Quantization>):<Quantization>`
   *   `<ModelName>.<Quantization>` (dot-separated quantization, e.g. `model.Q4_K_M`)
   *
   * @param modelId - Raw model identifier string
   * @returns Structured {@link ParsedModelId} with all detected fields
   */
  static mParseModelId(pModelId: string): ParsedModelId {
    const cResult: ParsedModelId = {
      activatedParams: null,
      modelName: null,
      orgName: null,
      params: null,
      quantization: null,
      raw: pModelId,
      sidecar: null,
      tags: []
    };

    // strip directory path and weight extension so a bare `-m /path/file.gguf`
    // parses like a clean repo id; the HF `org/model` form is preserved
    let vSource = fNormalizeModelName(pModelId).replace(MODEL_ID.WEIGHT_EXTENSION_REGEX, '');

    // 0. Detect sidecar prefix (mtp-, dflash-, mmproj-) before any other
    //    splitting so the inner id parses cleanly.
    const cPrefixMatch = vSource.match(MODEL_ID.SIDECAR_PREFIX_REGEX);

    if (cPrefixMatch) {
      cResult.sidecar = fSidecarFromFileToken(cPrefixMatch[1].toLowerCase());
      vSource = cPrefixMatch[2];

      // a sidecar filename's remainder may be just the quant token,
      // e.g. `mtp-Q4_0.gguf` or `mmproj-F16.gguf`
      if (MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(vSource)) {
        cResult.quantization = vSource.toUpperCase();
        vSource = '';
      }
    } else {
      // 0b. Detect `-<type>` suffix (`-mtp`, `-dflash`, `-dspark`, `-eagle3`).
      //     Only strip it when the segment preceding it looks like a real quant
      //     token, so a model literally named `MyModel-mtp` is not mistaken for a
      //     draft one.
      const cSuffixMatch = vSource.match(MODEL_ID.SIDECAR_SUFFIX_REGEX);

      if (cSuffixMatch) {
        const cCandidate = cSuffixMatch[1];
        const cHeadSeg = cCandidate.split(MODEL_ID.SEGMENT_SEPARATOR).pop();

        if (cHeadSeg && MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(cHeadSeg)) {
          cResult.sidecar = fSidecarFromFileToken(cSuffixMatch[2].toLowerCase());
          vSource = cCandidate;
        }
      }
    }

    // 1. Extract colon-separated quantization (e.g. `model:Q4_K_M`)
    const cColonIdx = vSource.indexOf(MODEL_ID.QUANTIZATION_SEPARATOR);

    let vModelPath: string;

    if (cColonIdx !== MODEL_ID.NOT_FOUND) {
      cResult.quantization = vSource.slice(cColonIdx + 1) || null;
      vModelPath = vSource.slice(0, cColonIdx);
    } else {
      vModelPath = vSource;
    }

    // 2. Extract org name (e.g. `org/model` -> org = "org")
    const cSlashIdx = vModelPath.indexOf(MODEL_ID.ORG_SEPARATOR);

    let vModelStr: string;

    if (cSlashIdx !== MODEL_ID.NOT_FOUND) {
      cResult.orgName = vModelPath.slice(0, cSlashIdx);
      vModelStr = vModelPath.slice(cSlashIdx + 1);
    } else {
      vModelStr = vModelPath;
    }

    // 3. Handle dot-separated quantization (e.g. `model-name.Q4_K_M`)
    const cDotIdx = vModelStr.lastIndexOf('.');

    if (cDotIdx !== MODEL_ID.NOT_FOUND && !cResult.quantization) {
      const cAfterDot = vModelStr.slice(cDotIdx + 1);

      if (MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(cAfterDot)) {
        cResult.quantization = cAfterDot;
        vModelStr = vModelStr.slice(0, cDotIdx);
      }
    }

    const aSegments = vModelStr.split(MODEL_ID.SEGMENT_SEPARATOR);

    // 4. Detect trailing quantization from dash-separated segments
    //    Handle UD-prefixed quantization (e.g. `UD-Q8_K_XL`) and
    //    standalone quantization (e.g. `Q4_K_M`, `BF16`, `F16`, `MXFP4`)
    if (!cResult.quantization && aSegments.length > 1) {
      const cLast = aSegments[aSegments.length - 1];
      const cSecondLast = aSegments.length > 2 ? aSegments[aSegments.length - 2] : null;

      if (MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(cLast)) {
        if (cSecondLast && MODEL_ID.CUSTOM_QUANTIZATION_PREFIX_REGEX.test(cSecondLast)) {
          cResult.quantization = `${cSecondLast}-${cLast}`;
          aSegments.splice(aSegments.length - 2, 2);
        } else {
          cResult.quantization = cLast;
          aSegments.pop();
        }
      }
    }

    // 5. Find params and activated params
    let vParamsIdx = MODEL_ID.NOT_FOUND;
    let vActivatedParamsIdx = MODEL_ID.NOT_FOUND;

    for (let vI = 0; vI < aSegments.length; vI++) {
      const cSeg = aSegments[vI];

      if (vParamsIdx === MODEL_ID.NOT_FOUND && MODEL_ID.PARAMS_REGEX.test(cSeg)) {
        vParamsIdx = vI;
        cResult.params = cSeg.toUpperCase();
      } else if (vParamsIdx !== MODEL_ID.NOT_FOUND && MODEL_ID.ACTIVATED_PARAMS_REGEX.test(cSeg)) {
        vActivatedParamsIdx = vI;
        cResult.activatedParams = cSeg.toUpperCase();
      }
    }

    // 6. Model name = segments before params; tags = remaining segments after params
    const cPivotIdx = vParamsIdx !== MODEL_ID.NOT_FOUND ? vParamsIdx : aSegments.length;
    const aModelSegments = aSegments.slice(0, cPivotIdx);

    // strip trailing container-format segments (e.g. GGUF) from the model name
    while (
      aModelSegments.length > 0 &&
      MODEL_ID.IGNORED_SEGMENTS.has(aModelSegments[aModelSegments.length - 1].toUpperCase())
    ) {
      aModelSegments.pop();
    }

    cResult.modelName = aModelSegments.join(MODEL_ID.SEGMENT_SEPARATOR) || null;

    if (vParamsIdx !== MODEL_ID.NOT_FOUND) {
      cResult.tags = aSegments.slice(vParamsIdx + 1).filter((p, pRelIdx) => {
        const cAbsIdx = vParamsIdx + 1 + pRelIdx;

        if (cAbsIdx === vActivatedParamsIdx) return false;

        return !MODEL_ID.IGNORED_SEGMENTS.has(aSegments[cAbsIdx].toUpperCase());
      });
    }

    return cResult;
  }

  /**
   * Unload a model (ROUTER mode only).
   * Sends POST request to `/models/unload`. Note: the endpoint returns success
   * before unloading completes — use polling to await actual unload status.
   *
   * @param modelId - Model identifier to unload
   * @returns Unload response from the server
   */
  static async mUnload(pModelId: string): Promise<ApiModelsUnloadResponse> {
    return fApiPost<ApiModelsUnloadResponse>(cApiModels.UNLOAD, { model: pModelId });
  }

  /**
   * Read the /models/sse feed and invoke onEvent for each parsed envelope.
   * Reconnects on network drops until the signal aborts. Splits the byte
   * stream into SSE records on the blank line boundary; the payload rides in
   * the data lines as a JSON envelope with its own model, event and data fields.
   */
  static async mWatchModelEvents(
    pSignal: AbortSignal,
    pOnEvent: (pEvent: ApiModelsSseEvent) => void
  ): Promise<void> {
    const cDecoder = new TextDecoder();

    while (!pSignal.aborted) {
      try {
        const cResponse = await fetch(`${base}${cApiModels.SSE}`, {
          headers: fGetAuthHeaders(),
          signal: pSignal
        });

        if (cResponse.ok && cResponse.body) {
          const cReader = cResponse.body.getReader();

          let vBuffer = '';

          while (!pSignal.aborted) {
            const { done: vDone, value: vValue } = await cReader.read();

            if (vDone) break;

            vBuffer += cDecoder.decode(vValue, { stream: true });

            const { records: aRecords, rest: vRest } = fSplitSseRecords(vBuffer);

            vBuffer = vRest;

            for (const cRecord of aRecords) {
              const cEvent = ModelsService.mParseStatusRecord(cRecord);

              if (cEvent) pOnEvent(cEvent);
            }
          }
        }
      } catch {
        // network drop or abort falls through to the reconnect delay
      }

      if (pSignal.aborted) return;

      await new Promise((pResolve) => setTimeout(pResolve, ModelsService.vSseReconnectMs));
    }
  }

  /**
   * Parse one SSE record into its JSON envelope, or null when the record
   * carries no data payload or malformed JSON.
   */
  private static mParseStatusRecord(pRecord: string): ApiModelsSseEvent | null {
    const cPayload = fExtractSseDataPayload(pRecord);

    if (cPayload.length === 0) return null;

    try {
      return JSON.parse(cPayload) as ApiModelsSseEvent;
    } catch {
      return null;
    }
  }
}
