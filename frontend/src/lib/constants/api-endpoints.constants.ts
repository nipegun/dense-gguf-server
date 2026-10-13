import { fApiUrl } from '$lib/api-url';
export const cApiModels = {
  /** Download a model from HuggingFace (ROUTER mode, POST) or cancel/remove it (DELETE) */
  DELETE: fApiUrl('/models'),
  DOWNLOAD: fApiUrl('/models'),
  LIST: fApiUrl('/v1/models'),
  LOAD: fApiUrl('/models/load'),
  SSE: fApiUrl('/models/sse'),
  UNLOAD: fApiUrl('/models/unload')
};

// chat completion routes, the control route drives realtime inference (e.g. end reasoning)
export const cApiChat = {
  COMPLETIONS: fApiUrl('./v1/chat/completions'),
  CONTROL: fApiUrl('./v1/chat/completions/control')
};

// slot introspection, requires the --slots flag on the server
export const cApiSlots = {
  LIST: fApiUrl('./slots')
};

export const cApiTools = {
  EXECUTE: fApiUrl('/tools'),
  LIST: fApiUrl('/tools')
};

// resumable stream routes, the conv::model identity travels as the conv_id query param
// because model names can contain slashes that a path segment cannot carry
// resume retry cadence while the owning model is still loading (server answers 503)
export const cStreamResumeRetryMs = 2000;

export const cApiStream = {
  BASE: fApiUrl('./v1/stream'),
  LOOKUP: fApiUrl('./v1/streams/lookup')
};

// query params for the resumable stream routes
export const cStreamQueryParams = {
  CONV_ID: 'conv_id',
  FROM: 'from'
} as const;

/** CORS proxy endpoint path */
export const cCorsProxyEndpoint = fApiUrl('/cors-proxy');
