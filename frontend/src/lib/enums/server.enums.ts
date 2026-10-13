/**
 * Server role enum - used for single/multi-model mode
 */
export enum ServerRole {
  /** Single model mode - server running with a specific model loaded */
  MODEL = 'model',
  /** Router mode - server managing multiple model instances */
  ROUTER = 'router'
}

/**
 * Model status enum - matches tools/server/server-models.h from C++ server
 * Used as the `value` field in the status object from /models endpoint
 */
export enum ServerModelStatus {
  cDownloadFailed = 'download_failed',
  cDownloadFinished = 'download_finished',
  cDownloaded = 'downloaded',
  cDownloading = 'downloading',
  FAILED = 'failed',
  cLoaded = 'loaded',
  LOADING = 'loading',
  cSleeping = 'sleeping',
  cUnloaded = 'unloaded'
}

/**
 * /models/sse event type enum - discriminates the records broadcast on the
 * model status feed in ROUTER mode. Matches the event names emitted by
 * tools/server/server-models.cpp from the C++ server.
 */
export enum ServerModelsSseEventType {
  cDownloadFailed = 'download_failed',
  cDownloadFinished = 'download_finished',
  cDownloadProgress = 'download_progress',
  cModelRemove = 'model_remove',
  cModelStatus = 'model_status',
  cModelsReload = 'models_reload',
  cStatusChange = 'status_change',
  cStatusUpdate = 'status_update'
}
