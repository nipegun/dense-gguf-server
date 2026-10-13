export const cErrorMessages = {
  HTTP: {
    ACCESS_DENIED: 'Access denied',
    GENERIC: 'Request failed',
    INTERNAL_ERROR: 'Server error - check server logs',
    NOT_FOUND: 'Not found',
    TEMPORARILY_UNAVAILABLE: 'Server temporarily unavailable'
  },
  NETWORK: {
    GENERIC: 'Failed to connect to server',
    NXDOMAIN: 'Server not found - check server address',
    REFUSED: 'Connection refused - server may be offline',
    TIMEOUT: 'Request timed out',
    UNREACHABLE: 'Server is not running or unreachable'
  }
};

export const dHttpCodeToString: Record<string, string> = {
  401: cErrorMessages.HTTP.ACCESS_DENIED,
  403: cErrorMessages.HTTP.ACCESS_DENIED,
  500: cErrorMessages.HTTP.INTERNAL_ERROR,
  503: cErrorMessages.HTTP.TEMPORARILY_UNAVAILABLE
};
