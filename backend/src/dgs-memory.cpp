#include "dgs-memory.h"

dgs_memory_status fDgsMemoryStatusCombine(dgs_memory_status pS0, dgs_memory_status pS1) {
  bool vHasUpdate = false;

  switch (pS0) {
    case cDgsMemoryStatusSuccess:
      {
        vHasUpdate = true;
        break;
      }
    case cDgsMemoryStatusNoUpdate:
      {
        break;
      }
    case cDgsMemoryStatusFailedPrepare:
    case cDgsMemoryStatusFailedCompute:
      {
        return pS0;
      }
  }

  switch (pS1) {
    case cDgsMemoryStatusSuccess:
      {
        vHasUpdate = true;
        break;
      }
    case cDgsMemoryStatusNoUpdate:
      {
        break;
      }
    case cDgsMemoryStatusFailedPrepare:
    case cDgsMemoryStatusFailedCompute:
      {
        return pS1;
      }
  }

  // if either status has an update, then the combined status has an update
  return vHasUpdate ? cDgsMemoryStatusSuccess : cDgsMemoryStatusNoUpdate;
}

bool fDgsMemoryStatusIsFail(dgs_memory_status pStatus) {
  switch (pStatus) {
    case cDgsMemoryStatusSuccess:
    case cDgsMemoryStatusNoUpdate:
      {
        return false;
      }
    case cDgsMemoryStatusFailedPrepare:
    case cDgsMemoryStatusFailedCompute:
      {
        return true;
      }
  }

  return false;
}
