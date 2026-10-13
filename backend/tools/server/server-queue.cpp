#include "server-queue.h"

#include "log.h"
#include "server-task.h"

#include <algorithm>
#include <chrono>
#include <thread>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

static bool fTaskResetsIdleTimer(server_task_type type) {
  return type != cServerTaskTypeMetrics;
}

int server_queue::mPost(server_task && pTask, bool front) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  GGML_ASSERT(pTask.vId != -1);
  // if this is cancel task make sure to clean up pending tasks
  if (pTask.type == cServerTaskTypeCancel) {
    mCleanupPendingTask(pTask.vIdTarget);
  }
  const int  cTaskId     = pTask.vId;
  const bool cResetTimer = fTaskResetsIdleTimer(pTask.type);
  QUE_DBG("new task, id = %d, front = %d\n", cTaskId, front);
  if (front) {
    lQueueTasks.push_front(std::move(pTask));
  } else {
    lQueueTasks.push_back(std::move(pTask));
  }
  if (cResetTimer) {
    vTimeLastTask = ggml_time_ms();
  }
  vConditionTasks.notify_one();
  return cTaskId;
}

int server_queue::mPost(std::vector<server_task> && pTasks, bool front) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  bool                         vResetTimer = false;
  for (auto & vTask : pTasks) {
    if (vTask.vId == -1) {
      vTask.vId = vId++;
    }
    // if this is cancel task make sure to clean up pending tasks
    if (vTask.type == cServerTaskTypeCancel) {
      mCleanupPendingTask(vTask.vIdTarget);
    }
    vResetTimer |= fTaskResetsIdleTimer(vTask.type);
    QUE_DBG("new task, id = %d/%d, front = %d\n", vTask.vId, (int) pTasks.size(), front);
    if (front) {
      lQueueTasks.push_front(std::move(vTask));
    } else {
      lQueueTasks.push_back(std::move(vTask));
    }
  }
  if (vResetTimer) {
    vTimeLastTask = ggml_time_ms();
  }
  vConditionTasks.notify_one();
  return 0;
}

void server_queue::mDefer(server_task && pTask) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  QUE_DBG("defer task, id = %d\n", pTask.vId);
  lQueueTasksDeferred.push_back(std::move(pTask));
  vTimeLastTask = ggml_time_ms();
  vConditionTasks.notify_one();
}

int server_queue::mGetNewId() {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  int                          vNewId = vId++;
  return vNewId;
}

void server_queue::mPopDeferredTask(int pIdSlot) {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  if (!lQueueTasksDeferred.empty()) {
    // try to find a task that uses the specified slot
    bool vFound = false;
    for (auto vIt = lQueueTasksDeferred.begin(); vIt != lQueueTasksDeferred.end(); ++vIt) {
      if (vIt->vIdSlot == pIdSlot) {
        QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", pIdSlot, vIt->vId);
        lQueueTasks.emplace_front(std::move(*vIt));
        lQueueTasksDeferred.erase(vIt);
        vFound = true;
        break;
      }
    }
    // if not tasks found using the slot, just pop the first deferred task (default behavior)
    if (!vFound) {
      QUE_DBG("pop deferred task, id_task = %d\n", lQueueTasksDeferred.front().vId);
      lQueueTasks.emplace_front(std::move(lQueueTasksDeferred.front()));
      lQueueTasksDeferred.pop_front();
    }
  }
  vTimeLastTask = ggml_time_ms();
  vConditionTasks.notify_one();
}

void server_queue::mWaitUntilNoSleep() {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  if (!vSleeping) {
    return;
  } else {
    if (!vReqStopSleeping) {
      QUE_DBG("%s", "requesting to stop sleeping\n");
      vReqStopSleeping = true;
      vConditionTasks.notify_one();  // only main thread is waiting on this
    }
    QUE_DBG("%s", "waiting until no sleep\n");
    vConditionTasks.wait(vLock, [&] { return !vSleeping; });
  }
}

void server_queue::mTerminate() {
  std::unique_lock<std::mutex> vLock(vMutexTasks);
  vRunning = false;
  vConditionTasks.notify_all();
}

bool server_queue::mProcessNewTasks(bool pIsYielding) {
  while (true) {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    if (!vRunning) {
      QUE_DBG("%s", "terminate\n");
      return true;
    }
    if (lQueueTasks.empty()) {
      return false;
    }
    server_task vTask = std::move(lQueueTasks.front());
    lQueueTasks.pop_front();
    vLock.unlock();

    QUE_DBG("processing task, id = %d\n", vTask.vId);
    if (!vCallbackNewTask(std::move(vTask), pIsYielding)) {
      // set it aside, do not put it back in the queue, else we offer it again in a loop
      GGML_ASSERT(pIsYielding && "a task can only be declined while yielding");
      QUE_DBG("task declined, id = %d\n", vTask.vId);
      vLock.lock();
      lQueueTasksUnhandled.push_back(std::move(vTask));
    }
  }
}

void server_queue::mWorkerLoop() {
  while (true) {
    {
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      // wait on busy instead of yielding - busy stays set even when the yield already ended
      vWorker.vCv.wait(vLock, [&] { return vWorker.vStop || vWorker.vBusy; });
      if (vWorker.vStop) {
        return;
      }
    }

    // process tasks while the yield is active
    while (true) {
      bool vTerminated = false;
      try {
        // note: do not hold any lock here, the callback may post new tasks
        vTerminated = mProcessNewTasks(true);
      } catch (...) {
        std::unique_lock<std::mutex> vLock(vMutexTasks);
        vWorker.vException = std::current_exception();
        break;
      }

      std::unique_lock<std::mutex> vLock(vMutexTasks);
      if (vTerminated || vWorker.vStop || !vWorker.vYielding) {
        break;
      }
      if (!lQueueTasks.empty()) {
        continue;  // a new task arrived in the meantime
      }
      vConditionTasks.wait(vLock,
                           [&] { return vWorker.vStop || !vRunning || !vWorker.vYielding || !lQueueTasks.empty(); });
    }

    // signal to yield_to_queue() that no more tasks will be processed
    {
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      vWorker.vBusy = false;
    }
    vConditionTasks.notify_all();
  }
}

void server_queue::mWorkerStop() {
  if (!vWorker.vThread.joinable()) {
    return;
  }
  {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    vWorker.vStop = true;
  }
  vWorker.vCv.notify_one();
  vConditionTasks.notify_all();
  vWorker.vThread.join();
}

void server_queue::mYieldToQueue(std::function<void()> && pWork) {
  GGML_ASSERT(vWorker.vThread.joinable() && "yield_to_queue() requires start_loop() to be running");

  QUE_DBG("%s", "yielding to queue\n");

  {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    GGML_ASSERT(!vWorker.vBusy && "yield_to_queue() cannot be nested");
    vWorker.vBusy     = true;
    vWorker.vYielding = true;
  }
  vWorker.vCv.notify_one();

  // run the work on the current thread, so that all ggml compute stays on the same thread
  std::exception_ptr vException;
  try {
    pWork();
  } catch (...) {
    vException = std::current_exception();
  }

  {
    std::unique_lock<std::mutex> vLock(vMutexTasks);

    // the yield is over, wait for the worker to finish its current task
    vWorker.vYielding = false;
    vConditionTasks.notify_all();
    vConditionTasks.wait(vLock, [&] { return !vWorker.vBusy; });

    // put the declined tasks back, keeping their order
    while (!lQueueTasksUnhandled.empty()) {
      lQueueTasks.push_front(std::move(lQueueTasksUnhandled.back()));
      lQueueTasksUnhandled.pop_back();
    }

    // make sure to avoid idle timeout here
    vTimeLastTask = ggml_time_ms();

    // an exception from work() takes precedence over the one from the worker
    if (!vException) {
      std::swap(vException, vWorker.vException);
    } else {
      vWorker.vException = nullptr;
    }
  }

  QUE_DBG("%s", "done yielding to queue\n");

  // note: rethrow only after the declined tasks are back in the queue, so they are not lost
  if (vException) {
    std::rethrow_exception(vException);
  }
}

void server_queue::mStartLoop(int64_t pIdleSleepMs) {
  vRunning      = true;
  vTimeLastTask = ggml_time_ms();

  // spawn the worker thread used by yield_to_queue()
  GGML_ASSERT(!vWorker.vThread.joinable() && "start_loop() is already running");
  vWorker.vStop     = false;
  vWorker.vBusy     = false;
  vWorker.vYielding = false;
  vWorker.vThread   = std::thread([this]() { mWorkerLoop(); });

  constexpr auto cMaxWaitTime = std::chrono::seconds(1);
  auto           vShouldSleep = [&]() -> bool {
    // caller must hold mutex_tasks
    if (pIdleSleepMs < 0) {
      return false;
    }
    int64_t vNow = ggml_time_ms();
    return (vNow - vTimeLastTask) >= pIdleSleepMs;
  };

  while (true) {
    QUE_DBG("%s", "processing new tasks\n");
    if (mProcessNewTasks(false)) {
      break;  // terminate
    }

    // all tasks in the current loop is processed, slots data is now ready
    QUE_DBG("%s", "update slots\n");

    // this will run the main inference process for all slots
    const int64_t cTUpdateSlots = ggml_time_ms();
    vCallbackUpdateSlots();
    {
      // update_slots() may take a while to finish, we need to make sure it's not counted as idle
      // shift instead of reset, so that non-task_resets_idle_timer tasks do not delay the sleep
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      const int64_t                cNow = ggml_time_ms();
      vTimeLastTask                     = std::min(cNow, vTimeLastTask + (cNow - cTUpdateSlots));
    }

    QUE_DBG("%s", "waiting for new tasks\n");
    while (true) {
      std::unique_lock<std::mutex> vLock(vMutexTasks);
      if (!vRunning || !lQueueTasks.empty()) {
        break;  // go back to process new tasks or terminate
      }

      // no tasks, check for sleeping state
      if (vShouldSleep()) {
        QUE_INF("%s", "entering sleeping state\n");
        vSleeping = true;
        // Call order cb0 -> cb1 -> cb{N}
        for (auto & vCb : lCallbackSleepingState) {
          vCb(true);
        }
        vReqStopSleeping = false;
        // wait until we are requested to exit sleeping state
        vConditionTasks.wait(vLock, [&] { return (!vRunning || vReqStopSleeping); });
        if (!vRunning) {  // may changed during sleep
          break;          // terminate
        }
        QUE_INF("%s", "exiting sleeping state\n");
        vReqStopSleeping = false;
        // Call order cb{N} -> cb1 -> cb0
        for (size_t vI = lCallbackSleepingState.size(); vI > 0; vI--) {
          lCallbackSleepingState[vI - 1](false);
        }
        vSleeping     = false;
        vTimeLastTask = ggml_time_ms();
        vConditionTasks.notify_all();  // notify wait_until_no_sleep()
        break;                         // process new tasks
      } else {
        // wait for new tasks or timeout for checking sleeping condition
        bool vRes = vConditionTasks.wait_for(vLock, cMaxWaitTime, [&] { return (!lQueueTasks.empty() || !vRunning); });
        if (vRes) {
          break;  // new task arrived or terminate
        }
        // otherwise, loop again to check sleeping condition
      }
    }
  }

  mWorkerStop();
}

void server_queue::mCleanupPendingTask(int pIdTarget) {
  // no need lock because this is called exclusively by post()
  auto vRmFunc = [pIdTarget](const server_task & pTask) {
    return pTask.vId == pIdTarget;
  };
  lQueueTasks.erase(std::remove_if(lQueueTasks.begin(), lQueueTasks.end(), vRmFunc), lQueueTasks.end());
  lQueueTasksDeferred.erase(std::remove_if(lQueueTasksDeferred.begin(), lQueueTasksDeferred.end(), vRmFunc),
                            lQueueTasksDeferred.end());
  // a task declined while yielding is not in queue_tasks yet, but it can still be cancelled
  lQueueTasksUnhandled.erase(std::remove_if(lQueueTasksUnhandled.begin(), lQueueTasksUnhandled.end(), vRmFunc),
                             lQueueTasksUnhandled.end());
}

//
// server_response
//

void server_response::mAddWaitingTaskId(int pIdTask) {
  RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", pIdTask, (int) sWaitingTaskIds.size());

  std::unique_lock<std::mutex> vLock(vMutexResults);
  sWaitingTaskIds.insert(pIdTask);
}

void server_response::mAddWaitingTaskIds(const std::unordered_set<int> & pIdTasks) {
  std::unique_lock<std::mutex> vLock(vMutexResults);

  for (const auto & vIdTask : pIdTasks) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", vIdTask, (int) sWaitingTaskIds.size());
    sWaitingTaskIds.insert(vIdTask);
  }
}

void server_response::mRemoveWaitingTaskId(int pIdTask) {
  RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", pIdTask,
          (int) sWaitingTaskIds.size());

  std::unique_lock<std::mutex> vLock(vMutexResults);
  sWaitingTaskIds.erase(pIdTask);
  // make sure to clean up all pending results
  lQueueResults.erase(std::remove_if(lQueueResults.begin(), lQueueResults.end(),
                                     [pIdTask](const server_task_result_ptr & pRes) { return pRes->vId == pIdTask; }),
                      lQueueResults.end());
}

void server_response::mRemoveWaitingTaskIds(const std::unordered_set<int> & pIdTasks) {
  std::unique_lock<std::mutex> vLock(vMutexResults);

  for (const auto & vIdTask : pIdTasks) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", vIdTask,
            (int) sWaitingTaskIds.size());
    sWaitingTaskIds.erase(vIdTask);
  }
}

server_task_result_ptr server_response::mRecv(const std::unordered_set<int> & pIdTasks) {
  while (true) {
    std::unique_lock<std::mutex> vLock(vMutexResults);
    vConditionResults.wait(vLock, [&] {
      if (!vRunning) {
        RES_DBG("%s : queue result stop\n", "recv");
        std::terminate();  // we cannot return here since the caller is HTTP code
      }
      return !lQueueResults.empty();
    });

    for (size_t vI = 0; vI < lQueueResults.size(); vI++) {
      if (pIdTasks.find(lQueueResults[vI]->vId) != pIdTasks.end()) {
        server_task_result_ptr vRes = std::move(lQueueResults[vI]);
        lQueueResults.erase(lQueueResults.begin() + vI);
        return vRes;
      }
    }
  }

  // should never reach here
}

server_task_result_ptr server_response::mRecvWithTimeout(const std::unordered_set<int> & pIdTasks, int pTimeout) {
  while (true) {
    std::unique_lock<std::mutex> vLock(vMutexResults);

    for (int vI = 0; vI < (int) lQueueResults.size(); vI++) {
      if (pIdTasks.find(lQueueResults[vI]->vId) != pIdTasks.end()) {
        server_task_result_ptr vRes = std::move(lQueueResults[vI]);
        lQueueResults.erase(lQueueResults.begin() + vI);
        return vRes;
      }
    }

    std::cv_status vCrRes = vConditionResults.wait_for(vLock, std::chrono::seconds(pTimeout));
    if (!vRunning) {
      RES_DBG("%s : queue result stop\n", __func__);
      std::terminate();  // we cannot return here since the caller is HTTP code
    }
    if (vCrRes == std::cv_status::timeout) {
      return nullptr;
    }
  }

  // should never reach here
}

server_task_result_ptr server_response::mRecv(int pIdTask) {
  std::unordered_set<int> sIdTasks = { pIdTask };
  return mRecv(sIdTasks);
}

void server_response::mSend(server_task_result_ptr && pResult) {
  RES_DBG("sending result for task id = %d\n", pResult->vId);

  std::unique_lock<std::mutex> vLock(vMutexResults);
  for (const auto & vIdTask : sWaitingTaskIds) {
    if (pResult->vId == vIdTask) {
      RES_DBG("task id = %d pushed to result queue\n", pResult->vId);

      lQueueResults.emplace_back(std::move(pResult));
      vConditionResults.notify_all();
      return;
    }
  }
}

void server_response::mBroadcast(server_task_result_ptr && pResult) {
  std::unique_lock<std::mutex> vLock(vMutexResults);
  for (const auto & vIdTask : sWaitingTaskIds) {
    RES_DBG("task id = %d pushed to result queue\n", vIdTask);
    server_task_result_ptr vResCopy(pResult->clone());
    vResCopy->vId = vIdTask;  // override id with target task id
    lQueueResults.emplace_back(std::move(vResCopy));
  }
  vConditionResults.notify_all();
}

void server_response::mTerminate() {
  vRunning = false;
  vConditionResults.notify_all();
}

//
// server_response_reader
//

void server_response_reader::mPostTask(server_task && pTask, bool front) {
  GGML_ASSERT(sIdTasks.empty() && "post_task() can only be called once per reader");
  GGML_ASSERT(!pTask.mIsParent() && "not supported, use post_tasks() instead");
  pTask.vIndex = 0;
  sIdTasks.insert(pTask.vId);
  lStates.push_back(task_result_state());
  vQueueResults.mAddWaitingTaskId(pTask.vId);
  vQueueTasks.mPost(std::move(pTask), front);
}

void server_response_reader::mPostTasks(std::vector<server_task> && pTasks,
                                        const common_chat_session & pSession,
                                        bool                        front) {
  GGML_ASSERT(sIdTasks.empty() && "post_tasks() can only be called once per reader");
  sIdTasks = server_task::mGetListId(pTasks);
  lStates.reserve(pTasks.size());
  size_t vIndex = 0;
  for (auto & vTask : pTasks) {
    vTask.vIndex = vIndex++;
    lStates.push_back(task_result_state(pSession));
    // for child tasks
    for (auto & vChildTask : vTask.lChildTasks) {
      vChildTask.vIndex = vIndex++;
      lStates.push_back(task_result_state(pSession));
    }
  }
  GGML_ASSERT(lStates.size() == sIdTasks.size());
  vQueueResults.mAddWaitingTaskIds(sIdTasks);
  vQueueTasks.mPost(std::move(pTasks), front);
}

bool server_response_reader::mHasNext() const {
  return !vCancelled && vReceivedCount < sIdTasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::mNext(const std::function<bool()> & pShouldStop) {
  while (true) {
    server_task_result_ptr vResult = vQueueResults.mRecvWithTimeout(sIdTasks, vPollingIntervalSeconds);
    if (vResult == nullptr) {
      // timeout, check stop condition
      if (pShouldStop()) {
        return nullptr;
      }
    } else {
      if (vResult->mIsError()) {
        mStop();  // cancel remaining tasks
        SRV_DBG("%s", "received error result, stopping further processing\n");
        return vResult;
      }
      if (!lStates.empty()) {
        // update the generation state if needed
        const size_t cIdx = vResult->vIndex;
        GGML_ASSERT(cIdx < lStates.size());
        vResult->mUpdate(lStates[cIdx]);
      }
      if (vResult->mIsStop()) {
        vReceivedCount++;
      }
      return vResult;
    }
  }

  // should not reach here
}

server_response_reader::batch_response server_response_reader::mWaitForAll(const std::function<bool()> & pShouldStop) {
  batch_response vBatchRes;
  vBatchRes.lResults.clear();
  vBatchRes.lResults.resize(sIdTasks.size());
  while (mHasNext()) {
    auto vRes = mNext(pShouldStop);
    if (vRes == nullptr) {
      vBatchRes.vIsTerminated = true;
      return vBatchRes;
    }
    if (vRes->mIsError()) {
      vBatchRes.vError = std::move(vRes);
      return vBatchRes;
    }
    const size_t cIdx = vRes->vIndex;
    GGML_ASSERT(cIdx < vBatchRes.lResults.size() && "index out of range");
    GGML_ASSERT(vBatchRes.lResults[cIdx] == nullptr && "duplicate result received");
    vBatchRes.lResults[cIdx] = std::move(vRes);
  }
  return vBatchRes;
}

void server_response_reader::mStop() {
  vQueueResults.mRemoveWaitingTaskIds(sIdTasks);
  if (mHasNext() && !vCancelled) {
    // if tasks is not finished yet, cancel them
    vCancelled = true;
    std::vector<server_task> lCancelTasks;
    lCancelTasks.reserve(sIdTasks.size());
    for (const auto & vIdTask : sIdTasks) {
      SRV_WRN("cancel task, id_task = %d\n", vIdTask);
      server_task vTask(cServerTaskTypeCancel);
      vTask.vIdTarget = vIdTask;
      vQueueResults.mRemoveWaitingTaskId(vIdTask);
      lCancelTasks.push_back(std::move(vTask));
    }
    // push to beginning of the queue, so it has highest priority
    vQueueTasks.mPost(std::move(lCancelTasks), true);
  } else {
    SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
  }
}
