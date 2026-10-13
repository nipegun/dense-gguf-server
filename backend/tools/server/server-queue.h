#pragma once

#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue_waiting {
  int    vId;
  size_t vIndex;
  bool   vFromWebui;
  bool   vQueueNotice;
};

struct server_queue {
private:
  int     vId              = 0;
  bool    vRunning         = false;
  bool    vSleeping        = false;
  bool    vReqStopSleeping = false;
  int64_t vTimeLastTask    = 0;

  // queues
  std::deque<server_task> lQueueTasks;
  std::deque<server_task> lQueueTasksDeferred;
  // tasks declined while yielding, put back in queue_tasks once the yield is done
  // note: kept as a member so that cleanup_pending_task() can also reach them
  std::deque<server_task> lQueueTasksUnhandled;

  std::mutex              vMutexTasks;
  std::condition_variable vConditionTasks;

  // used by yield_to_queue, all fields are guarded by mutex_tasks
  struct worker_t {
    std::thread             vThread;
    std::condition_variable vCv;         // the worker sleeps on this until a yield starts
    std::exception_ptr      vException;  // exception thrown while processing tasks, if any
    bool                    vStop = false;
    bool vBusy     = false;  // set by yield_to_queue(), cleared by the worker once it is done processing tasks
    bool vYielding = false;  // work() is still running on the start_loop() thread
  };

  worker_t vWorker;

  // callback functions
  std::function<bool(server_task &&, bool)> vCallbackNewTask;
  std::function<void(void)>                 vCallbackUpdateSlots;
  std::vector<std::function<void(bool)>>    lCallbackSleepingState;

public:
  ~server_queue() { mWorkerStop(); }

  // Add a new task to the end of the queue
  int mPost(server_task && pTask, bool front = false);

  // multi-task version of post()
  int mPost(std::vector<server_task> && pTasks, bool front = false);

  // Add a new task, but defer until one slot is available
  void mDefer(server_task && pTask);

  // Get the next id for creating a new task
  int mGetNewId();

  // Call when the state of one slot is changed, it will move one task from deferred to main queue
  // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
  void mPopDeferredTask(int pIdSlot);

  // if sleeping, request exiting sleep state and wait until it is done
  // returns immediately if not sleeping
  void mWaitUntilNoSleep();

  bool mIsSleeping() {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    return vSleeping;
  }

  // end the start_loop routine
  void mTerminate();

  /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     *   note: metrics tasks are processed as usual, but do not reset the idle timer
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
  void mStartLoop(int64_t pIdleSleepMs = -1);

  // while waiting for work() to finish, run process_new_tasks on the worker thread
  // returns once work() is done (may throw exceptions)
  // must be called from start_loop() thread (ideally inside callback_update_slots)
  // use case: return metrics while encode/decode is running
  //
  // tasks declined by callback_new_task are put back in the queue once this returns
  void mYieldToQueue(std::function<void()> && pWork);

  std::vector<server_queue_waiting> mDeferredTasks() {
    std::unique_lock<std::mutex>      vLock(vMutexTasks);
    std::vector<server_queue_waiting> lWaiting;
    for (const auto & vTask : lQueueTasksDeferred) {
      lWaiting.push_back({ vTask.vId, vTask.vIndex, vTask.vParams.vFromWebui, vTask.vParams.vQueueNotice });
    }
    return lWaiting;
  }

  // for metrics
  size_t mQueueTasksDeferredSize() {
    std::unique_lock<std::mutex> vLock(vMutexTasks);
    return lQueueTasksDeferred.size();
  }

  //
  // Functions below are not thread-safe, must only be used before start_loop() is called
  //

  // Register function to process a new task
  // the second argument tells whether the queue is currently yielding (see yield_to_queue)
  // only then may the callback return false to decline the task, and it must leave it
  // untouched, so that it can be put back in the queue later
  // note: while yielding, the callback runs on worker thread, not main thread
  void mOnNewTask(std::function<bool(server_task &&, bool)> pCallback) { vCallbackNewTask = std::move(pCallback); }

  // Register the function to be called when all slots data is ready to be processed
  void mOnUpdateSlots(std::function<void(void)> pCallback) { vCallbackUpdateSlots = std::move(pCallback); }

  // Register callback for sleeping state change; multiple callbacks are allowed
  // for example: register order cb0, cb1, cb2
  // entering sleep: queue.sleeping = true --> cb0(true) --> cb1(true) --> cb2(true)
  // leaving sleep: cb2(false) --> cb1(false) --> cb0(false) --> queue.sleeping = false
  // note: caller will hold mutex_tasks while calling the callbacks
  void mOnSleepingState(std::function<void(bool)> pCallback) { lCallbackSleepingState.push_back(std::move(pCallback)); }

private:
  void mCleanupPendingTask(int pIdTarget);

  // process all pending tasks in the queue
  // returns true if the queue is terminated, false if there is no more task to process
  // while yielding, declined tasks are moved to queue_tasks_unhandled
  bool mProcessNewTasks(bool pIsYielding);

  // for worker_t
  void mWorkerLoop();
  void mWorkerStop();
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
  bool vRunning = true;

  // for keeping track of all tasks waiting for the result
  std::unordered_set<int> sWaitingTaskIds;

  // the main result queue (using ptr for polymorphism)
  std::vector<server_task_result_ptr> lQueueResults;

  std::mutex              vMutexResults;
  std::condition_variable vConditionResults;

public:
  // add the id_task to the list of tasks waiting for response
  void mAddWaitingTaskId(int pIdTask);

  void mAddWaitingTaskIds(const std::unordered_set<int> & pIdTasks);

  // when the request is finished, we can remove task associated with it
  void mRemoveWaitingTaskId(int pIdTask);

  // remove multiple tasks from waiting list
  void mRemoveWaitingTaskIds(const std::unordered_set<int> & pIdTasks);

  // This function blocks the thread until there is a response for one of the id_tasks
  server_task_result_ptr mRecv(const std::unordered_set<int> & pIdTasks);

  // same as recv(), but have timeout in seconds
  // if timeout is reached, nullptr is returned
  server_task_result_ptr mRecvWithTimeout(const std::unordered_set<int> & pIdTasks, int pTimeout);

  // single-task version of recv()
  server_task_result_ptr mRecv(int pIdTask);

  // Send a new result to a waiting id_task
  void mSend(server_task_result_ptr && pResult);

  // broadcast a new result to all waiting tasks
  // (used by router mode)
  void mBroadcast(server_task_result_ptr && pResult);

  // terminate the waiting loop
  void mTerminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
  std::unordered_set<int> sIdTasks;
  server_queue &          vQueueTasks;
  server_response &       vQueueResults;
  size_t                  vReceivedCount = 0;
  bool                    vCancelled     = false;
  int                     vPollingIntervalSeconds;

  // tracking generation state and partial tool calls
  // only used by streaming completions
  std::vector<task_result_state> lStates;

  // should_stop function will be called each polling_interval_seconds
  server_response_reader(server_queue & pQueueTasks, server_response & pQueueResults, int pPollingIntervalSeconds) :
    vQueueTasks(pQueueTasks),
    vQueueResults(pQueueResults),
    vPollingIntervalSeconds(pPollingIntervalSeconds) {}

  ~server_response_reader() { mStop(); }

  int mGetNewId() { return vQueueTasks.mGetNewId(); }

  // if front = true, the task will be posted to the front of the queue (high priority)
  void mPostTask(server_task && pTask, bool front = false);
  void mPostTasks(std::vector<server_task> && pTasks, const common_chat_session & pSession = {}, bool front = false);
  bool mHasNext() const;

  // return nullptr if should_stop() is true before receiving a result
  // note: if one error is received, it will stop further processing and return error result
  server_task_result_ptr mNext(const std::function<bool()> & pShouldStop);

  struct batch_response {
    bool vIsTerminated = false;  // if true, indicates that processing was stopped before all results were received
    std::vector<server_task_result_ptr> lResults;
    server_task_result_ptr              vError;  // nullptr if no error
  };

  // aggregate multiple results
  batch_response mWaitForAll(const std::function<bool()> & pShouldStop);

  void mStop();
};
