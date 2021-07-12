#ifndef _QUEUE
#define _QUEUE

/*
  Queue using the SDL lock functions
*/

#include <SDL2/SDL.h>

#include <assert.h>
#include <stdlib.h>

#include "common.h"

struct _VPQueueItem;
typedef struct _VPQueueItem VPQueueItem;

struct _VPQueueItem {
  VPQueueItem *next;
  const void *value;
};

typedef struct {
  SDL_mutex *mutex;
  SDL_cond *cond;
  VPQueueItem *head;
  VPQueueItem *tail;
  int size;
  int maxSize;
} VPQueue;

/*
  maxSize = -1 for no max size
*/
VPQueue *queueAlloc(int maxSize) {
  SDL_mutex *mutex = SDL_CreateMutex();
  if (mutex == NULL) {
    assert(VP_FALSE);
    return NULL;
  }
  SDL_cond *cond = SDL_CreateCond();
  if (cond == NULL) {
    SDL_DestroyMutex(mutex);
    assert(VP_FALSE);
    return NULL;
  }
  VPQueue *q = (VPQueue *)malloc(sizeof(VPQueue));
  if (q == NULL) {
    assert(VP_FALSE);
    SDL_DestroyMutex(mutex);
    SDL_DestroyCond(cond);
    return NULL;
  }
  memset(q, 0, sizeof(VPQueue));
  q->mutex = mutex;
  q->cond = cond;
  q->head = NULL;
  q->tail = NULL;
  q->size = 0;
  q->maxSize = maxSize;
  return q;
}

void queueFree(VPQueue *q) {
  // LOG_INFO("%s", "calling queue free");
  SDL_LockMutex(q->mutex);
  VPQueueItem *i = q->head;
  while (i != NULL) {
    VPQueueItem *tmp = i->next;
    free(i);
    i = tmp;
  }
  q->head = NULL;
  q->tail = NULL;
  q->size = 0;
  SDL_UnlockMutex(q->mutex);
  SDL_DestroyCond(q->cond);
  SDL_DestroyMutex(q->mutex);
  free(q);
}

int queuePut(VPQueue *q, const void *value, int shouldBlock, int timeoutMs) {
  if (SDL_LockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_LockMutex failed");
    return VP_ERR_FATAL;
  }

  while ((q->maxSize > 0) && (q->size >= q->maxSize)) {
    int condRet;
    if (timeoutMs > 0) {
      condRet = SDL_CondWaitTimeout(q->cond, q->mutex, timeoutMs);
      if (condRet == SDL_MUTEX_TIMEDOUT) {
        if (SDL_UnlockMutex(q->mutex) < 0) {
          LOG_SDL_ERROR("SDL_UnlockMutex failed");
          return VP_ERR_FATAL;
        };
        return VP_ERR_TIMEDOUT;
      } else if (condRet < 0) {
        // if (SDL_UnlockMutex(q->mutex) < 0) {
        //   LOG_SDL_ERROR("SDL_UnlockMutex failed");
        //   return VP_ERR_FATAL;
        // };
        LOG_SDL_ERROR("SDL_CondWaitTimeout failed");
        return VP_ERR_FATAL;
      }
    } else if (shouldBlock) {
      if (SDL_CondWait(q->cond, q->mutex) < 0) {
        LOG_SDL_ERROR("SDL_CondWait failed");
        return VP_ERR_FATAL;
      }
    }
  }

  VPQueueItem *item = malloc(sizeof(VPQueueItem));
  if (item == NULL) {
    LOG_ERROR("%s", "malloc failed");
    return VP_ERR_FATAL;
  }
  item->next = NULL;
  item->value = value;

  if (q->head == NULL) {
    assert(q->tail == NULL);
    q->head = item;
  } else {
    q->tail->next = item;
  }
  q->tail = item;
  q->size++;
  if (SDL_CondSignal(q->cond) < 0) {
    LOG_SDL_ERROR("SDL_CondSignal failed");
    return VP_ERR_FATAL;
  }
  if (SDL_UnlockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_UnlockMutex failed");
    return VP_ERR_FATAL;
  };
  return 0;
}

int queueGet(VPQueue *q, const void **value, int shouldBlock, int timeoutMs) {
  if (SDL_LockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_LockMutex failed");
    return VP_ERR_FATAL;
  }

  int ret = 0;
  while (q->head == NULL) {
    if (timeoutMs > 0) {
      int condRet = SDL_CondWaitTimeout(q->cond, q->mutex, timeoutMs);
      if (condRet == SDL_MUTEX_TIMEDOUT) {
        ret = VP_ERR_TIMEDOUT;
        break;
      } else if (condRet < 0) {
        LOG_SDL_ERROR("SDL_CondWaitTimeout failed");
        return VP_ERR_FATAL;
      }
    } else if (shouldBlock) {
      if (SDL_CondWait(q->cond, q->mutex) < 0) {
        LOG_SDL_ERROR("SDL_CondWait failed");
        return VP_ERR_FATAL;
      }
    }
  }
  if ((q->head == NULL) && (ret == 0)) {
    ret = VP_ERR_AGAIN;
  }
  if (q->head != NULL) {
    assert(ret == 0);
    VPQueueItem *item = q->head;
    q->head = q->head->next;
    q->size--;
    if (q->head == NULL) {
      q->tail = NULL;
    }
    *value = item->value;
    free(item);
    if (SDL_CondSignal(q->cond) < 0) {
      LOG_SDL_ERROR("SDL_CondSignal failed");
      return VP_ERR_FATAL;
    }
    ret = 0;
  }
  if (SDL_UnlockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_UnlockMutex failed");
    return VP_ERR_FATAL;
  }
  return ret;
}

#endif // _QUEUE
