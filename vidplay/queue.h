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
  // TODO: add max size and block on queuePut
} VPQueue;

VPQueue *queueAlloc() {
  SDL_mutex *mutex = SDL_CreateMutex();
  if (mutex == NULL) {
    return NULL;
  }
  SDL_cond *cond = SDL_CreateCond();
  if (cond == NULL) {
    SDL_DestroyMutex(mutex);
    return NULL;
  }
  VPQueue *q = (VPQueue *)malloc(sizeof(VPQueue));
  if (q == NULL) {
    SDL_DestroyMutex(mutex);
    SDL_DestroyCond(cond);
    return NULL;
  }
  q->mutex = mutex;
  q->cond = cond;
  q->head = NULL;
  q->tail = NULL;
  q->size = 0;
  return q;
}

void queueFree(VPQueue *q) {
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

int queuePut(VPQueue *q, const void *value) {
  if (SDL_LockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_LockMutex failed");
    return VP_ERR_FATAL;
  }

  VPQueueItem *item = malloc(sizeof(VPQueueItem));
  if (item == NULL) {
    LOG_ERROR("malloc failed");
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

int queueGet(VPQueue *q, const void **value, int should_block, int timeout_ms) {
  if (SDL_LockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_LockMutex failed");
    return VP_ERR_FATAL;
  }

  if (timeout_ms > 0) {
    // NB: I might not need timeout, I could signal the
    // queue when I want to quit
    while (q->head == NULL) {
      int ret = SDL_CondWaitTimeout(q->cond, q->mutex, timeout_ms);
      if (ret == SDL_MUTEX_TIMEDOUT) {
        break;
      } else if (ret < 0) {
        LOG_SDL_ERROR("SDL_CondWaitTimeout failed");
        return VP_ERR_FATAL;
      }
    }
  } else if (should_block) {
    while (q->head == NULL) {
      if (SDL_CondWait(q->cond, q->mutex) < 0) {
        LOG_SDL_ERROR("SDL_CondWait failed");
        return VP_ERR_FATAL;
      }
    }
  }

  int ret = 0;
  if (q->head == NULL) {
    ret = VP_ERR_AGAIN;
  } else {
    VPQueueItem *item = q->head;
    q->head = q->head->next;
    q->size--;
    if (q->head == NULL) {
      q->tail = NULL;
    }
    *value = item->value;
    free(item);
    ret = 0;
  }
  if (SDL_UnlockMutex(q->mutex) < 0) {
    LOG_SDL_ERROR("SDL_UnlockMutex failed");
    return VP_ERR_FATAL;
  }
  return ret;
}

#endif // _QUEUE
