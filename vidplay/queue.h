#ifndef _QUEUE
#define _QUEUE

/*
  Queue using the SDL lock functions
*/

#include <SDL2/SDL.h>

#include <assert.h>
#include <stdlib.h>

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
} VPQueue;

VPQueue *queue_alloc() {
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

void queue_free(VPQueue *q) {
  SDL_LockMutex(q->mutex);
  VPQueueItem *i = q->head;
  while (i != NULL) {
    VPQueueItem *tmp = i->next;
    free(i);
    i = tmp;
  }
  SDL_UnlockMutex(q->mutex);

  SDL_DestroyCond(q->cond);
  SDL_DestroyMutex(q->mutex);
  q->head = NULL;
  q->tail = NULL;
  q->size = 0;
  free(q);
}

int queue_put(VPQueue *q, const void *value) {
  VPQueueItem *item = malloc(sizeof(VPQueueItem));
  if (item == NULL) {
    return -2;
  }
  item->next = NULL;
  item->value = value;

  if (SDL_LockMutex(q->mutex) < 0) {
    return -1;
  }
  if (q->head == NULL) {
    assert(q->tail == NULL);
    q->head = item;
  } else {
    q->tail->next = item;
  }
  q->tail = item;
  q->size++;
  if (SDL_CondSignal(q->cond) < 0) {
    return -1;
  }
  if (SDL_UnlockMutex(q->mutex) < 0) {
    return -1;
  };
  return 0;
}

int queue_get(VPQueue *q, const void **value) {
  if (SDL_LockMutex(q->mutex) < 0) {
    return -1;
  }
  while (q->head == NULL) {
    if (SDL_CondWait(q->cond, q->mutex) < 0) {
      return -1;
    }
  }
  VPQueueItem *item = q->head;
  q->head = q->head->next;
  q->size--;
  if (q->head == NULL) {
    q->tail = NULL;
  }
  *value = item->value;
  free(item);
  if (SDL_UnlockMutex(q->mutex) < 0) {
    return -1;
  }
  return 0;
}

#endif // _QUEUE
