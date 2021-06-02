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

int queue_init(VPQueue *q) {
  q->mutex = SDL_CreateMutex();
  if (q->mutex == NULL) {
    return -1;
  }
  q->cond = SDL_CreateCond();
  q->head = NULL;
  q->tail = NULL;
  q->size = 0;

  return 0;
}

void queue_close(VPQueue *q) {
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
}

int queue_put(VPQueue *q, const void *value) {
  // TODO: replace with cond
  if (SDL_LockMutex(q->mutex) < 0) {
    return -1;
  }
  VPQueueItem *item = malloc(sizeof(VPQueueItem));
  item->next = NULL;
  item->value = value;
  if (q->head == NULL) {
    assert(q->tail == NULL);
    q->head = item;
    q->tail = item;
  } else {
    q->tail->next = item;
    q->tail = item;
  }
  if (SDL_UnlockMutex(q->mutex) < 0) {
    return -1;
  };
  return 0;
}

int queue_get(VPQueue *q, void **value) { return 0; }

#endif // _QUEUE
