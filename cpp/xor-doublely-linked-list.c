#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CURSOR_CURRENT(cursor) ((Node *)((cursor).current))
#define CURSOR_CURRENT_PREVNEXT(cursor) (CURSOR_CURRENT(cursor))->prevnext
#define CURSOR_CURRENT_DATA(cursor) (CURSOR_CURRENT(cursor))->data

// Node in the doubly linked list
typedef struct {
  uintptr_t prevnext;
  void *data;
} Node;

// A cursor pointing to a position in the doubly linked list
typedef struct {
  uintptr_t previous;
  uintptr_t current;
} Cursor;

// Some data to store in the list
typedef struct {
  int value;
} IntData;

IntData *create_int_data(int value) {
  IntData *d = (IntData *)malloc(sizeof(IntData));
  assert(d != NULL);
  d->value = value;
  return d;
}

Cursor create_singleton_list(int value) {
  Node *head = (Node *)malloc(sizeof(Node));
  assert(head != NULL);
  head->prevnext = 0;
  head->data = (void *)create_int_data(value);
  Cursor c = {0, (uintptr_t)head};
  return c;
}

void append(Cursor head, void *d) {
  // Cursor c = head;
  while (1) {
    // uintptr_t prevnext = ((Node *)c.current)->prevnext;
    uintptr_t next = head.previous ^ CURSOR_CURRENT_PREVNEXT(head);
    if (next == 0) {
      break;
    }
    head.previous = head.current;
    head.current = next;
    // head = {head.current, next};
    // c.previous = c.current;
    // c.current = next;
  }
  Node *new = (Node *)malloc(sizeof(Node));
  assert(new != NULL);
  uintptr_t newIntPrt = (uintptr_t)(new);
  new->prevnext = head.current ^ 0;
  new->data = d;
  ((Node *)head.current)->prevnext = newIntPrt ^ 0;
  // printf("End\n");

  // while (head->prevnext != 0) {
  //   head = (Node *)head->prevnext;
  // }
  // Node *next = (Node *)malloc(sizeof(Node));
  // assert(next != NULL);
  // next->prevnext = 0;
  // next->data = d;
  // head->prevnext = (uintptr_t)next;
}

// void append(Node *head, void *d) {
//   while (head->prevnext != 0) {
//     head = (Node *)head->prevnext;
//   }
//   Node *next = (Node *)malloc(sizeof(Node));
//   assert(next != NULL);
//   next->prevnext = 0;
//   next->data = d;
//   head->prevnext = (uintptr_t)next;
// }

void free_list(Node *head) {
  uintptr_t currentIntPtr = (uintptr_t)head;
  while (currentIntPtr != 0) {
    Node *currentPtr = (Node *)currentIntPtr;
    free(currentPtr->data);
    currentIntPtr = currentPtr->prevnext;
    free(currentPtr);
  }
}

void print_list(Cursor head) {
  Cursor cursor = head;
  while (1) {
    uintptr_t prevnext = CURSOR_CURRENT_PREVNEXT(cursor);
    uintptr_t next = cursor.previous ^ prevnext;
    if (next == 0) {
      break;
    }
    int value = ((IntData *)CURSOR_CURRENT_DATA(cursor))->value;
    printf("%d, ", value);
    cursor.previous = cursor.current;
    cursor.current = next;
  }

  // uintptr_t currentIntPtr = (uintptr_t)head;
  // while (currentIntPtr != 0) {
  //   Node *currentPtr = (Node *)currentIntPtr;
  //   int value = ((IntData *)(currentPtr->data))->value;
  //   printf("%d, ", value);
  //   currentIntPtr = currentPtr->prevnext;
  // }
  // printf("\n");
}

// void print_list(Node *head) {
//   uintptr_t currentIntPtr = (uintptr_t)head;
//   while (currentIntPtr != 0) {
//     Node *currentPtr = (Node *)currentIntPtr;
//     int value = ((IntData *)(currentPtr->data))->value;
//     printf("%d, ", value);
//     currentIntPtr = currentPtr->prevnext;
//   }
//   printf("\n");
// }

int main(int argc, char *argv[]) {
  Cursor head = create_singleton_list(4);
  append(head, create_int_data(5));
  print_list(head);
  // append(head, create_int_data(6));
  // print_list(head);
  // free_list(head);
  return 0;
}
