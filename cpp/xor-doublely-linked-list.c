/* A doubly linked list where each node stores links
  as add(prev) ^ addr(next). Using properties of xor,
  we can traverse the list as normal.
  It uses a bit less space than holding forward and backward
  pointers on each node.
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NODE_PTR_AS_ADDR(ptr) ((uintptr_t)(ptr))
#define ADDR_AS_NODE_PTR(addr) ((Node *)(addr))
#define CURSOR_CURRENT(cursor) (ADDR_AS_NODE_PTR((cursor).current))
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
  while (1) {
    uintptr_t next = head.previous ^ CURSOR_CURRENT_PREVNEXT(head);
    if (next == 0) {
      break;
    }
    head.previous = head.current;
    head.current = next;
  }
  Node *new = (Node *)malloc(sizeof(Node));
  assert(new != NULL);
  CURSOR_CURRENT_PREVNEXT(head) = head.previous ^ NODE_PTR_AS_ADDR(new);
  new->prevnext = head.current ^ 0;
  new->data = d;
}

void free_list(Cursor head) {
  while (head.current != 0) {
    if (CURSOR_CURRENT_DATA(head)) {
      free(CURSOR_CURRENT_DATA(head));
    }
    Node *tmp = CURSOR_CURRENT(head);
    uintptr_t next = CURSOR_CURRENT_PREVNEXT(head) ^ head.previous;
    head.previous = head.current;
    head.current = next;
    // head.current->prevnext is broken as this point, but
    // we are freeing the list, so it's not a big deal. Also
    // head.previous is pointing to freed memory...
    // We'd want:
    // next->prevnext = 0 ^ (next->prevnext ^ head.current)
    // head.previous = 0
    // Or something like this...
    free(tmp);
  }
}

void print_list(Cursor head) {
  while (head.current != 0) {
    int value = ((IntData *)CURSOR_CURRENT_DATA(head))->value;
    printf("%d, ", value);
    uintptr_t next = CURSOR_CURRENT_PREVNEXT(head) ^ head.previous;
    head.previous = head.current;
    head.current = next;
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  Cursor head = create_singleton_list(4);
  append(head, create_int_data(5));
  append(head, create_int_data(6));
  print_list(head);
  free_list(head);
  return 0;
}
