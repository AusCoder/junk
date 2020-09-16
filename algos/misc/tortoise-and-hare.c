#include <stdio.h>
#include <stdlib.h>

struct Elem {
  struct Elem *next;
  void *data;
};

struct Data {
  int value;
};

struct Elem *detectCycle(struct Elem *head) {
  struct Elem *tortoise = head;
  struct Elem *hare = head;

  while ((hare != NULL) && (tortoise != NULL) && (tortoise->next != NULL)) {
    hare = hare->next;
    tortoise = tortoise->next->next;
    if (hare == tortoise) {
      break;
    }
  }

  if ((hare == NULL) || (tortoise == NULL) || (tortoise->next == NULL)) {
    return NULL;
  } else {
    hare = head;
    while (hare != tortoise) {
      hare = hare->next;
      tortoise = tortoise->next;
    }
    return hare;
  }
}

void printList(struct Elem *head) {
  while (head != NULL) {
    int value = *(int *)head->data;
    printf("Value %d\n", value);
    head = head->next;
  }
}

struct Elem *createListAndData(int hasCycle) {
  struct Elem *head = malloc(sizeof(struct Elem));
  head->data = malloc(sizeof(struct Data));
  ((struct Data *)head->data)->value = 2;

  head->next = malloc(sizeof(struct Elem));
  head->next->data = malloc(sizeof(struct Data));
  ((struct Data *)head->next->data)->value = 0;

  head->next->next = malloc(sizeof(struct Elem));
  head->next->next->data = malloc(sizeof(struct Data));
  ((struct Data *)head->next->next->data)->value = 1;

  head->next->next->next = malloc(sizeof(struct Elem));
  head->next->next->next->data = malloc(sizeof(struct Data));
  ((struct Data *)head->next->next->next->data)->value = 3;

  head->next->next->next->next = malloc(sizeof(struct Elem));
  head->next->next->next->next->data = malloc(sizeof(struct Data));
  ((struct Data *)head->next->next->next->next->data)->value = 4;

  if (hasCycle) {
    head->next->next->next->next->next = head->next->next;
  } else {
    head->next->next->next->next->next = NULL;
  }
  return head;
}

void freeListAndData(struct Elem *head, struct Elem *cyclePoint) {
  // Break the cycle
  if (cyclePoint) {
    struct Elem *tmp = head;
    int hitCount = 0;
    while ((hitCount == 0) || (tmp->next != cyclePoint)) {
      if (tmp->next == cyclePoint) {
        hitCount++;
      }
      tmp = tmp->next;
    }
    tmp->next = NULL;
  }

  while (head != NULL) {
    free(head->data);
    struct Elem *next = head->next;
    free(head);
    head = next;
  }
}

int main(int argc, char **argv) {
  int hasCycle = 1;
  struct Elem *head = createListAndData(hasCycle);
  if (!hasCycle) {
    printList(head);
  }
  struct Elem *cyclePoint = detectCycle(head);
  if (cyclePoint) {
    int value = ((struct Data *)cyclePoint->data)->value;
    printf("Detected cycle at value %d\n", value);
  } else {
    printf("No cycle detected\n");
  }
  freeListAndData(head, cyclePoint);
  printf("Success\n");
}
