#include <stdio.h>

typedef void (*fnptr)(void);

void log_fnptr(fnptr ptr) { printf("Was passed a fnptr at %p\n", ptr); }

int func(int x) { return x * 2; }

int main(int argc, char **argv) {
  int (*ptr)(int) = (log_fnptr((fnptr)func), func);
  printf("ptr is %p\n", ptr);
  printf("success\n");
  return 0;
}
