#include <glib.h>

gboolean func(gpointer ptr) {
  g_return_val_if_fail(ptr != NULL, FALSE);
  return TRUE;
}

int main(int argc, char **argv) {
  int x = 1;
  gpointer ptr = &x;
  if (func(ptr)) {
    g_print("success\n");
  } else {
    g_print("fail\n");
  }
}
