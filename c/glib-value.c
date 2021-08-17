#include <glib.h>
#include <glib-object.h>

int main(int argc, char **argv) {
  GValue value;
  g_value_init(&value, G_TYPE_INT);
  g_value_set_int(&value, 100);
  g_print("Value is %d\n", g_value_get_int(&value));
  g_assert(FALSE);
}
