/*
Compile with:
    gcc basic-tutorial-1.c -o basic-tutorial-1 \
        `pkg-config --cflags --libs gstreamer-1.0`
*/

#include <gst/gst.h>

static gboolean bus_callback(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;

  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS: {
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  }
  case GST_MESSAGE_ERROR: {
    gchar *debug;
    GError *error;

    gst_message_parse_error(msg, &error, &debug);
    g_free(debug);

    g_printerr("Error: %s\n", error->message);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

int main(int argc, char *argv[]) {
  GMainLoop *loop;
  GstElement *pipeline;
  GstBus *bus;
  // GstMessage *msg;

  /* Initialize GStreamer */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Build the pipeline */
  pipeline = gst_parse_launch("playbin "
                              "uri=dev:///dev/video0",
                              NULL);

  /* Start playing */
  g_print("Now playing\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait until error or EOS */
  bus = gst_element_get_bus(pipeline);
  // msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
  //                                  GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
  guint bus_watch_id = gst_bus_add_watch(bus, bus_callback, loop);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  g_print("Running...\n");
  g_main_loop_run(loop);

  g_print("Returned, stopping\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
