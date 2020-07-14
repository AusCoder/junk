#include <glib.h>
#include <gst/gst.h>

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;

  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
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

// static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
//   GstPad *sinkpad;
//   GstElement *decoder = (GstElement *)data;

//   g_print("Dynamic pad created, linking demuxer/decoder\n");
//   sinkpad = gst_element_get_static_pad(decoder, "sink");
//   gst_pad_link(pad, sinkpad);
//   gst_object_unref(sinkpad);
// }

int main(int argc, char *argv[]) {
  GMainLoop *loop;

  GstElement *pipeline, *source, *videoconvert, *encoder, *mux, *sink;
  GstBus *bus;
  guint bus_watch_id, major, minor, micro, nano;

  gst_version(&major, &minor, &micro, &nano);
  g_print("Using Gstreamer version: %d.%d.%d.%d\n", major, minor, micro, nano);

  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  pipeline = gst_pipeline_new("write-webcam");
  source = gst_element_factory_make("v4l2src", "video-source");
  videoconvert = gst_element_factory_make("videoconvert", "videoconvert");
  encoder = gst_element_factory_make("x264enc", "h264-encoder");
  mux = gst_element_factory_make("mp4mux", "muxer");
  sink = gst_element_factory_make("filesink", "file-sink");

  if (!pipeline || !source || !videoconvert || !encoder || !mux || !sink) {
    g_printerr("Element not created\n");
    return -1;
  }

  g_object_set(G_OBJECT(source), "num-buffers", 50, NULL);
  g_object_set(G_OBJECT(source), "device", "/dev/video0", NULL);
  g_object_set(G_OBJECT(sink), "location", "test.mp4", NULL);
  g_object_set(G_OBJECT(sink), "async", FALSE, NULL);

  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  gst_bin_add_many(GST_BIN(pipeline), source, videoconvert, encoder, mux, sink,
                   NULL);

  gst_element_link_many(source, videoconvert, encoder, mux, sink, NULL);

  g_print("Now playing\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  g_print("Running...\n");
  g_main_loop_run(loop);

  g_print("Returned, stopping\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);

  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return 0;
}
