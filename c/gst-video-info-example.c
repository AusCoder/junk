#include <glib.h>
#include <gst/gst.h>
#include <gst/video/video.h>

int main(int argc, char **argv) {
  gst_init(NULL, NULL);

  GstCaps *caps =
      gst_caps_from_string("video/x-raw,format=RGB,height=100,width=100");
  GstVideoInfo *info = gst_video_info_new();
  if (gst_video_info_from_caps(info, caps)) {
    g_print("Video frame size: %lu\n", GST_VIDEO_INFO_SIZE(info));
  } else {
    g_print("Couldn't get video info from caps\n");
  }
  gst_video_info_free(info);
  gst_caps_unref(caps);
  g_print("success\n");
}
