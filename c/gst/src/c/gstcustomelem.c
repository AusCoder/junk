#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstcustomelem.h"
#include <gst/gst.h>

GST_DEBUG_CATEGORY_STATIC(gst_customelem_debug_category);
#define GST_CAT_DEFAULT gst_customelem_debug_category

/* prototypes */

/* static void gst_customelem_set_property(GObject *object, guint
 * property_id, */
/*                                             const GValue *value, */
/*                                             GParamSpec *pspec); */
/* static void gst_customelem_get_property(GObject *object, guint
 * property_id, */
/*                                             GValue *value, GParamSpec
 * *pspec); */

static void gst_customelem_finalize(GObject *object);
static GstFlowReturn gst_customelem_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer);
static gboolean gst_customelem_sink_event(GstPad *pad, GstObject *parent, GstEvent *event);
static gboolean gst_customelem_src_event(GstPad *pad, GstObject *parent, GstEvent *event);

enum { PROP_0 };

/* pad templates */

static GstStaticPadTemplate gst_customelem_src_template =
    GST_STATIC_PAD_TEMPLATE(
        "src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw"
                        ",format=RGB,height=100,width=100"));

static GstStaticPadTemplate gst_customelem_sink_template =
    GST_STATIC_PAD_TEMPLATE(
        "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
            "video/x-raw"
            ",format=RGB,height=[1,2147483647],width=[1,2147483647]"));

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(
    GstCustomelem, gst_customelem, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_customelem_debug_category, "customelem",
                            0, "debug category for customelem element"));

static void gst_customelem_class_init(GstCustomelemClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_customelem_src_template);
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_customelem_sink_template);

  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "FIXME Long name", "Generic",
      "FIXME Description", "FIXME <fixme@example.com>");

  // gobject_class->set_property = gst_customelem_set_property;
  // gobject_class->get_property = gst_customelem_get_property;

  gobject_class->finalize = gst_customelem_finalize;
}

static void gst_customelem_init(GstCustomelem *customelem) {
  customelem->srcpad = gst_pad_new_from_static_template(&gst_customelem_src_template, "srcpad");
  customelem->sinkpad = gst_pad_new_from_static_template(&gst_customelem_sink_template, "sinkpad");
  // Q: Do I need to free or remove this sometime?
  gst_element_add_pad(GST_ELEMENT(customelem), customelem->srcpad);
  gst_element_add_pad(GST_ELEMENT(customelem), customelem->sinkpad);
  gst_pad_set_event_function(customelem->sinkpad, gst_customelem_sink_event);
  gst_pad_set_event_function(customelem->srcpad, gst_customelem_src_event);
  gst_pad_set_chain_function(customelem->sinkpad, gst_customelem_chain);

  customelem->outcaps = gst_static_caps_get(&gst_customelem_src_template.static_caps);
  customelem->ininfo = gst_video_info_new();
  if (!gst_caps_is_fixed(customelem->outcaps)) {
    GST_ERROR_OBJECT(customelem, "src caps not fixed!");
  }
  customelem->outinfo = gst_video_info_new();
  if (!gst_video_info_from_caps(customelem->outinfo, customelem->outcaps)) {
    GST_WARNING_OBJECT(customelem, "couldn't read video info from src caps");
  }
}

/* void gst_customelem_set_property(GObject *object, guint property_id, */
/*                                      const GValue *value, GParamSpec *pspec)
 * { */
/*   GstCustomelem *customelem = GST_CUSTOMELEM(object); */

/*   GST_DEBUG_OBJECT(customelem, "set_property"); */

/*   switch (property_id) { */
/*   default: */
/*     G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec); */
/*     break; */
/*   } */
/* } */

/* void gst_customelem_get_property(GObject *object, guint property_id, */
/*                                      GValue *value, GParamSpec *pspec) { */
/*   GstCustomelem *customelem = GST_CUSTOMELEM(object); */

/*   GST_DEBUG_OBJECT(customelem, "get_property"); */

/*   switch (property_id) { */
/*   default: */
/*     G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec); */
/*     break; */
/*   } */
/* } */

void gst_customelem_finalize(GObject *object) {
  GstCustomelem *customelem = GST_CUSTOMELEM(object);

  GST_DEBUG_OBJECT(customelem, "finalize");

  gst_video_info_free(customelem->ininfo);
  gst_video_info_free(customelem->outinfo);
  gst_caps_unref(customelem->outcaps);

  G_OBJECT_CLASS(gst_customelem_parent_class)->finalize(object);
}

static GstFlowReturn gst_customelem_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer) {
  GstCustomelem *customelem = GST_CUSTOMELEM(parent);

  gint size = GST_VIDEO_INFO_SIZE(customelem->outinfo);
  GstBuffer *outbuffer = gst_buffer_new_allocate(NULL, size, NULL);
  if (!outbuffer)
    goto allocfail;

  GstVideoFrame inframe, outframe;
  if (!gst_video_frame_map(&inframe, customelem->ininfo, buffer,
                           GST_MAP_READ))
    goto mapfail;
 if (!gst_video_frame_map(&outframe, customelem->outinfo, outbuffer,
                           GST_MAP_WRITE)) {
    gst_video_frame_unmap(&inframe);
    goto mapfail;
  }

  guint8 *indata = GST_VIDEO_FRAME_PLANE_DATA(&inframe, 0);
  guint instride = GST_VIDEO_FRAME_PLANE_STRIDE(&inframe, 0);
  guint inpixstride = GST_VIDEO_FRAME_COMP_PSTRIDE(&inframe, 0);
  guint8 *outdata = GST_VIDEO_FRAME_PLANE_DATA(&outframe, 0);
  guint outstride = GST_VIDEO_FRAME_PLANE_STRIDE(&outframe, 0);

  for (guint h = 0; h < MIN(GST_VIDEO_FRAME_HEIGHT(&inframe),
                            GST_VIDEO_FRAME_HEIGHT(&outframe));
       h++) {
    for (guint w = 0; w < MIN(GST_VIDEO_FRAME_WIDTH(&inframe),
                              GST_VIDEO_FRAME_WIDTH(&outframe));
         w++) {
      guint8 *inpix = indata + h * instride + w * inpixstride;
      guint8 *outpix = outdata + h * outstride + w * inpixstride;
      memcpy(outpix, inpix, inpixstride);
    }
  }

  gst_video_frame_unmap(&inframe);
  gst_video_frame_unmap(&outframe);
  gst_buffer_unref(buffer);
  return gst_pad_push(customelem->srcpad, outbuffer);

mapfail:
  GST_WARNING_OBJECT(customelem, "failed to map buffers");
  gst_buffer_unref(outbuffer);
  gst_buffer_unref(buffer);
  return GST_FLOW_ERROR;
 allocfail:
  GST_ERROR_OBJECT(customelem, "failed to allocate buffer");
  gst_buffer_unref(buffer);
  return GST_FLOW_ERROR;
}

static gboolean gst_customelem_sink_event(GstPad *pad, GstObject *parent, GstEvent *event) {
  GstCustomelem *customelem = GST_CUSTOMELEM(parent);
  gboolean ret;

  GST_DEBUG_OBJECT(customelem, "received event on sinkpad %" GST_PTR_FORMAT, event);

  switch (GST_EVENT_TYPE(event)) {
  case GST_EVENT_CAPS:
    GstCaps *caps;
    gst_event_parse_caps(event, &caps);
    if (!gst_video_info_from_caps(customelem->ininfo, caps)) {
      GST_WARNING_OBJECT(customelem, "failed to parse video info from caps");
    }
    gst_event_unref(event);
    GstEvent *newevent = gst_event_new_caps(customelem->outcaps);
    GST_DEBUG_OBJECT(customelem, "pushing event %" GST_PTR_FORMAT, newevent);
    ret = gst_pad_push_event(customelem->srcpad, newevent);
    break;
  default:
    ret = gst_pad_event_default(pad, parent, event);
    break;
  }

  return ret;
}

static gboolean gst_customelem_src_event(GstPad *pad, GstObject *parent, GstEvent *event) {
  GstCustomelem *customelem = GST_CUSTOMELEM(parent);
  GST_DEBUG_OBJECT(customelem, "received event on srcpad %" GST_PTR_FORMAT, event);
  return gst_pad_event_default(pad, parent, event);
}

static gboolean plugin_init(GstPlugin *plugin) {

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register(plugin, "customelem", GST_RANK_NONE,
                              GST_TYPE_CUSTOMELEM);
}

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */
#ifndef VERSION
#define VERSION "0.0.FIXME"
#endif
#ifndef PACKAGE
#define PACKAGE "FIXME_package"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "FIXME_package_name"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://FIXME.org/"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, customelem,
                  "FIXME plugin description", plugin_init, VERSION, "LGPL",
                  PACKAGE_NAME, GST_PACKAGE_ORIGIN)
