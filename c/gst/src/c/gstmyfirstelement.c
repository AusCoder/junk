/**
 * SECTION:element-gstmyfirstelement
 *
 * The myfirstelement element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! myfirstelement ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstmyfirstelement.h"
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

GST_DEBUG_CATEGORY_STATIC(gst_myfirstelement_debug_category);
#define GST_CAT_DEFAULT gst_myfirstelement_debug_category

/* prototypes */

/* static void gst_myfirstelement_set_property(GObject *object, guint
 * property_id, */
/*                                             const GValue *value, */
/*                                             GParamSpec *pspec); */
/* static void gst_myfirstelement_get_property(GObject *object, guint
 * property_id, */
/*                                             GValue *value, GParamSpec
 * *pspec); */

static void gst_myfirstelement_finalize(GObject *object);
static GstCaps *gst_myfirstelement_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
                                                  GstCaps *caps,
                                                  GstCaps *filter);
static gboolean gst_myfirstelement_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_myfirstelement_get_unit_size(GstBaseTransform *trans,
                                                 GstCaps *caps, gsize *size);
static GstFlowReturn gst_myfirstelement_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf);

enum { PROP_0 };

/* pad templates */

static GstStaticPadTemplate gst_myfirstelement_src_template =
    GST_STATIC_PAD_TEMPLATE(
        "src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw"
                        ",format=RGB,height=100,width=100"));

static GstStaticPadTemplate gst_myfirstelement_sink_template =
    GST_STATIC_PAD_TEMPLATE(
        "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
            "video/x-raw"
            ",format=RGB,height=[1,2147483647],width=[1,2147483647]"));

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(
    GstMyfirstelement, gst_myfirstelement, GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(gst_myfirstelement_debug_category, "myfirstelement",
                            0, "debug category for myfirstelement element"));

static void gst_myfirstelement_class_init(GstMyfirstelementClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  GST_DEBUG("something");

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_myfirstelement_src_template);
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_myfirstelement_sink_template);

  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "FIXME Long name", "Generic",
      "FIXME Description", "FIXME <fixme@example.com>");

  // gobject_class->set_property = gst_myfirstelement_set_property;
  // gobject_class->get_property = gst_myfirstelement_get_property;

  gobject_class->finalize = gst_myfirstelement_finalize;

  base_transform_class->transform_caps =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_transform_caps);
  base_transform_class->set_caps =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_set_caps);
  base_transform_class->get_unit_size =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_get_unit_size);
  base_transform_class->transform =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_transform);
}

static void gst_myfirstelement_init(GstMyfirstelement *myfirstelement) {
  myfirstelement->ininfo = gst_video_info_new();
  myfirstelement->outinfo = gst_video_info_new();
}

/* void gst_myfirstelement_set_property(GObject *object, guint property_id, */
/*                                      const GValue *value, GParamSpec *pspec)
 * { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(object); */

/*   GST_DEBUG_OBJECT(myfirstelement, "set_property"); */

/*   switch (property_id) { */
/*   default: */
/*     G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec); */
/*     break; */
/*   } */
/* } */

/* void gst_myfirstelement_get_property(GObject *object, guint property_id, */
/*                                      GValue *value, GParamSpec *pspec) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(object); */

/*   GST_DEBUG_OBJECT(myfirstelement, "get_property"); */

/*   switch (property_id) { */
/*   default: */
/*     G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec); */
/*     break; */
/*   } */
/* } */

void gst_myfirstelement_finalize(GObject *object) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(object);

  GST_DEBUG_OBJECT(myfirstelement, "finalize");

  gst_video_info_free(myfirstelement->ininfo);
  gst_video_info_free(myfirstelement->outinfo);

  G_OBJECT_CLASS(gst_myfirstelement_parent_class)->finalize(object);
}

static GstCaps *gst_myfirstelement_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
                                                  GstCaps *caps,
                                                  GstCaps *filter) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans);
  GstCaps *othercaps;

  GST_DEBUG_OBJECT(myfirstelement, "transform_caps");

  othercaps = gst_caps_copy(caps);

  /* Copy other caps and modify as appropriate */
  /* This works for the simplest cases, where the transform modifies one
   * or more fields in the caps structure.  It does not work correctly
   * if passthrough caps are preferred. */

  for (guint idx = 0; idx < gst_caps_get_size(othercaps); idx++) {
    GstStructure *structure = gst_caps_get_structure(othercaps, idx);
    GValue val = G_VALUE_INIT;
    if (direction == GST_PAD_SRC) {
      g_value_init(&val, GST_TYPE_INT_RANGE);
      gst_value_set_int_range(&val, 1, G_MAXINT32);
      if (gst_structure_has_field(structure, "width")) {
        gst_structure_set_value(structure, "width", &val);
      }
      if (gst_structure_has_field(structure, "height")) {
        gst_structure_set_value(structure, "height", &val);
      }

    } else {
      g_value_init(&val, G_TYPE_INT);
      g_value_set_int(&val, 100);
      if (gst_structure_has_field(structure, "width")) {
        gst_structure_set_value(structure, "width", &val);
      }
      if (gst_structure_has_field(structure, "height")) {
        gst_structure_set_value(structure, "height", &val);
      }
    }
    g_value_unset(&val);
  }

  if (filter) {
    GstCaps *intersect;

    intersect = gst_caps_intersect(othercaps, filter);
    gst_caps_unref(othercaps);

    return intersect;
  } else {
    return othercaps;
  }
}

static gboolean gst_myfirstelement_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps, GstCaps *outcaps) {
  GstMyfirstelement *myelem = GST_MYFIRSTELEMENT(trans);
  GST_DEBUG_OBJECT(myelem, "set_caps");

  if (gst_video_info_from_caps(myelem->ininfo, incaps) &&
      gst_video_info_from_caps(myelem->outinfo, outcaps)) {
    return TRUE;
  }

  GST_DEBUG_OBJECT(myelem, "couldn't set video info from caps");
  return FALSE;
}

static gboolean gst_myfirstelement_get_unit_size(GstBaseTransform *trans,
                                                 GstCaps *caps, gsize *size) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans);
  GST_DEBUG_OBJECT(myfirstelement, "get_unit_size");
  GstVideoInfo *info = gst_video_info_new();
  gboolean ret = FALSE;
  if (gst_video_info_from_caps(info, caps)) {
    *size = GST_VIDEO_INFO_SIZE(info);
    ret = TRUE;
  }
  gst_video_info_free(info);
  return ret;
}

/* transform */
static GstFlowReturn gst_myfirstelement_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans);
  GST_DEBUG_OBJECT(myfirstelement, "transform");

  GstVideoFrame inframe, outframe;
  if (!gst_video_frame_map(&inframe, myfirstelement->ininfo, inbuf,
                           GST_MAP_READ))
    goto mapfail;
  if (!gst_video_frame_map(&outframe, myfirstelement->outinfo, outbuf,
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
  return GST_FLOW_OK;

mapfail:
  GST_WARNING_OBJECT(trans, "failed to map buffers");
  return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin *plugin) {

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register(plugin, "myfirstelement", GST_RANK_NONE,
                              GST_TYPE_MYFIRSTELEMENT);
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

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, myfirstelement,
                  "FIXME plugin description", plugin_init, VERSION, "LGPL",
                  PACKAGE_NAME, GST_PACKAGE_ORIGIN)
