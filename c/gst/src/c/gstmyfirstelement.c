/* GStreamer
 * Copyright (C) 2021 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
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

/* static void gst_myfirstelement_set_property(GObject *object, guint property_id, */
/*                                             const GValue *value, */
/*                                             GParamSpec *pspec); */
/* static void gst_myfirstelement_get_property(GObject *object, guint property_id, */
/*                                             GValue *value, GParamSpec *pspec); */
/* static void gst_myfirstelement_dispose(GObject *object); */
/* static void gst_myfirstelement_finalize(GObject *object); */

static GstCaps *gst_myfirstelement_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
                                                  GstCaps *caps,
                                                  GstCaps *filter);

/* static GstCaps *gst_myfirstelement_fixate_caps(GstBaseTransform *trans, */
/*                                                GstPadDirection direction, */
/*                                                GstCaps *caps, */
/*                                                GstCaps *othercaps); */
/* static gboolean gst_myfirstelement_accept_caps(GstBaseTransform *trans, */
/*                                                GstPadDirection direction, */
/*                                                GstCaps *caps); */
static gboolean gst_myfirstelement_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps, GstCaps *outcaps);
/* static gboolean gst_myfirstelement_query(GstBaseTransform *trans, */
/*                                          GstPadDirection direction, */
/*                                          GstQuery *query); */
/* static gboolean gst_myfirstelement_decide_allocation(GstBaseTransform *trans, */
/*                                                      GstQuery *query); */
/* static gboolean gst_myfirstelement_filter_meta(GstBaseTransform *trans, */
/*                                                GstQuery *query, GType api, */
/*                                                const GstStructure *params); */
/* static gboolean gst_myfirstelement_propose_allocation(GstBaseTransform *trans, */
/*                                                       GstQuery *decide_query, */
/*                                                       GstQuery *query); */
/* static gboolean gst_myfirstelement_transform_size(GstBaseTransform *trans, */
/*                                                   GstPadDirection direction, */
/*                                                   GstCaps *caps, gsize size, */
/*                                                   GstCaps *othercaps, */
/*                                                   gsize *othersize); */
static gboolean gst_myfirstelement_get_unit_size(GstBaseTransform *trans,
                                                 GstCaps *caps, gsize *size);
/* static gboolean gst_myfirstelement_start(GstBaseTransform *trans); */
/* static gboolean gst_myfirstelement_stop(GstBaseTransform *trans); */
/* static gboolean gst_myfirstelement_sink_event(GstBaseTransform *trans, */
/*                                               GstEvent *event); */
/* static gboolean gst_myfirstelement_src_event(GstBaseTransform *trans, */
/*                                              GstEvent *event); */

/* static GstFlowReturn */
/* gst_myfirstelement_prepare_output_buffer(GstBaseTransform *trans, */
/*                                          GstBuffer *input, GstBuffer **outbuf); */

/* static gboolean gst_myfirstelement_copy_metadata(GstBaseTransform *trans, */
/*                                                  GstBuffer *input, */
/*                                                  GstBuffer *outbuf); */
/* static gboolean gst_myfirstelement_transform_meta(GstBaseTransform *trans, */
/*                                                   GstBuffer *outbuf, */
/*                                                   GstMeta *meta, */
/*                                                   GstBuffer *inbuf); */
/* static void gst_myfirstelement_before_transform(GstBaseTransform *trans, */
/*                                                 GstBuffer *buffer); */
static GstFlowReturn gst_myfirstelement_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf);
/* static GstFlowReturn gst_myfirstelement_transform_ip(GstBaseTransform *trans, */
/*                                                      GstBuffer *buf); */

enum { PROP_0 };

/* pad templates */

static GstStaticPadTemplate gst_myfirstelement_src_template =
    GST_STATIC_PAD_TEMPLATE(
        "src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
            "video/x-raw"
            ",format=RGB,height=100,width=100"));

static GstStaticPadTemplate gst_myfirstelement_sink_template =
    GST_STATIC_PAD_TEMPLATE(
        "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw"
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
  // gobject_class->dispose = gst_myfirstelement_dispose;
  // gobject_class->finalize = gst_myfirstelement_finalize;

  base_transform_class->transform_caps =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_transform_caps);

  // base_transform_class->fixate_caps =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_fixate_caps);
  // base_transform_class->accept_caps =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_accept_caps);
  base_transform_class->set_caps =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_set_caps);
  // base_transform_class->query = GST_DEBUG_FUNCPTR(gst_myfirstelement_query);
  // base_transform_class->decide_allocation =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_decide_allocation);
  // base_transform_class->filter_meta =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_filter_meta);
  // base_transform_class->propose_allocation =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_propose_allocation);
  // base_transform_class->transform_size =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_transform_size);

  base_transform_class->get_unit_size =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_get_unit_size);

  // base_transform_class->start = GST_DEBUG_FUNCPTR(gst_myfirstelement_start);
  // base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_myfirstelement_stop);
  // base_transform_class->sink_event =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_sink_event);
  // base_transform_class->src_event =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_src_event);

  /* base_transform_class->prepare_output_buffer = */
  /*     GST_DEBUG_FUNCPTR(gst_myfirstelement_prepare_output_buffer); */

  // base_transform_class->copy_metadata =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_copy_metadata);
  // base_transform_class->transform_meta =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_transform_meta);
  // base_transform_class->before_transform =
  //     GST_DEBUG_FUNCPTR(gst_myfirstelement_before_transform);

  base_transform_class->transform =
      GST_DEBUG_FUNCPTR(gst_myfirstelement_transform);

  /* base_transform_class->transform_ip = */
      /* GST_DEBUG_FUNCPTR(gst_myfirstelement_transform_ip); */
}

static void gst_myfirstelement_init(GstMyfirstelement *myfirstelement) {
  myfirstelement->inheight = 0;
  myfirstelement->inwidth = 0;
  myfirstelement->outheight = 0;
  myfirstelement->outwidth = 0;
}

/* void gst_myfirstelement_set_property(GObject *object, guint property_id, */
/*                                      const GValue *value, GParamSpec *pspec) { */
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

/* void gst_myfirstelement_dispose(GObject *object) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(object); */

/*   GST_DEBUG_OBJECT(myfirstelement, "dispose"); */

/*   /\* clean up as possible.  may be called multiple times *\/ */

/*   G_OBJECT_CLASS(gst_myfirstelement_parent_class)->dispose(object); */
/* } */

/* void gst_myfirstelement_finalize(GObject *object) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(object); */

/*   GST_DEBUG_OBJECT(myfirstelement, "finalize"); */

/*   /\* clean up object here *\/ */

/*   G_OBJECT_CLASS(gst_myfirstelement_parent_class)->finalize(object); */
/* } */

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

/* static GstCaps *gst_myfirstelement_fixate_caps(GstBaseTransform *trans, */
/*                                                GstPadDirection direction, */
/*                                                GstCaps *caps, */
/*                                                GstCaps *othercaps) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "fixate_caps"); */

/*   return NULL; */
/* } */

/* static gboolean gst_myfirstelement_accept_caps(GstBaseTransform *trans, */
/*                                                GstPadDirection direction, */
/*                                                GstCaps *caps) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "accept_caps"); */

/*   return TRUE; */
/* } */

static gboolean gst_myfirstelement_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps, GstCaps *outcaps) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans);

  GST_DEBUG_OBJECT(myfirstelement, "set_caps");

  GstStructure *instructure = gst_caps_get_structure(incaps, 0);
  GstStructure *outstructure = gst_caps_get_structure(outcaps, 0);
  if (gst_structure_get_int(instructure, "height", &myfirstelement->inheight) &&
      gst_structure_get_int(instructure, "width", &myfirstelement->inwidth) &&
      gst_structure_get_int(outstructure, "height", &myfirstelement->outheight) &&
      gst_structure_get_int(outstructure, "width", &myfirstelement->outwidth)) {
    return TRUE;
  }
  GST_DEBUG_OBJECT(myfirstelement, "couldn't find height and width on caps");
  return FALSE;
}

/* static gboolean gst_myfirstelement_query(GstBaseTransform *trans, */
/*                                          GstPadDirection direction, */
/*                                          GstQuery *query) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "query"); */

/*   return TRUE; */
/* } */

/* /\* decide allocation query for output buffers *\/ */
/* static gboolean gst_myfirstelement_decide_allocation(GstBaseTransform *trans, */
/*                                                      GstQuery *query) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "decide_allocation"); */

/*   return TRUE; */
/* } */

/* static gboolean gst_myfirstelement_filter_meta(GstBaseTransform *trans, */
/*                                                GstQuery *query, GType api, */
/*                                                const GstStructure *params) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "filter_meta"); */

/*   return TRUE; */
/* } */

/* /\* propose allocation query parameters for input buffers *\/ */
/* static gboolean gst_myfirstelement_propose_allocation(GstBaseTransform *trans, */
/*                                                       GstQuery *decide_query, */
/*                                                       GstQuery *query) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "propose_allocation"); */

/*   return TRUE; */
/* } */

/* /\* transform size *\/ */
/* static gboolean gst_myfirstelement_transform_size(GstBaseTransform *trans, */
/*                                                   GstPadDirection direction, */
/*                                                   GstCaps *caps, gsize size, */
/*                                                   GstCaps *othercaps, */
/*                                                   gsize *othersize) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "transform_size"); */

/*   return TRUE; */
/* } */

static gboolean gst_myfirstelement_get_unit_size(GstBaseTransform *trans,
                                                 GstCaps *caps, gsize *size) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans);

  GST_DEBUG_OBJECT(myfirstelement, "get_unit_size");

  GstStructure *structure = gst_caps_get_structure(caps, 0);
  gint width, height;
  if (gst_structure_get_int(structure, "width", &width) &&
      gst_structure_get_int(structure, "height", &height)) {
    *size = width * height * 3;
    return TRUE;
  }

  // maybe use GST_ELEMENT_ERROR?
  GST_DEBUG_OBJECT(myfirstelement, "failed to read width, height on caps");
  return FALSE;
}

/* /\* states *\/ */
/* static gboolean gst_myfirstelement_start(GstBaseTransform *trans) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "start"); */

/*   return TRUE; */
/* } */

/* static gboolean gst_myfirstelement_stop(GstBaseTransform *trans) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "stop"); */

/*   return TRUE; */
/* } */

/* /\* sink and src pad event handlers *\/ */
/* static gboolean gst_myfirstelement_sink_event(GstBaseTransform *trans, */
/*                                               GstEvent *event) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "sink_event"); */

/*   return GST_BASE_TRANSFORM_CLASS(gst_myfirstelement_parent_class) */
/*       ->sink_event(trans, event); */
/* } */

/* static gboolean gst_myfirstelement_src_event(GstBaseTransform *trans, */
/*                                              GstEvent *event) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "src_event"); */

/*   return GST_BASE_TRANSFORM_CLASS(gst_myfirstelement_parent_class) */
/*       ->src_event(trans, event); */
/* } */

/* static GstFlowReturn */
/* gst_myfirstelement_prepare_output_buffer(GstBaseTransform *trans, */
/*                                          GstBuffer *input, GstBuffer **outbuf) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "prepare_output_buffer"); */

/*   return GST_FLOW_OK; */
/* } */

/* /\* metadata *\/ */
/* static gboolean gst_myfirstelement_copy_metadata(GstBaseTransform *trans, */
/*                                                  GstBuffer *input, */
/*                                                  GstBuffer *outbuf) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "copy_metadata"); */

/*   return TRUE; */
/* } */

/* static gboolean gst_myfirstelement_transform_meta(GstBaseTransform *trans, */
/*                                                   GstBuffer *outbuf, */
/*                                                   GstMeta *meta, */
/*                                                   GstBuffer *inbuf) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "transform_meta"); */

/*   return TRUE; */
/* } */

/* static void gst_myfirstelement_before_transform(GstBaseTransform *trans, */
/*                                                 GstBuffer *buffer) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   GST_DEBUG_OBJECT(myfirstelement, "before_transform"); */
/* } */

/* transform */
static GstFlowReturn gst_myfirstelement_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf) {
  GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans);
  GstElement *elem = GST_ELEMENT(trans);

  GstMapInfo inmapinfo, outmapinfo;

  GST_DEBUG_OBJECT(myfirstelement, "transform");
  if (!gst_buffer_map(inbuf, &inmapinfo, GST_MAP_READ))
    goto mapfail;
  if (!gst_buffer_map(outbuf, &outmapinfo, GST_MAP_WRITE)) {
    gst_buffer_unmap(inbuf, &inmapinfo);
    goto mapfail;
  }

  GstPad *sinkpad = GST_PAD(g_list_nth_data(elem->sinkpads, 0));
  GstCaps *caps = gst_pad_get_current_caps(sinkpad);
  g_print("my current sink caps %s\n", gst_caps_to_string(caps));
  gst_caps_unref(caps);

  /* memset(outmapinfo.data, 128, outmapinfo.size); */
  /* memcpy(outmapinfo.data, inmapinfo.data, outmapinfo.size); */

  // Q: How do I know the current height and width of my caps here?
  // A: use the set_caps method!

  // All sorts of pointer overruns here

  // if inh > inw

  gpointer outptr, inptr;
  outptr = outmapinfo.data;
  inptr = inmapinfo.data + (myfirstelement->inheight / 2) * myfirstelement->inwidth * 3 + (myfirstelement->inwidth / 2) * 3;
  if ((myfirstelement->inheight > 0) && (myfirstelement->inwidth > 0)) {
    for (guint idx = 0; idx < myfirstelement->outheight; idx++) {
      memcpy(outptr, inptr, myfirstelement->outwidth * 3);
      outptr += myfirstelement->outwidth * 3;
      inptr += myfirstelement->inwidth * 3;
    }
  } else {
    GST_WARNING_OBJECT(trans, "height width not set");
  }

  gst_buffer_unmap(inbuf, &inmapinfo);
  gst_buffer_unmap(outbuf, &outmapinfo);
  return GST_FLOW_OK;

 mapfail:
  GST_WARNING_OBJECT(trans, "failed to map buffers");
  return GST_FLOW_OK;
}

/* static GstFlowReturn gst_myfirstelement_transform_ip(GstBaseTransform *trans, */
/*                                                      GstBuffer *buf) { */
/*   GstMyfirstelement *myfirstelement = GST_MYFIRSTELEMENT(trans); */

/*   // g_strdup_printf("some value"); */
/*   // g_log("some message\n"); */

/*   GST_DEBUG_OBJECT(myfirstelement, "transform_ip"); */

/*   return GST_FLOW_OK; */
/* } */

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
