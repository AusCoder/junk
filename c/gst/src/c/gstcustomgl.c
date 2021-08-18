#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstcustomgl.h"
#include <gst/gst.h>
#include <gst/gl/gl.h>

GST_DEBUG_CATEGORY_STATIC(gst_customgl_debug_category);
#define GST_CAT_DEFAULT gst_customgl_debug_category

/* prototypes */

/* static void gst_customgl_set_property(GObject *object, guint
 * property_id, */
/*                                             const GValue *value, */
/*                                             GParamSpec *pspec); */
/* static void gst_customgl_get_property(GObject *object, guint
 * property_id, */
/*                                             GValue *value, GParamSpec
 * *pspec); */

static void gst_customgl_finalize(GObject *object);
static GstFlowReturn gst_customgl_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer);
static gboolean gst_customgl_sink_event(GstPad *pad, GstObject *parent, GstEvent *event);
static gboolean gst_customgl_src_event(GstPad *pad, GstObject *parent, GstEvent *event);
static gboolean gst_customgl_sink_query(GstPad *pad, GstObject *parent, GstQuery *query);
static gboolean gst_customgl_src_query(GstPad *pad, GstObject *parent, GstQuery *query);

static void gst_customgl_set_context(GstElement *element, GstContext *context);
static GstStateChangeReturn gst_customgl_change_state(GstElement *element, GstStateChange transition);

/* static gboolean gst_customgl_find_bufferpool(GstCustomgl *customgl, GstCaps *outcaps); */

enum { PROP_0 };

/* pad templates */

static GstStaticPadTemplate gst_customgl_src_template =
    GST_STATIC_PAD_TEMPLATE(
        "src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("text"));

static GstStaticPadTemplate gst_customgl_sink_template =
    GST_STATIC_PAD_TEMPLATE(
        "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
            "video/x-raw(memory:GLMemory)"
            ",format=RGBA,height=[1,2147483647],width=[1,2147483647],texture-target=2D"));

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(
    GstCustomgl, gst_customgl, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_customgl_debug_category, "customgl",
                            0, "debug category for customgl element"));

static void gst_customgl_class_init(GstCustomglClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_customgl_src_template);
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_customgl_sink_template);

  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "FIXME Long name", "Generic",
      "FIXME Description", "FIXME <fixme@example.com>");

  // gobject_class->set_property = gst_customgl_set_property;
  // gobject_class->get_property = gst_customgl_get_property;

  gobject_class->finalize = gst_customgl_finalize;
  element_class->change_state = gst_customgl_change_state;
  element_class->set_context = gst_customgl_set_context;
}

static void gst_customgl_init(GstCustomgl *customgl) {
  customgl->srcpad = gst_pad_new_from_static_template(&gst_customgl_src_template, "srcpad");
  customgl->sinkpad = gst_pad_new_from_static_template(&gst_customgl_sink_template, "sinkpad");
  // Q: Do I need to free or remove this sometime?
  gst_element_add_pad(GST_ELEMENT(customgl), customgl->srcpad);
  gst_element_add_pad(GST_ELEMENT(customgl), customgl->sinkpad);
  gst_pad_set_event_function(customgl->sinkpad, gst_customgl_sink_event);
  gst_pad_set_event_function(customgl->srcpad, gst_customgl_src_event);
  gst_pad_set_chain_function(customgl->sinkpad, gst_customgl_chain);
  gst_pad_set_query_function(customgl->sinkpad, gst_customgl_sink_query);
  gst_pad_set_query_function(customgl->srcpad, gst_customgl_src_query);

  customgl->ininfo = gst_video_info_new();

  customgl->display = NULL;
  customgl->context = NULL;
  g_rec_mutex_init(&customgl->mutex);
}

/* void gst_customgl_set_property(GObject *object, guint property_id, */
/*                                      const GValue *value, GParamSpec *pspec)
 * { */
/*   GstCustomgl *customgl = GST_CUSTOMGL(object); */

/*   GST_DEBUG_OBJECT(customgl, "set_property"); */

/*   switch (property_id) { */
/*   default: */
/*     G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec); */
/*     break; */
/*   } */
/* } */

/* void gst_customgl_get_property(GObject *object, guint property_id, */
/*                                      GValue *value, GParamSpec *pspec) { */
/*   GstCustomgl *customgl = GST_CUSTOMGL(object); */

/*   GST_DEBUG_OBJECT(customgl, "get_property"); */

/*   switch (property_id) { */
/*   default: */
/*     G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec); */
/*     break; */
/*   } */
/* } */

void gst_customgl_finalize(GObject *object) {
  GstCustomgl *customgl = GST_CUSTOMGL(object);

  GST_DEBUG_OBJECT(customgl, "finalize");

  gst_video_info_free(customgl->ininfo);
  g_rec_mutex_clear(&customgl->mutex);

  G_OBJECT_CLASS(gst_customgl_parent_class)->finalize(object);
}

static void gst_customgl_set_context(GstElement *element, GstContext *context) {
  GstCustomgl *customgl = GST_CUSTOMGL(element);
  GstGLContext *other_context = NULL;
  GstGLDisplay *display = NULL;

  GST_DEBUG_OBJECT(customgl, "received context %" GST_PTR_FORMAT, context);

  gst_gl_handle_set_context (element, context, &display, &other_context);
  g_rec_mutex_lock(&customgl->mutex);

  if (customgl->other_context) {
    gst_object_unref(customgl->other_context);
  }
  customgl->other_context = other_context;
  if (customgl->display) {
    gst_object_unref(customgl->display);
  }
  customgl->display = display;

  /* if (gl_sink->display) */
    /* gst_gl_display_filter_gl_api (gl_sink->display, SUPPORTED_GL_APIS); */
  g_rec_mutex_unlock(&customgl->mutex);

  GST_ELEMENT_CLASS (gst_customgl_parent_class)->set_context (element, context);
}

static gboolean _ensure_gl_context(GstCustomgl *customgl) {
  GstGLContext *context = NULL;
  gboolean ret =
      gst_gl_query_local_gl_context (GST_ELEMENT (customgl), GST_PAD_SRC,
      &context);
  g_rec_mutex_lock(&customgl->mutex);
  if (ret) {
    GST_DEBUG_OBJECT(customgl, "local gl context query gave context %" GST_PTR_FORMAT, context);
    if (customgl->display == context->display) {
      customgl->context = context;
    }
  }
  g_rec_mutex_unlock(&customgl->mutex);
  return TRUE;
}

static GstStateChangeReturn gst_customgl_change_state(GstElement *element, GstStateChange transition) {
  GstCustomgl *customgl = GST_CUSTOMGL(element);
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;

  GST_DEBUG_OBJECT(
      customgl, "changing state: %s => %s",
      gst_element_state_get_name(GST_STATE_TRANSITION_CURRENT(transition)),
      gst_element_state_get_name(GST_STATE_TRANSITION_NEXT(transition)));

  switch (transition) {
  case GST_STATE_CHANGE_NULL_TO_READY:
    // do the gst_gl search
    if (!gst_gl_ensure_element_data (customgl, &customgl->display,
              &customgl->other_context))
        return GST_STATE_CHANGE_FAILURE;
    // This needs to do more I think, like create the context or query upstream
    _ensure_gl_context(customgl);
    break;
  default:
    break;
  }

  ret = GST_ELEMENT_CLASS(gst_customgl_parent_class)->change_state(element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    return ret;
  }

  switch (transition) {
  case GST_STATE_CHANGE_READY_TO_NULL:
    // clear gl state
    break;
  default:
    break;
  }

  return ret;
}

static GstFlowReturn gst_customgl_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer) {
  GstCustomgl *customgl = GST_CUSTOMGL(parent);
  gst_buffer_unref(buffer);

  // call add_thread somewhere here

  GstBuffer *outbuf = gst_buffer_new_allocate(NULL, 10, NULL);
  GstMapInfo outmapinfo;
  if (!gst_buffer_map(outbuf, &outmapinfo, GST_MAP_WRITE)) {
    GST_DEBUG_OBJECT(customgl, "failed to map out buffer");
    gst_buffer_unref(outbuf);
    return GST_FLOW_ERROR;
  }
  memcpy(outmapinfo.data, "yoyoyo", 6);
  gst_buffer_unmap(outbuf, &outmapinfo);
  return gst_pad_push(customgl->srcpad, outbuf);
}

static gboolean gst_customgl_sink_event(GstPad *pad, GstObject *parent, GstEvent *event) {
  GstCustomgl *customgl = GST_CUSTOMGL(parent);
  gboolean ret;

  GST_DEBUG_OBJECT(customgl, "received event on sinkpad %" GST_PTR_FORMAT, event);

  switch (GST_EVENT_TYPE(event)) {
  case GST_EVENT_CAPS:
    GstCaps *caps;
    gst_event_parse_caps(event, &caps);
    if (!gst_video_info_from_caps(customgl->ininfo, caps)) {
      GST_WARNING_OBJECT(customgl, "failed to parse video info from caps");
    }
    gst_event_unref(event);

    GstCaps *outcaps = gst_static_caps_get(&gst_customgl_src_template.static_caps);
    GstEvent *newevent = gst_event_new_caps(outcaps);
    GST_DEBUG_OBJECT(customgl, "pushing event %" GST_PTR_FORMAT, newevent);
    ret = gst_pad_push_event(customgl->srcpad, newevent);
    gst_caps_unref(outcaps);
    break;
  default:
    ret = gst_pad_event_default(pad, parent, event);
    break;
  }

  return ret;
}

static gboolean gst_customgl_src_event(GstPad *pad, GstObject *parent, GstEvent *event) {
  GstCustomgl *customgl = GST_CUSTOMGL(parent);
  GST_DEBUG_OBJECT(customgl, "received event on srcpad %" GST_PTR_FORMAT, event);
  return gst_pad_event_default(pad, parent, event);
}

/* static gboolean gst_customgl_find_bufferpool(GstCustomgl *customgl, GstCaps *outcaps) { */
/*   _ensure_gl_context(customgl); */

/*   GstQuery *query = gst_query_new_allocation(outcaps, TRUE); */
/*   if (!gst_pad_peer_query(customgl->srcpad, query)) { */
/*     GST_DEBUG_OBJECT(customgl, "allocation query failed"); */
/*   } else { */
/*     // Question: is query an inout param in get_pad_peer_query? */
/*     GST_DEBUG_OBJECT(customgl, "got query response %" GST_PTR_FORMAT, query); */
/*   } */
/*   gst_query_unref(query); */
/*   return TRUE; */
/* } */

// TODO: we might need to handle context queries here using
static gboolean gst_customgl_sink_query(GstPad *pad, GstObject *parent, GstQuery *query) {
  GstCustomgl *customgl = GST_CUSTOMGL(parent);
  gboolean ret = FALSE;
  GST_DEBUG_OBJECT(customgl, "received query on sinkpad %" GST_PTR_FORMAT, query);

  switch (GST_QUERY_TYPE(query)) {
  case GST_QUERY_CONTEXT:
    GstGLDisplay *display = NULL;
    GstGLContext *context = NULL;
    GstGLContext *other_context = NULL;

    g_rec_mutex_lock(&customgl->mutex);
    if (customgl->display)
      display = gst_object_ref(customgl->display);
    if (customgl->context)
      context = gst_object_ref(customgl->context);
    if (customgl->other_context)
      other_context = gst_object_ref(customgl->other_context);
    g_rec_mutex_unlock(&customgl->mutex);

    ret = gst_gl_handle_context_query(GST_ELEMENT_CAST(customgl), query, display, context, other_context);

    if (display)
      gst_object_unref(display);
    if (context)
      gst_object_unref(context);
    if (other_context)
      gst_object_unref(other_context);

    break;
  default:
    ret = gst_pad_query_default(pad, parent, query);
    break;
  }
  return ret;
}

static gboolean gst_customgl_src_query(GstPad *pad, GstObject *parent, GstQuery *query) {
  GstCustomgl *customgl = GST_CUSTOMGL(parent);
  gboolean ret;
  GST_DEBUG_OBJECT(customgl, "received query on srcpad %" GST_PTR_FORMAT, query);
  switch (GST_QUERY_TYPE(query)) {
  /* case GST_QUERY_CAPS: */
  /*   gst_query_set_caps_result(query, caps); */
  /*   ret = TRUE; */
  /*   break; */
  default:
    ret = gst_pad_query_default(pad, parent, query);
    break;
  }
  return ret;
}

static gboolean plugin_init(GstPlugin *plugin) {

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register(plugin, "customgl", GST_RANK_NONE,
                              GST_TYPE_CUSTOMGL);
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

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, customgl,
                  "FIXME plugin description", plugin_init, VERSION, "LGPL",
                  PACKAGE_NAME, GST_PACKAGE_ORIGIN)
