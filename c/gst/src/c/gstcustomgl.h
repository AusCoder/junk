#ifndef _GST_CUSTOMGL_H_
#define _GST_CUSTOMGL_H_

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/gl/gstgl_fwd.h>

G_BEGIN_DECLS

#define GST_TYPE_CUSTOMGL   (gst_customgl_get_type())
#define GST_CUSTOMGL(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CUSTOMGL,GstCustomgl))
#define GST_CUSTOMGL_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CUSTOMGL,GstCustomglClass))
#define GST_IS_CUSTOMGL(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CUSTOMGL))
#define GST_IS_CUSTOMGL_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CUSTOMGL))

typedef struct _GstCustomgl GstCustomgl;
typedef struct _GstCustomglClass GstCustomglClass;

struct _GstCustomgl
{
  GstElement base_customgl;

  GstPad *srcpad;
  GstPad *sinkpad;

  GstVideoInfo *ininfo;

  GstGLDisplay *display;
  GstGLContext *context;
  GstGLContext *other_context;  // application provided gl context
  GRecMutex mutex;
};

struct _GstCustomglClass
{
  GstElementClass base_customgl_class;
};

GType gst_customgl_get_type (void);

G_END_DECLS

#endif
