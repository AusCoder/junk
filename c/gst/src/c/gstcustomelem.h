#ifndef _GST_CUSTOMELEM_H_
#define _GST_CUSTOMELEM_H_

#include <gst/gst.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_CUSTOMELEM   (gst_customelem_get_type())
#define GST_CUSTOMELEM(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CUSTOMELEM,GstCustomelem))
#define GST_CUSTOMELEM_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CUSTOMELEM,GstCustomelemClass))
#define GST_IS_CUSTOMELEM(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CUSTOMELEM))
#define GST_IS_CUSTOMELEM_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CUSTOMELEM))

typedef struct _GstCustomelem GstCustomelem;
typedef struct _GstCustomelemClass GstCustomelemClass;

struct _GstCustomelem
{
  GstElement base_customelem;

  GstPad *srcpad;
  GstPad *sinkpad;

  GstCaps *outcaps;
  GstVideoInfo *ininfo;
  GstVideoInfo *outinfo;
};

struct _GstCustomelemClass
{
  GstElementClass base_customelem_class;
};

GType gst_customelem_get_type (void);

G_END_DECLS

#endif
