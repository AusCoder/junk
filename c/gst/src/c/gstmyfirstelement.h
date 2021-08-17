#ifndef _GST_MYFIRSTELEMENT_H_
#define _GST_MYFIRSTELEMENT_H_

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_MYFIRSTELEMENT   (gst_myfirstelement_get_type())
#define GST_MYFIRSTELEMENT(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_MYFIRSTELEMENT,GstMyfirstelement))
#define GST_MYFIRSTELEMENT_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_MYFIRSTELEMENT,GstMyfirstelementClass))
#define GST_IS_MYFIRSTELEMENT(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_MYFIRSTELEMENT))
#define GST_IS_MYFIRSTELEMENT_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_MYFIRSTELEMENT))

typedef struct _GstMyfirstelement GstMyfirstelement;
typedef struct _GstMyfirstelementClass GstMyfirstelementClass;

struct _GstMyfirstelement
{
  GstBaseTransform base_myfirstelement;

  GstVideoInfo *ininfo;
  GstVideoInfo *outinfo;
};

struct _GstMyfirstelementClass
{
  GstBaseTransformClass base_myfirstelement_class;
};

GType gst_myfirstelement_get_type (void);

G_END_DECLS

#endif
