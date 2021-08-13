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
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_MYFIRSTELEMENT_H_
#define _GST_MYFIRSTELEMENT_H_

#include <gst/base/gstbasetransform.h>

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

  gint inheight;
  gint inwidth;
  gint outheight;
  gint outwidth;
};

struct _GstMyfirstelementClass
{
  GstBaseTransformClass base_myfirstelement_class;
};

GType gst_myfirstelement_get_type (void);

G_END_DECLS

#endif
