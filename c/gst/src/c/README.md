## Caps
### Sticky events
`gst_pad_get_current_caps` gets the caps from the last GST_EVENT_CAPS that happened on the pad.
See this from the docs:
* Some events will be sticky on the pad, meaning that after they pass
  on the pad they can be queried later with gst_pad_get_sticky_event
  and gst_pad_sticky_events_foreach. gst_pad_get_current_caps and
  gst_pad_has_current_caps are convenience functions to query the
  current sticky CAPS event on a pad.

## BaseTransform
### set_caps
The point of `set_caps` is so that the subclass instance can know
which caps were chosen. It means we can do stuff like
`trans->height = gst_structure_get_int(structure, "height")`
### transform_caps
How does this transform element change caps? It has to be
implemented in both directions (not sure I understand this bit).
### transform
Actually do something to the outbuf. If we implement `transform`
and `get_unit_size`, new output buffers will be allocated of
`get_unit_size` and passed to `transform`.

## Glib
### Reference
[Glib api docs](https://docs.gtk.org/glib/index.html)
[GObject api docs](https://docs.gtk.org/gobject/)
### Debugging
Set to kill the program on a warning or higher message `G_DEBUG=fatal_warnings`

## Gst
### Debugging
Set element specific log levels with `export
GST_DEBUG=myfirstelement:6,basetransform:6`.

This is not restricted to element names, you can also use things like
`export GST_DEBUG=GST_CONTEXT:4` which turns on logging for statements
like:

```c
GST_CAT_INFO_OBJECT (GST_CAT_CONTEXT, element,
    "found context (%p) in downstream query", ctxt)
```
