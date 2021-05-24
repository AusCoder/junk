# Vidplay

Tutorial from [here](http://dranger.com/ffmpeg/).

## Dev
```shell
# build
gcc -o tut1 -lavcodec -lavformat -lavutil -lswscale tut1.c
# Run
./tut1 ./simple.mp4
# Memcheck
valgrind --leak-check=full ./tut1 ./simple.mp4
```

## FFmpeg notes
### Data pointers and linesize
`avFrame->data` contains data pointers for each image plane.
`avFrame->linesize` contains the stride for each image plane.

For YUV, we would have `avFrame->data[0]` points to the start of the Y plane.
`avFrame->data[1]` points to the start of the U plane and `avFrame->data[2]` points to start of V plane.
Linesize is the stride for jumping to the next line of the image. Eg for a 480x640 YUV image,
`avFrame->linesize[0] == 640`. NB: the YUV sample will matter here too, image rows for some of U and V
are smaller for a lower sampling rate.

For RGB image, there is a single image plane with pixels stored sequentially `R G B R G B R G B ...`.
`avFrame->data[0]` points to the start of the image.
For a 480x640 image, `avFrame->linesize[0]` will be `640 * 3`.

See [linesize example](https://stackoverflow.com/questions/13286022/can-anyone-help-in-understanding-avframe-linesize).
See [linesize alignment](https://stackoverflow.com/questions/35678041/what-is-linesize-alignment-meaning).
### Reference counting
See [here](https://stackoverflow.com/questions/49449411/how-to-use-av-frame-unref-in-ffmpeg).
