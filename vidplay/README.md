# Vidplay

Tutorial from [here](http://dranger.com/ffmpeg/).

## Questions
* SDL texture vs surface, what's the difference?
* Is AVFrame data always continuous? Even across image rows and planes?
    Do I need to copy to a continuous buffer before calling SDL_UpdateTexture?
* What does SDL_TEXTUREACCESS_ do?
* Should I call `avcodec_send_packet` and `avcodec_receive_frame` in subloops?

## Dev
```shell
# build
gcc -lavcodec -lavformat -lavutil -lswscale -o tut1 tut1.c
# with sdl dependency
gcc -Wall -lSDL2 -lavformat -lswscale -lavcodec -lavutil -o tut2 tut2.c
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

## Dependencies
For reference, I built these samples on Arch linux with
```txt
ffmpeg version n4.4 Copyright (c) 2000-2021 the FFmpeg developers
built with gcc 10.2.0 (GCC)
configuration: --prefix=/usr --disable-debug --disable-static --disable-stripping --enable-amf --enable-avisynth --enable-cuda-llvm --enable-lto --enable-fontconfig --enable-gmp --enable-gnutls --enable-gpl --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libdav1d --enable-libdrm --enable-libfreetype --enable-libfribidi --enable-libgsm --enable-libiec61883 --enable-libjack --enable-libmfx --enable-libmodplug --enable-libmp3lame --enable-libopencore_amrnb --enable-libopencore_amrwb --enable-libopenjpeg --enable-libopus --enable-libpulse --enable-librav1e --enable-librsvg --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libtheora --enable-libv4l2 --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxcb --enable-libxml2 --enable-libxvid --enable-libzimg --enable-nvdec --enable-nvenc --enable-shared --enable-version3
libavutil      56. 70.100 / 56. 70.100
libavcodec     58.134.100 / 58.134.100
libavformat    58. 76.100 / 58. 76.100
libavdevice    58. 13.100 / 58. 13.100
libavfilter     7.110.100 /  7.110.100
libswscale      5.  9.100 /  5.  9.100
libswresample   3.  9.100 /  3.  9.100
libpostproc    55.  9.100 / 55.  9.100
```

TODO: add sdl
