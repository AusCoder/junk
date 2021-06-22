/* Vidplay structure

[Read packets thread]
  -> audio queue
     read by SDL thread for audio playback
  -> video queue
     read by our video decode thread

[Video decode thread]
  decode video
  -> put on decoded video queue

[Main thread]
  runs the mainloop
  process user events
  schedule and process refresh events
  on a refresh event
    pull from video queue and present
 */
#include <assert.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>

#include "common.h"
#include "queue.h"

#define MAX_QUEUE_SIZE 50
#define SCREEN_WIDTH (640 * 2)
#define SCREEN_HEIGHT (480 * 2)
#define AUDIO_SAMPLE_SIZE 1024
#define MAX_AUDIO_FRAME_SIZE 192000

// presentation context
typedef struct {
  // video
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *texture;
  // audio
  SDL_AudioSpec *audioSpec;
  SDL_AudioDeviceID audioDeviceId;
} VPPresContext;

// video decoding context
typedef struct {
  // file format
  AVFormatContext *formatCtx;
  int videoStreamIdx;
  int audioStreamIdx;
  // video codec
  AVCodecParameters *vCodecPar;
  AVCodec *vCodec;
  AVCodecContext *vCodecCtx;
  // audio codec
  AVCodecParameters *aCodecPar;
  AVCodec *aCodec;
  AVCodecContext *aCodecCtx;
  // frames
  AVFrame *frameDecoded;
  AVFrame *frameYUV;
  struct SwsContext *swsCtx;
  /* // Audio queue */
  /* VPQueue *audioQueue; */
  /* int isRunning; */
  /* // unused */
  /* uint8_t *bufYUV; */
} VPVidContext;

// Communication
typedef struct {
  VPQueue *videoPacketQueue;
  VPQueue *audioPacketQueue;
  // TODO: add ring buffer for decoded frames
} VPCommContext;

typedef struct {
  VPVidContext *vidCtx;
  VPPresContext *presCtx;
  VPCommContext *commCtx;
  int isRunning;
} VPContext;

int presContextInit(VPPresContext *presCtx,
                    const SDL_AudioSpec *wantedAudioSpec) {
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
    LOG_SDL_ERROR("SDL_Init");
    return -1;
  }

  // Init SDL
  SDL_Window *window = SDL_CreateWindow("Window", 100, 100, SCREEN_WIDTH,
                                        SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
  if (window == NULL) {
    LOG_SDL_ERROR("SDL_CreateWindow");
    SDL_Quit();
    return -1;
  }

  SDL_Renderer *renderer = SDL_CreateRenderer(
      window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (renderer == NULL) {
    LOG_SDL_ERROR("SDL_CreateRenderer");
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }

  SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_IYUV,
                                           SDL_TEXTUREACCESS_STREAMING,
                                           SCREEN_WIDTH, SCREEN_HEIGHT);

  if (texture == NULL) {
    LOG_SDL_ERROR("SDL_CreateTexture");
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }

  SDL_AudioSpec *receivedAudioSpec = NULL;
  SDL_AudioDeviceID audioDeviceId = 0;
  if (wantedAudioSpec != NULL) {
    LOG_WARNING("creating sdl audio");
    receivedAudioSpec = (SDL_AudioSpec *)malloc(sizeof(SDL_AudioSpec));
    if (receivedAudioSpec == NULL) {
      LOG_ERROR("malloc failed");
      SDL_DestroyTexture(texture);
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(window);
      SDL_Quit();
      return -1;
    }
    audioDeviceId =
        SDL_OpenAudioDevice(NULL, 0, wantedAudioSpec, receivedAudioSpec,
                            SDL_AUDIO_ALLOW_FORMAT_CHANGE);
    if (audioDeviceId <= 0) {
      LOG_SDL_ERROR("SDL_OpenAudioDevice failed");
      free(receivedAudioSpec);
      SDL_DestroyTexture(texture);
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(window);
      SDL_Quit();
      return -1;
    }
    if (wantedAudioSpec->format != receivedAudioSpec->format) {
      // Something else?
      LOG_WARNING("didn't get expected audio format");
    }
    SDL_PauseAudioDevice(audioDeviceId, 0);
  }

  presCtx->window = window;
  presCtx->renderer = renderer;
  presCtx->texture = texture;
  // audio
  presCtx->audioSpec = receivedAudioSpec;
  presCtx->audioDeviceId = audioDeviceId;
  return 0;
}

static void presContextClose(VPPresContext *presCtx) {
  if (presCtx->audioSpec != NULL) {
    free(presCtx->audioSpec);
  }
  if (presCtx->audioDeviceId > 0) {
    SDL_CloseAudioDevice(presCtx->audioDeviceId);
  }
  SDL_DestroyTexture(presCtx->texture);
  SDL_DestroyRenderer(presCtx->renderer);
  SDL_DestroyWindow(presCtx->window);
  SDL_Quit();
}

static int vidContextOpenCodecContext(const AVCodecParameters *codecPar,
                                      AVCodec **codec,
                                      AVCodecContext **codecCtx) {
  AVCodec *c = avcodec_find_decoder(codecPar->codec_id);
  if (c == NULL) {
    LOG_ERROR("Unsupported codec");
    return -1;
  }
  AVCodecContext *cCtx = avcodec_alloc_context3(c);
  if (avcodec_parameters_to_context(cCtx, codecPar) != 0) {
    LOG_ERROR("Failed to set codec parameters");
    return -1;
  }
  if (avcodec_open2(cCtx, c, NULL) < 0) {
    LOG_ERROR("Failed to open codec context");
    avcodec_free_context(&cCtx);
    return -1;
  }
  *codec = c;
  *codecCtx = cCtx;
  return 0;
}

static void vidContextCloseCodecContext(AVCodecContext **codecCtx) {
  if (*codecCtx != NULL) {
    avcodec_close(*codecCtx);
    avcodec_free_context(codecCtx);
  }
}

static int vidContextInit(VPVidContext *vidCtx, const char *path) {
  // TODO:
  //   missing audio clean up on some failure cases
  AVFormatContext *pFormatCtx = NULL;
  if (avformat_open_input(&pFormatCtx, path, NULL, NULL) != 0) {
    return -1;
  }
  if (avformat_find_stream_info(pFormatCtx, NULL) != 0) {
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  int videoStreamIdx = -1;
  int audioStreamIdx = -1;
  for (int i = 0; i < pFormatCtx->nb_streams; i++) {
    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStreamIdx = i;
    } else if (pFormatCtx->streams[i]->codecpar->codec_type ==
               AVMEDIA_TYPE_AUDIO) {
      audioStreamIdx = i;
    }
  }
  if (videoStreamIdx < 0) {
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  // Video codec parameters
  AVCodecParameters *vCodecPar = pFormatCtx->streams[videoStreamIdx]->codecpar;
  AVCodec *vCodec = NULL;
  AVCodecContext *vCodecCtx = NULL;
  if (vidContextOpenCodecContext(vCodecPar, &vCodec, &vCodecCtx) < 0) {
    LOG_ERROR("Failed to create video codec context");
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  // Audio codec parameters
  AVCodecParameters *aCodecPar = NULL;
  AVCodec *aCodec = NULL;
  AVCodecContext *aCodecCtx = NULL;
  if (audioStreamIdx < 0) {
    LOG_WARNING("no audo stream found");
  } else {
    aCodecPar = pFormatCtx->streams[audioStreamIdx]->codecpar;
    if (vidContextOpenCodecContext(aCodecPar, &aCodec, &aCodecCtx) < 0) {
      return -1;
    }
  }
  // Getting frames
  AVFrame *pFrameDecoded = av_frame_alloc();
  if (pFrameDecoded == NULL) {
    LOG_ERROR("Failed to alloc frame");
    vidContextCloseCodecContext(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  AVFrame *pFrameYUV = av_frame_alloc();
  if (pFrameYUV == NULL) {
    LOG_ERROR("Failed to alloc frame");
    av_frame_free(&pFrameDecoded);
    vidContextCloseCodecContext(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  int numBytes = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, SCREEN_WIDTH,
                                          SCREEN_HEIGHT, 1);
  uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  if (buffer == NULL) {
    LOG_ERROR("Failed to alloc buffer");
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameYUV);
    vidContextCloseCodecContext(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  if (av_image_fill_arrays(pFrameYUV->data, pFrameYUV->linesize, buffer,
                           AV_PIX_FMT_YUV420P, SCREEN_WIDTH, SCREEN_HEIGHT,
                           1) < 0) {
    LOG_ERROR("av_image_fill_arrays failed");
    av_free(buffer);
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameYUV);
    vidContextCloseCodecContext(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  struct SwsContext *pSwsCtx = sws_getContext(
      vCodecCtx->width, vCodecCtx->height, vCodecCtx->pix_fmt, SCREEN_WIDTH,
      SCREEN_HEIGHT, AV_PIX_FMT_YUV420P, SWS_BILINEAR, NULL, NULL, NULL);
  if (pSwsCtx == NULL) {
    LOG_ERROR("sws_getContext failed");
    av_free(buffer);
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameYUV);
    vidContextCloseCodecContext(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  /* VPQueue *q = queueAlloc(); */
  /* if (q == NULL) { */
  /*   LOG_ERROR("queueAlloc failed"); */
  /*   sws_freeContext(pSwsCtx); */
  /*   av_free(buffer); */
  /*   av_frame_free(&pFrameDecoded); */
  /*   av_frame_free(&pFrameYUV); */
  /*   vidContextCloseCodecContext(&vCodecCtx); */
  /*   avformat_close_input(&pFormatCtx); */
  /*   return -1; */
  /* } */

  vidCtx->formatCtx = pFormatCtx;
  vidCtx->videoStreamIdx = videoStreamIdx;
  vidCtx->audioStreamIdx = audioStreamIdx;

  vidCtx->vCodecPar = vCodecPar;
  vidCtx->vCodec = vCodec;
  vidCtx->vCodecCtx = vCodecCtx;

  vidCtx->aCodecPar = aCodecPar;
  vidCtx->aCodec = aCodec;
  vidCtx->aCodecCtx = aCodecCtx;

  vidCtx->frameDecoded = pFrameDecoded;
  vidCtx->frameYUV = pFrameYUV;
  vidCtx->swsCtx = pSwsCtx;

  /* vidCtx->audioQueue = q; */
  /* vidCtx->isRunning = 1; */

  /* vidCtx->bufYUV = NULL; */
  return 0;
}

void vidContextClose(VPVidContext *ctx) {
  // audio
  queueFree(ctx->audioQueue);
  if (ctx->aCodecCtx != NULL) {
    vidContextCloseCodecContext(&ctx->aCodecCtx);
  }
  // video
  sws_freeContext(ctx->swsCtx);
  av_frame_free(&ctx->frameDecoded);
  av_free(ctx->frameYUV->data[0]);
  av_frame_free(&ctx->frameYUV);
  vidContextCloseCodecContext(&ctx->vCodecCtx);
  avformat_close_input(&ctx->formatCtx);
}

int commContextInit(VPCommContext *commCtx) {
  commCtx->videoPacketQueue = queueAlloc();
  if (commCtx->videoPacketQueue == NULL) {
    return VP_ERR_FATAL;
  }
  commCtx->audioPacketQueue = queueAlloc();
  if (commCtx->audioPacketQueue == NULL) {
    queueFree(commCtx->videoPacketQueue);
    return VP_ERR_FATAL;
  }
  return 0;
}

void commContextClose(VPCommContext *commCtx) {
  queueFree(commCtx->videoPacketQueue);
  queueFree(commCtx->audioPacketQueue);
}

VPContext *contextAllocAndInit(const char *path) {
  // TODO: clean up correctly if init fails
  // is goto to right way to do this?
  VPContext *ctx = NULL;
  VPVidContext *vidCtx = NULL;
  VPPresContext *presCtx = NULL;
  VPCommContext *commCtx = NULL;
  vidCtx = (VPVidContext *)malloc(sizeof(VPVidContext));
  if (vidCtx == NULL) {
    goto contextAlloc_error;
  }
  presCtx = (VPPresContext *)malloc(sizeof(VPPresContext));
  if (presCtx == NULL) {
    goto contextAlloc_error;
  }
  commCtx = (VPCommContext *)malloc(sizeof(VPCommContext));
  if (commCtx == NULL) {
    goto contextAlloc_error;
  }
  ctx = (VPContext *)malloc(sizeof(VPContext));
  if (ctx == NULL) {
    goto contextAlloc_error;
  }

  ctx->vidCtx = vidCtx;
  ctx->presCtx = presCtx;
  ctx->commCtx = commCtx;
  ctx->isRunning = 1;

  if (vidContextInit(ctx->vidCtx, path) < 0) {
    LOG_ERROR("vidContextInit");
    goto contextAlloc_error;
  }

  SDL_AudioSpec wantedAudioSpec;
  SDL_AudioSpec *wantedAudioSpecPtr = NULL;
  if (ctx->vidCtx->audioStreamIdx >= 0) {
    printf("Sample format: %s. Number hannels: %d\n",
           av_get_sample_fmt_name(ctx->vidCtx->aCodecCtx->sample_fmt),
           ctx->vidCtx->aCodecCtx->channels);

    wantedAudioSpec.freq = ctx->vidCtx->aCodecCtx->sample_rate;
    wantedAudioSpec.format = AUDIO_F32;
    wantedAudioSpec.channels = ctx->vidCtx->aCodecCtx->channels;
    wantedAudioSpec.silence = 0;
    wantedAudioSpec.samples = AUDIO_SAMPLE_SIZE;
    wantedAudioSpec.callback = audioCallback;
    wantedAudioSpec.userdata = (void *)&vidCtx;
    wantedAudioSpecPtr = &wantedAudioSpec;
  }

  if (presContextInit(ctx->presCtx, wantedAudioSpecPtr) < 0) {
    LOG_ERROR("presContextInit");
    vidContextClose(ctx->vidCtx);
    goto contextAlloc_error;
  }
  return ctx;

contextAlloc_error:
  if (commCtx != NULL)
    free(commCtx);
  if (presCtx != NULL)
    free(presCtx);
  if (vidCtx != NULL)
    free(vidCtx);
  if (ctx != NULL)
    free(ctx);
  return NULL;
}

static void contextCloseAndFree(VPContext *ctx) {
  vidContextClose(ctx->vidCtx);
  presContextClose(ctx->presCtx);
  commContextClose(ctx->commCtx);
  free(ctx->commCtx);
  free(ctx->presCtx);
  free(ctx->vidCtx);
  free(ctx);
}

// static int get_format_from_sample_fmt(const char **fmt,
//                                       enum AVSampleFormat sample_fmt) {
//   int i;
//   struct sample_fmt_entry {
//     enum AVSampleFormat sample_fmt;
//     const char *fmt_be, *fmt_le;
//   } sample_fmt_entries[] = {
//       {AV_SAMPLE_FMT_U8, "u8", "u8"},
//       {AV_SAMPLE_FMT_S16, "s16be", "s16le"},
//       {AV_SAMPLE_FMT_S32, "s32be", "s32le"},
//       {AV_SAMPLE_FMT_FLT, "f32be", "f32le"},
//       {AV_SAMPLE_FMT_DBL, "f64be", "f64le"},
//   };
//   *fmt = NULL;

//   for (i = 0; i < FF_ARRAY_ELEMS(sample_fmt_entries); i++) {
//     struct sample_fmt_entry *entry = &sample_fmt_entries[i];
//     if (sample_fmt == entry->sample_fmt) {
//       *fmt = AV_NE(entry->fmt_be, entry->fmt_le);
//       return 0;
//     }
//   }

//   fprintf(stderr, "sample format %s is not supported as output format\n",
//           av_get_sample_fmt_name(sample_fmt));
//   return -1;
// }

static int presentFrame(VPVidContext *vidCtx, VPPresContext *presCtx) {
  sws_scale(vidCtx->swsCtx, (uint8_t const *const *)vidCtx->frameDecoded->data,
            vidCtx->frameDecoded->linesize, 0, vidCtx->vCodecCtx->height,
            vidCtx->frameYUV->data, vidCtx->frameYUV->linesize);

  assert(vidCtx->frameYUV->linesize[0] == SCREEN_WIDTH);
  assert(SCREEN_WIDTH * SDL_BYTESPERPIXEL(SDL_PIXELFORMAT_IYUV) ==
         SCREEN_WIDTH);

  // YUV - Y has 1 byte per pixel
  assert(vidCtx->frameYUV->data[1] ==
         vidCtx->frameYUV->data[0] + SCREEN_WIDTH * SCREEN_HEIGHT * 1);
  assert(vidCtx->frameYUV->data[2] ==
         vidCtx->frameYUV->data[1] + SCREEN_WIDTH * SCREEN_HEIGHT * 1 / 2 / 2);

  // SDL_LockTexture version
  uint8_t *pixels;
  int pitch;
  if (SDL_LockTexture(presCtx->texture, NULL, (void **)&pixels, &pitch) < 0) {
    LOG_SDL_ERROR("SDL_LockTexture failed");
    return -1;
  }
  for (int plane = 0; plane < 3; plane++) {
    int widthBytes = (plane == 0 ? SCREEN_WIDTH : SCREEN_WIDTH / 2) *
                     SDL_BYTESPERPIXEL(SDL_PIXELFORMAT_IYUV);
    int height = plane == 0 ? SCREEN_HEIGHT : SCREEN_HEIGHT / 2;
    int avFrameOffset = 0;
    for (int y = 0; y < height; y++) {
      memcpy(pixels, vidCtx->frameYUV->data[plane] + avFrameOffset, widthBytes);
      avFrameOffset += widthBytes;
      pixels += widthBytes;
    }
  }
  SDL_UnlockTexture(presCtx->texture);

  SDL_RenderCopy(presCtx->renderer, presCtx->texture, NULL, NULL);
  SDL_RenderPresent(presCtx->renderer);
  SDL_Delay(100);
  return 0;
}

static int packetThreadTarget(VPContext *ctx) {
  /* Problem: audio and video frames come interleaved.
     Currently we are decoding video, presenting, waiting and pushing
     audio to a separate queue. This means audio buffering to the queue
     is blocked by video presenting to the screen and the frame delay.

     Instead, we want to queue both frames and audio and present them
     at the same time.
   */
  AVPacket packet;
  for (;;) {
    if (!ctx->isRunning) {
      break;
    }

    if (ctx->commCtx->videoPacketQueue.size > MAX_QUEUE_SIZE ||
        ctx->commCtx->audioPacketQueue.size > MAX_QUEUE_SIZE) {
      SDL_Delay(10);
      continue;
    }

    if (av_read_frame(ctx->vidCtx->formatCtx, &packet) < 0) {
      if (ctx->vidCtx->formatCtx->pb->error == 0) {
        // No error, wait for user input
        SDL_Delay(100);
        continue;
      } else {
        break;
      }
    }
    if (packet.stream_index == ctx->vidCtx->videoStreamIdx ||
        packet.stream_index == ctx->vidCtx->audioStreamIdx) {
      AVPacket *packetClone = av_packet_clone(&packet);
      if (packetClone == NULL) {
        LOG_ERROR("av_packet_clone failed\n");
        return -1;
      }
      VPQueue *q = packet.stream_index == ctx->vidCtx->videoStreamIdx ?
        ctx->commCtx->videoPacketQueue :
        ctx->commCtx->audioPacketQueue;
      if (queuePut(q, packetClone) < 0) {
        LOG_ERROR("queuePut failed\n");
        av_packet_unref(packetClone);
        return -1;
      }
    }
    av_packet_unref(&packet);
  }
  return 0;
}

  /* // old */
  /* SDL_Event event; */
  /* int frameIdx = 0; */
  /* while (av_read_frame(vidCtx->formatCtx, &packet) >= 0) { */
  /*   // Handle events */
  /*   SDL_PollEvent(&event); */
  /*   switch (event.type) { */
  /*   case SDL_QUIT: */
  /*     vidCtx->isRunning = 0; */
  /*     av_packet_unref(&packet); */
  /*     return 0; */
  /*     break; */
  /*   default: */
  /*     break; */
  /*   } */

  /*   // Check packet from video stream */
  /*   if (packet.stream_index == vidCtx->videoStreamIdx) { */
  /*     // Try to send packet for decoding */
  /*     int sendRet = avcodec_send_packet(vidCtx->vCodecCtx, &packet); */
  /*     if (sendRet == AVERROR(EAGAIN)) { */
  /*       // try receiving frames */
  /*     } else if (sendRet < 0) { */
  /*       LOG_ERROR("avcodec_send_packet failed\n"); */
  /*       return -1; */
  /*     } */

  /*     // Try to read a frame from decoder */
  /*     for (;;) { */
  /*       int recvRet = */
  /*           avcodec_receive_frame(vidCtx->vCodecCtx, vidCtx->frameDecoded); */
  /*       if (recvRet == AVERROR(EAGAIN)) { */
  /*         // Can't receive a frame, need to try to send again */
  /*         break; */
  /*       } else if (recvRet < 0) { */
  /*         LOG_ERROR("avcodec_receive_frame failed\n"); */
  /*         return -1; */
  /*       } else { */
  /*         // Got a frame */
  /*         frameIdx++; */
  /*         if (presentFrame(vidCtx, presCtx) < 0) { */
  /*           return -1; */
  /*         } */
  /*       } */
  /*     } */
  /*   } else if (packet.stream_index == vidCtx->audioStreamIdx) { */
  /*     AVPacket *packetClone = av_packet_clone(&packet); */
  /*     if (packetClone == NULL) { */
  /*       LOG_ERROR("av_packet_clone failed\n"); */
  /*       return -1; */
  /*     } */
  /*     if (queuePut(vidCtx->audioQueue, (const void *)packetClone) < 0) { */
  /*       LOG_ERROR("queuePut failed\n"); */
  /*       return -1; */
  /*     } */
  /*   } */
  /*   av_packet_unref(&packet); */
  /* } */
  /* printf("Frames seen %d\n", frameIdx); */
  /* return 0; */
/* } */

static int audioDecodeFrame(VPVidContext *vidCtx, uint8_t *audioBuf,
                            int audioBufSize) {
  // static AVPacket pkt;
  AVPacket *pkt;
  static AVFrame frame;
  for (;;) {
    if (!vidCtx->isRunning) {
      return -1;
    }
    if (queueGet(vidCtx->audioQueue, (const void **)&pkt) < 0) {
      LOG_ERROR("queueGet failed\n");
      return -1;
    }
    int sendRet = avcodec_send_packet(vidCtx->aCodecCtx, pkt);
    if ((sendRet < 0) && (sendRet != AVERROR(EAGAIN))) {
      LOG_ERROR("avcodec_send_packet failed\n");
      av_packet_free(&pkt);
      return -1;
    }

    for (;;) {
      int recvRet = avcodec_receive_frame(vidCtx->aCodecCtx, &frame);
      if (recvRet == AVERROR(EAGAIN)) {
        break;
      } else if (recvRet < 0) {
        LOG_ERROR("avcodec_receive_frame failed\n");
        av_packet_free(&pkt);
        return -1;
      } else {
        // Got a frame

        int frameBufSize = av_samples_get_buffer_size(
            NULL, vidCtx->aCodecCtx->channels, frame.nb_samples,
            vidCtx->aCodecCtx->sample_fmt, 1);
        assert(frameBufSize <= audioBufSize);

        // planar audio format
        // TODO: add a check using av_sample_fmt_is_planar
        assert(av_sample_fmt_is_planar(vidCtx->aCodecCtx->sample_fmt));
        assert(vidCtx->aCodecCtx->channels == 2);

        int bsPerSample =
            av_get_bytes_per_sample(vidCtx->aCodecCtx->sample_fmt);
        assert(bsPerSample == 4);
        /* printf("bsPerSample: %d\n", bsPerSample); */
        for (int sIdx = 0; sIdx < frame.nb_samples; sIdx++) {
          for (int cIdx = 0; cIdx < vidCtx->aCodecCtx->channels; cIdx++) {
            memcpy(audioBuf, frame.data[cIdx] + sIdx * bsPerSample,
                   bsPerSample);
            audioBuf += bsPerSample;
          }
        }

        // printf("frameBufSize: %d\n", frameBufSize);
        // memcpy(audioBuf, frame.data[0], frameBufSize);
        av_packet_free(&pkt);
        return frameBufSize;
      }
    }
    av_packet_free(&pkt);
  }
}

// typedef void (SDLCALL * SDL_AudioCallback) (void *userdata, Uint8 * stream,
//                                             int len);

static void audioCallback(void *userdata, Uint8 *stream, int len) {
  // TODO: need to think about the vidCtx->isRunning check
  // we can probably stuck in the queueGet blocking call in the audio thread
  VPVidContext *vidCtx = (VPVidContext *)userdata;

  static uint8_t audio_buf[(MAX_AUDIO_FRAME_SIZE * 3) / 2];
  unsigned int audio_buf_size = 0;

  while (len > 0) {
    // get frame
    int audio_dec_ret = audioDecodeFrame(vidCtx, audio_buf, sizeof(audio_buf));
    if (audio_dec_ret < 0) {
      audio_buf_size = 1024;
      memset(audio_buf, 0, audio_buf_size);
    } else {
      audio_buf_size = audio_dec_ret;
    }
    // copy frame to SDL audio stream buffer
    int len1 = audio_buf_size <= len ? audio_buf_size : len;
    memcpy(stream, audio_buf, len1);
    len -= len1;
    stream += len1;
  }
}

int main(int argc, char *argv[]) {
  VPContext *ctx = contextAllocAndInit(argv[1]);
  if (playVideo(ctx) < 0) {
    LOG_ERROR("playVideo failed");
  }
  contextCloseAndFree(ctx);
}
