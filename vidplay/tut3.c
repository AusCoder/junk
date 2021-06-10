/* Plays sound!

   Probems with this program:
   - audio queue get can block at the end
     when we haven't updated the isRunning boolean
   - We should decouple reading the file from presenting
     frames.
 */
#include <assert.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>

#include "queue.h"

#define SCREEN_WIDTH (640 * 2)
#define SCREEN_HEIGHT (480 * 2)
#define AUDIO_SAMPLE_SIZE 1024
#define MAX_AUDIO_FRAME_SIZE 192000

#define LOG_ERROR(msg) fprintf(stderr, "Error: %s\n", (msg))

#define LOG_WARNING(msg) fprintf(stderr, "Warning: %s\n", (msg))

#define LOG_SDL_ERROR(msg)                                                     \
  fprintf(stderr, "%s. SDL error: %s\n", (msg), SDL_GetError())

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
  // Audio queue
  VPQueue *audioQueue;
  int isRunning;
  // unused
  uint8_t *bufYUV;
} VPVidContext;

int pres_context_init(VPPresContext *presCtx, SDL_AudioSpec *wantedAudioSpec) {
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

static void pres_context_close(VPPresContext *presCtx) {
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

static int vid_context_open_codec_context(const AVCodecParameters *codecPar,
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

static void vid_context_close_codec_context(AVCodecContext **codecCtx) {
  if (*codecCtx != NULL) {
    avcodec_close(*codecCtx);
    avcodec_free_context(codecCtx);
  }
}

static int vid_context_init(VPVidContext *vidCtx, const char *path) {
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
  if (vid_context_open_codec_context(vCodecPar, &vCodec, &vCodecCtx) < 0) {
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
    if (vid_context_open_codec_context(aCodecPar, &aCodec, &aCodecCtx) < 0) {
      return -1;
    }
  }
  // Getting frames
  AVFrame *pFrameDecoded = av_frame_alloc();
  if (pFrameDecoded == NULL) {
    LOG_ERROR("Failed to alloc frame");
    vid_context_close_codec_context(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  AVFrame *pFrameYUV = av_frame_alloc();
  if (pFrameYUV == NULL) {
    LOG_ERROR("Failed to alloc frame");
    av_frame_free(&pFrameDecoded);
    vid_context_close_codec_context(&vCodecCtx);
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
    vid_context_close_codec_context(&vCodecCtx);
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
    vid_context_close_codec_context(&vCodecCtx);
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
    vid_context_close_codec_context(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  VPQueue *q = queue_alloc();
  if (q == NULL) {
    LOG_ERROR("queue_alloc failed");
    sws_freeContext(pSwsCtx);
    av_free(buffer);
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameYUV);
    vid_context_close_codec_context(&vCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

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

  vidCtx->audioQueue = q;
  vidCtx->isRunning = 1;

  vidCtx->bufYUV = NULL;
  return 0;
}

void vid_context_close(VPVidContext *ctx) {
  // audio
  queue_free(ctx->audioQueue);
  if (ctx->aCodecCtx != NULL) {
    vid_context_close_codec_context(&ctx->aCodecCtx);
  }
  // video
  sws_freeContext(ctx->swsCtx);
  av_frame_free(&ctx->frameDecoded);
  av_free(ctx->frameYUV->data[0]);
  av_frame_free(&ctx->frameYUV);
  vid_context_close_codec_context(&ctx->vCodecCtx);
  avformat_close_input(&ctx->formatCtx);
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

static int present_frame(VPVidContext *vidCtx, VPPresContext *presCtx) {
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

static int playVideo(VPVidContext *vidCtx, VPPresContext *presCtx) {
  /* Problem: audio and video frames come interleaved.
     Currently we are decoding video, presenting, waiting and pushing
     audio to a separate queue. This means audio buffering to the queue
     is blocked by video presenting to the screen and the frame delay.

     Instead, we want to queue both frames and audio and present them
     at the same time.
   */
  AVPacket packet;
  SDL_Event event;
  int frameIdx = 0;
  while (av_read_frame(vidCtx->formatCtx, &packet) >= 0) {
    // Handle events
    SDL_PollEvent(&event);
    switch (event.type) {
    case SDL_QUIT:
      vidCtx->isRunning = 0;
      av_packet_unref(&packet);
      return 0;
      break;
    default:
      break;
    }

    // Check packet from video stream
    if (packet.stream_index == vidCtx->videoStreamIdx) {
      // Try to send packet for decoding
      int sendRet = avcodec_send_packet(vidCtx->vCodecCtx, &packet);
      if (sendRet == AVERROR(EAGAIN)) {
        // try receiving frames
      } else if (sendRet < 0) {
        LOG_ERROR("avcodec_send_packet failed\n");
        return -1;
      }

      // Try to read a frame from decoder
      for (;;) {
        int recvRet =
            avcodec_receive_frame(vidCtx->vCodecCtx, vidCtx->frameDecoded);
        if (recvRet == AVERROR(EAGAIN)) {
          // Can't receive a frame, need to try to send again
          break;
        } else if (recvRet < 0) {
          LOG_ERROR("avcodec_receive_frame failed\n");
          return -1;
        } else {
          // Got a frame
          frameIdx++;
          if (present_frame(vidCtx, presCtx) < 0) {
            return -1;
          }
        }
      }
    } else if (packet.stream_index == vidCtx->audioStreamIdx) {
      AVPacket *packetClone = av_packet_clone(&packet);
      if (packetClone == NULL) {
        LOG_ERROR("av_packet_clone failed\n");
        return -1;
      }
      if (queue_put(vidCtx->audioQueue, (const void *)packetClone) < 0) {
        LOG_ERROR("queue_put failed\n");
        return -1;
      }
    }
    av_packet_unref(&packet);
  }
  printf("Frames seen %d\n", frameIdx);
  return 0;
}

static int audioDecodeFrame(VPVidContext *vidCtx, uint8_t *audioBuf,
                            int audioBufSize) {
  // static AVPacket pkt;
  AVPacket *pkt;
  static AVFrame frame;
  for (;;) {
    if (!vidCtx->isRunning) {
      return -1;
    }
    if (queue_get(vidCtx->audioQueue, (const void **)&pkt) < 0) {
      LOG_ERROR("queue_get failed\n");
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
  // we can probably stuck in the queue_get blocking call in the audio thread
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
  VPVidContext vidCtx;
  if (vid_context_init(&vidCtx, argv[1]) < 0) {
    LOG_ERROR("vid_context_init failed");
    return -1;
  }

  SDL_AudioSpec wantedAudioSpec;
  SDL_AudioSpec *p_wantedAudioSpec = NULL;
  if (vidCtx.audioStreamIdx >= 0) {
    printf("Sample format: %s. Number hannels: %d\n",
           av_get_sample_fmt_name(vidCtx.aCodecCtx->sample_fmt),
           vidCtx.aCodecCtx->channels);

    wantedAudioSpec.freq = vidCtx.aCodecCtx->sample_rate;
    wantedAudioSpec.format = AUDIO_F32;
    wantedAudioSpec.channels = vidCtx.aCodecCtx->channels;
    wantedAudioSpec.silence = 0;
    wantedAudioSpec.samples = AUDIO_SAMPLE_SIZE;
    wantedAudioSpec.callback = audioCallback;
    wantedAudioSpec.userdata = (void *)&vidCtx;
    p_wantedAudioSpec = &wantedAudioSpec;
  }
  VPPresContext presCtx;
  if (pres_context_init(&presCtx, p_wantedAudioSpec) < 0) {
    LOG_ERROR("pres_context_init failed");
    return -1;
  }

  if (playVideo(&vidCtx, &presCtx) < 0) {
    LOG_ERROR("playVideo failed");
  }

  vid_context_close(&vidCtx);
  pres_context_close(&presCtx);
  printf("success\n");
}
