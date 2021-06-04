/* Plays sound!
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

#define LOG_ERROR(msg) fprintf(stderr, "Error: %s\n", (msg))

#define LOG_WARNING(msg) fprintf(stderr, "Warning: %s\n", (msg))

#define LOG_SDL_ERROR(msg)                                                     \
  fprintf(stderr, "%s. SDL error: %s\n", (msg), SDL_GetError())

// presentation context
typedef struct {
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *texture;
  SDL_AudioSpec *audioSpec;
} VPPresContext;

// video decoding context
typedef struct {
  AVFormatContext *formatCtx;
  int videoStreamIdx;
  int audioStreamIdx;
  // video
  AVCodecParameters *vCodecPar;
  AVCodec *vCodec;
  AVCodecContext *vCodecCtx;
  // auto
  AVCodecParameters *aCodecPar;
  AVCodec *aCodec;
  AVCodecContext *aCodecCtx;
  // frames
  AVFrame *frameDecoded;
  AVFrame *frameYUV;
  struct SwsContext *swsCtx;
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

  SDL_AudioSpec *spec = NULL;
  if (wantedAudioSpec != NULL) {
    LOG_WARNING("creating sdl audio");
    spec = (SDL_AudioSpec *)malloc(sizeof(SDL_AudioSpec));
    if (spec == NULL) {
      LOG_ERROR("malloc failed");
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(window);
      SDL_Quit();
      return -1;
    }
    if (SDL_OpenAudio(wantedAudioSpec, spec) < 0) {
      LOG_SDL_ERROR("SDL_OpenAudio failed");
      free(spec);
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(window);
      SDL_Quit();
      return -1;
    }
  }

  presCtx->window = window;
  presCtx->renderer = renderer;
  presCtx->texture = texture;
  presCtx->audioSpec = spec;
  return 0;
}

static void pres_context_close(VPPresContext *presCtx) {
  free(presCtx->audioSpec);
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
  vidCtx->formatCtx = pFormatCtx;
  vidCtx->videoStreamIdx = videoStreamIdx;

  vidCtx->vCodecPar = vCodecPar;
  vidCtx->vCodec = vCodec;
  vidCtx->vCodecCtx = vCodecCtx;

  vidCtx->aCodecPar = aCodecPar;
  vidCtx->aCodec = aCodec;
  vidCtx->aCodecCtx = aCodecCtx;

  vidCtx->frameDecoded = pFrameDecoded;
  vidCtx->frameYUV = pFrameYUV;
  vidCtx->swsCtx = pSwsCtx;
  vidCtx->bufYUV = NULL;
  return 0;
}

void vid_context_close(VPVidContext *ctx) {
  sws_freeContext(ctx->swsCtx);
  av_frame_free(&ctx->frameDecoded);
  av_free(ctx->frameYUV->data[0]);
  av_frame_free(&ctx->frameYUV);
  vid_context_close_codec_context(&ctx->vCodecCtx);
  avformat_close_input(&ctx->formatCtx);
}

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
  SDL_Delay(10);
  return 0;
}

static int play_video(VPVidContext *vidCtx, VPPresContext *presCtx) {
  AVPacket packet;
  SDL_Event event;
  int frameIdx = 0;
  while (av_read_frame(vidCtx->formatCtx, &packet) >= 0) {
    // Handle events
    SDL_PollEvent(&event);
    switch (event.type) {
    case SDL_QUIT:
      av_packet_unref(&packet);
      return 0;
      break;
    default:
      break;
    }

    // Check packet from video stream
    if (packet.stream_index == vidCtx->videoStreamIdx) {
      // Should I send many and read many in this inner loop?

      // Try to send packet for decoding
      int sendRet = avcodec_send_packet(vidCtx->vCodecCtx, &packet);
      if (sendRet == AVERROR(EAGAIN)) {
        // try receiving frames
      } else if (sendRet < 0) {
        LOG_ERROR("avcodec_send_packet failed\n");
        return -1;
      }

      // Try to read a frame from decoder
      while (1) {
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
    }
    av_packet_unref(&packet);
  }
  printf("Frames seen %d\n", frameIdx);
  return 0;
}

// void q_test() {
//   VPQueue *q = queue_alloc();
//   if (q == NULL) {
//     LOG_ERROR("queue_init failed");
//     return;
//   };
//   // int x = 1;
//   int y = 2;
//   // int z = 3;
//   queue_put(q, (const void *)&y);
//   int *elem;
//   queue_get(q, (const void **)&elem);
//   printf("From queue: %d\n", *elem);
//   queue_free(q);
//   return;
// }

int main(int argc, char *argv[]) {
  // q_test();
  // return 0;

  VPVidContext vidCtx;
  if (vid_context_init(&vidCtx, argv[1]) < 0) {
    LOG_ERROR("vid_context_init failed");
    return -1;
  }

  SDL_AudioSpec wantedAudioSpec;
  SDL_AudioSpec *p_wantedAudioSpec = NULL;
  if (vidCtx.audioStreamIdx >= 0) {
    wantedAudioSpec.freq = vidCtx.aCodecCtx->sample_rate;
    wantedAudioSpec.format = AUDIO_S16SYS;
    wantedAudioSpec.samples = AUDIO_SAMPLE_SIZE;
    wantedAudioSpec.silence = 0;
    wantedAudioSpec.callback = NULL; // TODO: work from here

    p_wantedAudioSpec = &wantedAudioSpec;
  }
  VPPresContext presCtx;
  if (pres_context_init(&presCtx, p_wantedAudioSpec) < 0) {
    LOG_ERROR("pres_context_init failed");
    return -1;
  }

  if (play_video(&vidCtx, &presCtx) < 0) {
    LOG_ERROR("play_video failed");
  }

  vid_context_close(&vidCtx);
  pres_context_close(&presCtx);
  printf("success\n");
}
