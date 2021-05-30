/* Decodes and displays a video in an SDL window.

Questions:
  - Is AVFrame->data always continuous? Or do I
    need to copy to a continuous buffer for SDL texture?
*/
#include <assert.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480

#define LOG_ERROR(msg) fprintf(stderr, "Error: %s\n", (msg))

#define LOG_SDL_ERROR(msg)                                                     \
  fprintf(stderr, "%s. SDL error: %s\n", (msg), SDL_GetError())

typedef struct {
  AVFormatContext *formatCtx;
  AVCodecParameters *codecPar;
  AVCodec *codec;
  AVCodecContext *codecCtx;
  AVFrame *frameDecoded;
  AVFrame *frameYUV;
  struct SwsContext *swsCtx;
  int videoStreamIdx;
  uint8_t *bufYUV;
} VPVidContext;

int create_vidplay_context(VPVidContext *vidCtx, const char *path) {
  AVFormatContext *pFormatCtx = NULL;
  if (avformat_open_input(&pFormatCtx, path, NULL, NULL) != 0) {
    return -1;
  }
  if (avformat_find_stream_info(pFormatCtx, NULL) != 0) {
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  AVCodecParameters *pCodecPar = NULL;
  int videoStreamIdx = -1;
  for (int i = 0; i < pFormatCtx->nb_streams; i++) {
    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStreamIdx = i;
      break;
    }
  }
  if (videoStreamIdx < 0 || videoStreamIdx >= pFormatCtx->nb_streams) {
    return -1;
  }
  pCodecPar = pFormatCtx->streams[videoStreamIdx]->codecpar;
  // Open codec
  AVCodec *pCodec = avcodec_find_decoder(pCodecPar->codec_id);
  if (pCodec == NULL) {
    LOG_ERROR("Unsupported codec");
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  AVCodecContext *pCodecCtx = avcodec_alloc_context3(pCodec);
  if (avcodec_parameters_to_context(pCodecCtx, pCodecPar) != 0) {
    LOG_ERROR("Failed to set codec parameters");
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
    LOG_ERROR("Failed to open codec context");
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  // Getting frames
  AVFrame *pFrameDecoded = av_frame_alloc();
  if (pFrameDecoded == NULL) {
    LOG_ERROR("Failed to alloc frame");
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  AVFrame *pFrameYUV = av_frame_alloc();
  if (pFrameYUV == NULL) {
    LOG_ERROR("Failed to alloc frame");
    av_frame_free(&pFrameDecoded);
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
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
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
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
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  struct SwsContext *pSwsCtx = sws_getContext(
      pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, SCREEN_WIDTH,
      SCREEN_HEIGHT, AV_PIX_FMT_YUV420P, SWS_BILINEAR, NULL, NULL, NULL);
  if (pSwsCtx == NULL) {
    LOG_ERROR("sws_getContext failed");
    av_free(buffer);
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameYUV);
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  vidCtx->formatCtx = pFormatCtx;
  vidCtx->codecPar = pCodecPar;
  vidCtx->codec = pCodec;
  vidCtx->codecCtx = pCodecCtx;
  vidCtx->frameDecoded = pFrameDecoded;
  vidCtx->frameYUV = pFrameYUV;
  vidCtx->swsCtx = pSwsCtx;
  vidCtx->videoStreamIdx = videoStreamIdx;
  vidCtx->bufYUV = NULL;
  return 0;
}

void close_vidplay_context(VPVidContext *ctx) {
  sws_freeContext(ctx->swsCtx);
  avcodec_close(ctx->codecCtx);
  av_frame_free(&ctx->frameDecoded);
  av_free(ctx->frameYUV->data[0]);
  av_frame_free(&ctx->frameYUV);
  avcodec_free_context(&ctx->codecCtx);
  avformat_close_input(&ctx->formatCtx);
}

int play_video(VPVidContext *vidCtx, SDL_Renderer *renderer,
               SDL_Texture *texture) {

  AVPacket packet;
  int frameIdx = 0;
  while (av_read_frame(vidCtx->formatCtx, &packet) >= 0) {
    // Check packet from video stream
    if (packet.stream_index == vidCtx->videoStreamIdx) {
      // Should I send many and read many in this inner loop?

      // Try to send packet for decoding
      int sendRet = avcodec_send_packet(vidCtx->codecCtx, &packet);
      if (sendRet == AVERROR(EAGAIN)) {
        // need to try reading
      } else if (sendRet < 0) {
        LOG_ERROR("avcodec_send_packet failed\n");
        return -1;
      }
      // Try to read a frame from decoder
      int recvRet =
          avcodec_receive_frame(vidCtx->codecCtx, vidCtx->frameDecoded);
      if (recvRet == AVERROR(EAGAIN)) {
        // Can't receive a frame, need to try to send again
      } else if (recvRet < 0) {
        LOG_ERROR("avcodec_receive_frame failed\n");
        return -1;
      } else {
        // Got a frame
        sws_scale(vidCtx->swsCtx,
                  (uint8_t const *const *)vidCtx->frameDecoded->data,
                  vidCtx->frameDecoded->linesize, 0, vidCtx->codecCtx->height,
                  vidCtx->frameYUV->data, vidCtx->frameYUV->linesize);
        frameIdx++;

        assert(vidCtx->frameYUV->linesize[0] == SCREEN_WIDTH);
        assert(SCREEN_WIDTH * SDL_BYTESPERPIXEL(SDL_PIXELFORMAT_IYUV) ==
               SCREEN_WIDTH);

        // YUV - Y has 1 byte per pixel
        assert(vidCtx->frameYUV->data[1] ==
               vidCtx->frameYUV->data[0] + SCREEN_WIDTH * SCREEN_HEIGHT * 1);
        assert(vidCtx->frameYUV->data[2] ==
               vidCtx->frameYUV->data[1] +
                   SCREEN_WIDTH * SCREEN_HEIGHT * 1 / 2 / 2);

        if (SDL_UpdateTexture(
                texture, NULL, vidCtx->frameYUV->data[0],
                SCREEN_WIDTH * SDL_BYTESPERPIXEL(SDL_PIXELFORMAT_IYUV)) < 0) {
          LOG_SDL_ERROR("SDL_UpdateTexture failed");
          return -1;
        }
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        SDL_Delay(10);
      }
    }
    av_packet_unref(&packet);
  }
  return 0;
}

int main(int argc, char *argv[]) {
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

  // What does SDL_TEXTUREACCESS_ do?
  // or SDL_PIXELFORMAT_IYUV
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

  VPVidContext vidCtx;
  if (create_vidplay_context(&vidCtx, argv[1]) < 0) {
    LOG_ERROR("create_vidplay_context failed");
    return -1;
  }

  if (play_video(&vidCtx, renderer, texture) < 0) {
    LOG_ERROR("play_video failed");
  }

  close_vidplay_context(&vidCtx);
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  printf("success\n");
}
