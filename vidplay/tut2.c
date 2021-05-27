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
  AVFrame *frameRGB;
  struct SwsContext *swsCtx;
} VPVidContext;

int create_vidplay_context(VPVidContext *ctx, const char *path) {
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
  AVFrame *pFrameRGB = av_frame_alloc();
  if (pFrameRGB == NULL) {
    LOG_ERROR("Failed to alloc frame");
    av_frame_free(&pFrameDecoded);
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecPar->width,
                                          pCodecPar->height, 1);
  uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  if (buffer == NULL) {
    LOG_ERROR("Failed to alloc buffer");
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameRGB);
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  if (av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer,
                           AV_PIX_FMT_RGB24, pCodecPar->width,
                           pCodecPar->height, 1) < 0) {
    LOG_ERROR("av_image_fill_arrays failed");
    av_free(buffer);
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameRGB);
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }

  struct SwsContext *pSwsCtx = sws_getContext(
      pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
      pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
  if (pSwsCtx == NULL) {
    LOG_ERROR("sws_getContext failed");
    av_free(buffer);
    av_frame_free(&pFrameDecoded);
    av_frame_free(&pFrameRGB);
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return -1;
  }
  ctx->formatCtx = pFormatCtx;
  ctx->codecPar = pCodecPar;
  ctx->codec = pCodec;
  ctx->codecCtx = pCodecCtx;
  ctx->frameDecoded = pFrameDecoded;
  ctx->frameRGB = pFrameRGB;
  ctx->swsCtx = pSwsCtx;
  return 0;
}

void close_vidplay_context(VPVidContext *ctx) {
  sws_freeContext(ctx->swsCtx);
  avcodec_close(ctx->codecCtx);
  av_frame_free(&ctx->frameDecoded);
  av_free(ctx->frameRGB->data[0]);
  av_frame_free(&ctx->frameRGB);
  avcodec_free_context(&ctx->codecCtx);
  avformat_close_input(&ctx->formatCtx);
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

  SDL_Texture *texture =
      SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                        SDL_TEXTUREACCESS_STATIC, SCREEN_WIDTH, SCREEN_HEIGHT);
  if (texture == NULL) {
    LOG_SDL_ERROR("SDL_CreateTexture");
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return -1;
  }

  // SDL_Surface *screen;
  // screen = SDL_setVideo

  VPVidContext vidCtx;
  if (create_vidplay_context(&vidCtx, argv[1]) < 0) {
    LOG_ERROR("create_vidplay_context failed");
    return -1;
  }

  close_vidplay_context(&vidCtx);
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  printf("success\n");
}
