#include <assert.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#define PRINT_STDERR(msg) fprintf(stderr, (msg))

void save_frame(AVFrame *pFrame, int width, int height, int frameIdx) {
  char szFilename[32];
  sprintf(szFilename, "frame%d.ppm", frameIdx);

  FILE *pFile = fopen(szFilename, "wb");
  if (pFile == NULL) {
    PRINT_STDERR("Failed to open file\n");
    return;
  }

  fprintf(pFile, "P6\n%d %d\n255\n", width, height);
  for (int i = 0; i < height; i++) {
    fwrite(pFrame->data[0] + i * pFrame->linesize[0], 1, width * 3, pFile);
  }
  fclose(pFile);
}

int main(int argc, char *argv[]) {
  // av_register_all();
  const char *find_path = argv[1];
  AVFormatContext *pFormatCtx = NULL;
  if (avformat_open_input(&pFormatCtx, find_path, NULL, NULL) != 0) {
    return -1;
  }
  if (avformat_find_stream_info(pFormatCtx, NULL) != 0) {
    return -1;
  }
  // Debug info, like ffprobe
  av_dump_format(pFormatCtx, 0, find_path, 0);
  // Find video stream
  // AVCodecContext *pCodecCtxOrig = NULL;
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
    PRINT_STDERR("Unsupported codec!\n");
    return -1;
  }
  // AVCodecContext *pCodecCtxOrig = NULL;
  AVCodecContext *pCodecCtx = avcodec_alloc_context3(pCodec);
  if (avcodec_parameters_to_context(pCodecCtx, pCodecPar) != 0) {
    PRINT_STDERR("Failed to set codec parametsr\n");
    return -1;
  }
  if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
    return -1;
  }
  // Getting frames
  AVFrame *pFrame = av_frame_alloc();
  if (pFrame == NULL) {
    return -1;
  }
  AVFrame *pFrameRGB = av_frame_alloc();
  if (pFrameRGB == NULL) {
    return -1;
  }

  int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecPar->width,
                                          pCodecPar->height, 1);
  uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  if (buffer == NULL) {
    PRINT_STDERR("av_alloc failed\n");
    return -1;
  }
  if (av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer,
                           AV_PIX_FMT_RGB24, pCodecPar->width,
                           pCodecPar->height, 1) < 0) {
    PRINT_STDERR("av_image_fill_arrays failed\n");
    return -1;
  }

  // Read frames
  struct SwsContext *swsCtx = sws_getContext(
      pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
      pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
  if (swsCtx == NULL) {
    PRINT_STDERR("sws_getContext failed\n");
    return -1;
  }

  AVPacket avPacket;
  int frameIdx = 0;
  while (av_read_frame(pFormatCtx, &avPacket) >= 0) {
    // Check packet from video stream
    if (avPacket.stream_index == videoStreamIdx) {
      // Decode video frame
      // avcodec_decode_video2(
      //    pCodecCtx, pFrame, &frameFinished, &packet
      // );
      int sendRet = avcodec_send_packet(pCodecCtx, &avPacket);
      if (sendRet == AVERROR(EAGAIN)) {
        // need to try reading
        printf("Can't send frame, will try to receive\n");
      } else if (sendRet < 0) {
        // error
        PRINT_STDERR("avcodec_send_packet failed\n");
        return -1;
      }
      int recvRet = avcodec_receive_frame(pCodecCtx, pFrame);
      if (recvRet == AVERROR(EAGAIN)) {
        printf("Can't receive frame, will try to send\n");
        continue;
      } else if (recvRet < 0) {
        PRINT_STDERR("avcodec_receive_frame failed\n");
        return -1;
      }
      // We got a frame!

      // Surely this will fail because we have not called av_image_fill_arrays
      // pFrame contains a null pointer data array
      sws_scale(swsCtx, (uint8_t const *const *)pFrame->data, pFrame->linesize,
                0, pCodecCtx->height, pFrameRGB->data, pFrameRGB->linesize);
      if (frameIdx < 5) {
        save_frame(pFrameRGB, pCodecCtx->width, pCodecCtx->height, frameIdx);
      }
      frameIdx++;
    }
  }
  sws_freeContext(swsCtx);
  avcodec_close(pCodecCtx);
  av_frame_free(&pFrame);
  // free buffer in here too?
  av_frame_free(&pFrameRGB);
  av_free(buffer);
  avcodec_free_context(&pCodecCtx);
  avformat_close_input(&pFormatCtx);

  printf("success\n");
}
