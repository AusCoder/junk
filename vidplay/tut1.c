#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

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
    fprintf(stderr, "Unsupported codec!\n");
    return -1;
  }
  // AVCodecContext *pCodecCtxOrig = NULL;
  AVCodecContext *pCodecCtx = avcodec_alloc_context3(pCodec);
  if (avcodec_parameters_to_context(pCodecCtx, pCodecPar) != 0) {
    fprintf(stderr, "Failed to set codec parametsr\n");
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

  // int numBytes =
  //     avpicture_get_size(AV_PIX_FMT_RGB24, pCodecPar->width,
  //     pCodecPar->height);
  int numBytes =
      av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecPar->width,
                               pCodecPar->height, pCodecPar->block_align);
  uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  av_image_fill_arrays(pFrameRGB->data, pFrameRGB->linesize, buffer,
                       AV_PIX_FMT_RGB24, pCodecPar->width, pCodecPar->height,
                       pCodecPar->block_align);

  // Read frames
  int frameFinished;
  AVPacket avPacket;
  struct SwsContext *swsCtx = sws_getContext(
      pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
      pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
  if (swsCtx == NULL) {
    fprintf(stderr, "sws_getContext failed\n");
    return -1;
  }

  while (av_read_frame(pFormatCtx, &avPacket) >= 0) {
    // Check packet from video stream
    if (avPacket.stream_index == videoStreamIdx) {

      // Decode video frame
      // avcodec_decode_video2(
      //    pCodecCtx, pFrame, &frameFinished, &packet
      // );
      int ret = avcodec_send_packet(pCodecCtx, &avPacket);
      if (ret == AVERROR(EAGAIN)) {
        // need to try reading
      } else if (ret < 0) {
        // error
      }
      // if () {
      //   fprintf(stderr, "avcodec_send_packet failed\n");
      //   return -1;
      // }
    }
  }

  printf("success\n");
}
