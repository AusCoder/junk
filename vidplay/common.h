#ifndef _COMMON_H
#define _COMMON_H

int VP_ERR_FATAL = -1;
int VP_ERR_TIMEDOUT = -2;
int VP_ERR_AGAIN = -3;

#define VP_TRUE 1
#define VP_FALSE 0

#define LOG(level, fmt, ...)                                                   \
  fprintf(stderr, "%s: " fmt "\n", level, __VA_ARGS__)

#define LOG_ERROR_FMT(fmt, ...) LOG("Error", fmt, __VA_ARGS__)
#define LOG_ERROR_MSG(msg) LOG_ERROR_FMT("%s", msg)

#define LOG_WARNING_FMT(fmt, ...) LOG("Warning", fmt, __VA_ARGS__)
#define LOG_WARNING_MSG(msg) LOG_WARNING_FMT("%s", msg)

#define LOG_INFO_FMT(fmt, ...) LOG("Info", fmt, __VA_ARGS__)
#define LOG_INFO_MSG(msg) LOG_INFO_FMT("%s", msg)

#define LOG_SDL_ERROR(msg)                                                     \
  fprintf(stderr, "%s. SDL error: %s\n", (msg), SDL_GetError())

#endif // _COMMON_H
