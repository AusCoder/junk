#ifndef _COMMON_H
#define _COMMON_H

int VP_ERR_FATAL = -1;
int VP_ERR_TIMEDOUT = -2;
int VP_ERR_AGAIN = -3;

#define VP_TRUE 1
#define VP_FALSE 0

#define LOG(level, fmt, ...)                                                   \
  fprintf(stderr, "%s: " fmt "\n", level, __VA_ARGS__)

#define LOG_ERROR(fmt, ...) LOG("Error", fmt, __VA_ARGS__)

#define LOG_WARNING(fmt, ...) LOG("Warning", fmt, __VA_ARGS__)

#define LOG_INFO(fmt, ...) LOG("Info", fmt, __VA_ARGS__)

#define LOG_SDL_ERROR(msg)                                                     \
  fprintf(stderr, "%s. SDL error: %s\n", (msg), SDL_GetError())

#endif // _COMMON_H
