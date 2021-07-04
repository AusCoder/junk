#ifndef _COMMON_H
#define _COMMON_H

int VP_ERR_FATAL = -1;
int VP_ERR_AGAIN = -2;

#define LOG_ERROR(msg) fprintf(stderr, "Error: %s\n", (msg))

#define LOG_WARNING(msg) fprintf(stderr, "Warning: %s\n", (msg))

#define LOG_SDL_ERROR(msg)                                                     \
  fprintf(stderr, "%s. SDL error: %s\n", (msg), SDL_GetError())

#endif // _COMMON_H
