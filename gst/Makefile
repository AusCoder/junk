CFLAGS = -g -Wall -std=c11
CPPFLAGS = -I./include $(shell pkg-config --cflags gstreamer-1.0)
LDFLAGS = -L./lib $(shell pkg-config --libs gstreamer-1.0)
LDLIBS =

writeWebcam: writeWebcam.o
	$(CC) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj writeWebcam
