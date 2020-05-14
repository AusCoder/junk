CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I./include $(shell pkg-config --cflags gstreamer-1.0)
LDFLAGS = -L./lib $(shell pkg-config --libs gstreamer-1.0)
LDLIBS =

writeWebcam: writeWebcam.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj writeWebcam
