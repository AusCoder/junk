CC = gcc-8
CXX = g++-8
CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I/usr/include/opencv4 -I./include
LDFLAGS =
LDLIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui

imShow: imShow.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj imShow
