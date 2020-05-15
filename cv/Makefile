# CC = gcc-8
# CXX = g++-8
CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I/usr/include/opencv4 -I./include
LDFLAGS =
LDLIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

imShow: imShow.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

resizeImage: resizeImage.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

showPixels: showPixels.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj imShow resizeImage showPixels
