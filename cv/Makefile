CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = $(shell pkg-config --cflags opencv4)
LDFLAGS =
LDLIBS = $(shell pkg-config --libs opencv4)

# Note to self: The order of arguments matters in the linking!
# This was once the cause of great confusion, may it never happen again!
imShow: imShow.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

resizeImage: resizeImage.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

showPixels: showPixels.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

hsvThreshold: hsvThreshold.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj imShow resizeImage showPixels hsvThreshold
