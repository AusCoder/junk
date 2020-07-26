CXX = g++
CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS =
LDFLAGS =
LDLIBS =

list-main: list.o list-main.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) list-main *.o *.obj
