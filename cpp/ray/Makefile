CXX=g++
CXXFLAGS=-g -Wall -Werror -std=c++17
CPPFLAGS=

vec3: vec3.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

ray-tracer: ray-tracer.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

write-simple-ppm: write-simple-ppm.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	$(RM) write-simple-ppm ray-tracer simple.ppm
