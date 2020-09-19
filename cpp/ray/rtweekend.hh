#ifndef _RTWEEKEND_HH
#define _RTWEEKEND_HH

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

#include "ray.hh"
#include "vec3.hh"

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
  return pi * degrees / 180.0;
}

inline double random_double() {
  static std::uniform_real_distribution<double> distribution{0.0, 1.0};
  static std::mt19937 generator;
  return distribution(generator);
}

inline double random_double(double min, double max) {
  return min + (max - min) * random_double();
}

#endif
