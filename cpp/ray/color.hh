#ifndef _COLOR_HH
#define _COLOR_HH

#include "rtweekend.hh"
#include "vec3.hh"
#include <ostream>

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
  // In multi sampled pixels, the color is not between
  // 0 and 1. It is between 0 and samples_per_pixel.
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

  auto scale = 1.0 / samples_per_pixel;
  r *= scale;
  g *= scale;
  b *= scale;

  out << static_cast<int>(256 * clamp(r, 0, 0.999)) << ' '
      << static_cast<int>(256 * clamp(g, 0, 0.999)) << ' '
      << static_cast<int>(256 * clamp(b, 0, 0.999)) << '\n';
}

#endif
