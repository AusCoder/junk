#include "color.hh"
#include "ray.hh"
#include "vec3.hh"
#include <iostream>

color ray_color(const ray &r) {
  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {
  // Image size
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 400;
  const int image_height = static_cast<int>(image_width / aspect_ratio);

  // Camera position
  auto viewport_height = 2.0;
  auto viewport_width = aspect_ratio * viewport_height;
  auto focal_length = 1.0;

  std::cout << "P3\n" << image_width << ' ' << image_height << ' ' << "\n255\n";

  for (int j = image_height - 1; j >= 0; j--) {
    std::cerr << "\rScan lines remaining: " << j << std::flush;
    for (int i = 0; i < image_width; i++) {
      color pixel_color{static_cast<double>(i) / (image_width - 1),
                        static_cast<double>(j) / (image_height - 1), 0.25};
      write_color(std::cout, pixel_color);
    }
  }
  std::cerr << "\nDone\n";
}
