#include "camera.hh"
#include "color.hh"
#include "hittable_list.hh"
#include "material.hh"
#include "rtweekend.hh"
#include "sphere.hh"

#include <iostream>

color ray_color(const ray &r, const hittable &world, int depth) {
  hit_record rec;
  if (depth <= 0) {
    return {0, 0, 0};
  }
  // Use min of 0.001 to deal with floating point approximation errors
  // of rays reflected off objects
  if (world.hit(r, 0.001, infinity, rec)) {
    // point3 target = rec.point + rec.normal + random_unit_vector();
    // return 0.5 * ray_color({rec.point, target - rec.point}, world, depth -
    // 1);
    ray scattered;
    color attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered) > 0) {
      return attenuation * ray_color(scattered, world, depth - 1);
    } else {
      return {0, 0, 0};
    }
  }
  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {
  // Image size
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 400;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 100;
  const int max_depth = 50;

  // World
  hittable_list world;

  auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
  auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
  auto material_left = make_shared<dielectric>(1.5);
  auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

  world.add(
      make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
  world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
  world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
  world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.45, material_left));
  world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

  // Camera
  camera cam{point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0), 20.0,
             aspect_ratio};

  std::cout << "P3\n" << image_width << ' ' << image_height << ' ' << "\n255\n";

  for (int j = image_height - 1; j >= 0; j--) {
    std::cerr << "\r                                  "
              << "\rScan lines remaining: " << j << std::flush;
    for (int i = 0; i < image_width; i++) {
      color pixel_color{0, 0, 0};
      for (int s = 0; s < samples_per_pixel; s++) {
        auto u = static_cast<double>(i + random_double()) / (image_width - 1);
        auto v = static_cast<double>(j + random_double()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, max_depth);
      }
      write_color(std::cout, pixel_color, samples_per_pixel);
    }
  }
  std::cerr << "\nDone\n";
}
