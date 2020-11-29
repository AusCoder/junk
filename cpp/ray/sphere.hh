#ifndef _SPHERE_H
#define _SPHERE_H

#include "hittable.hh"
#include "ray.hh"
#include "vec3.hh"

class sphere : public hittable {
public:
  sphere() {}
  sphere(const point3 &c, double r, shared_ptr<material> m)
      : center{c}, radius{r}, mat_ptr{m} {}

  virtual bool hit(const ray &r, double t_min, double t_max,
                   hit_record &record) const override;

private:
  point3 center;
  double radius;
  shared_ptr<material> mat_ptr;
};

bool sphere::hit(const ray &r, double t_min, double t_max,
                 hit_record &record) const {
  vec3 oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius * radius;
  auto descriminant = half_b * half_b - a * c;

  bool is_hit = false;
  double t = 0;
  if (descriminant >= 0) {
    auto sqrt_descriminant = std::sqrt(descriminant);
    auto t1 = (-half_b - sqrt_descriminant) / a;
    auto t2 = (-half_b + sqrt_descriminant) / a;
    if ((t1 > t_min) && (t1 <= t_max)) {
      is_hit = true;
      t = t1;
    } else if ((t2 >= t_min) && (t2 <= t_max)) {
      is_hit = true;
      t = t2;
    }
  }
  if (is_hit) {
    record.point = r.at(t);
    auto outward_normal = (record.point - center) / radius;
    record.set_front_face(r, outward_normal);
    record.t = t;
    record.mat_ptr = mat_ptr;
  }
  return is_hit;
}

#endif
