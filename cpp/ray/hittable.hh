#ifndef _HITTABLE_HH
#define _HITTABLE_HH

#include "ray.hh"
#include "rtweekend.hh"
#include "vec3.hh"

struct material;

struct hit_record {
  point3 point;
  vec3 normal;
  shared_ptr<material> mat_ptr;
  double t;
  bool front_face;

  inline void set_front_face(const ray &r, const vec3 &outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  virtual bool hit(const ray &r, double t_min, double t_max,
                   hit_record &record) const = 0;
};

#endif
