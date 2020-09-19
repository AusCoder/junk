#ifndef _HITTABLE_LIST_HH
#define _HITTABLE_LIST_HH

#include "hittable.hh"

#include <memory>
#include <vector>

class hittable_list : public hittable {
public:
  hittable_list() {}
  hittable_list(std::shared_ptr<hittable> object) { add(object); }

  void clear() { objects.clear(); }
  void add(std::shared_ptr<hittable> object) { objects.push_back(object); }

  virtual bool hit(const ray &r, double t_min, double t_max,
                   hit_record &record) const override;

private:
  std::vector<std::shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray &r, double t_min, double t_max,
                        hit_record &record) const {
  hit_record temp_rec;
  bool hit_anything = false;
  auto min_so_far = t_max;
  for (auto &obj : objects) {
    if (obj->hit(r, t_min, min_so_far, temp_rec)) {
      hit_anything = true;
      min_so_far = temp_rec.t;
      record = temp_rec;
    }
  }
  return hit_anything;
}

#endif
