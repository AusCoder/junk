#ifndef _VEC_3_HH
#define _VEC_3_HH

#include "rtweekend.hh"
#include <array>
#include <cmath>
#include <ostream>

class vec3 {
public:
  vec3() : elements{0, 0, 0} {}
  vec3(double e1, double e2, double e3) : elements{e1, e2, e3} {}

  vec3(const vec3 &x) = default;

  double x() const { return elements[0]; }
  double y() const { return elements[1]; }
  double z() const { return elements[2]; }

  vec3 operator-() const { return {-elements[0], -elements[1], -elements[2]}; }
  double operator[](int i) const { return elements.at(i); }
  double &operator[](int i) { return elements.at(i); }

  vec3 &operator+=(const vec3 &other) {
    elements[0] += other.elements[0];
    elements[1] += other.elements[1];
    elements[2] += other.elements[2];
    return *this;
  }
  vec3 &operator*=(double x) {
    elements[0] *= x;
    elements[1] *= x;
    elements[2] *= x;
    return *this;
  }
  vec3 &operator/=(double x) { return *this *= 1 / x; }

  double length() const { return std::sqrt(length_squared()); }

  double length_squared() const {
    return elements[0] * elements[0] + elements[1] * elements[1] +
           elements[2] * elements[2];
  }

  bool near_zero() const {
    const auto s = 1e-8;
    return (fabs(elements.at(0)) < s) && (fabs(elements.at(1)) < s) &&
           (fabs(elements.at(2)) < s);
  }

  inline static vec3 random() {
    return {random_double(), random_double(), random_double()};
  }

  inline static vec3 random(double min, double max) {
    return {random_double(min, max), random_double(min, max),
            random_double(min, max)};
  }

private:
  std::array<double, 3> elements;
};

// Aliases
using point3 = vec3;
using color = vec3;

// Utility functions
inline std::ostream &operator<<(std::ostream &s, const vec3 &x) {
  return s << x[0] << ' ' << x[1] << ' ' << x[2];
}

inline vec3 operator+(const vec3 &x, const vec3 &y) {
  return {x[0] + y[0], x[1] + y[1], x[2] + y[2]};
}

inline vec3 operator-(const vec3 &x, const vec3 &y) {
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

inline vec3 operator*(const vec3 &x, const vec3 &y) {
  return {x[0] * y[0], x[1] * y[1], x[2] * y[2]};
}

inline vec3 operator*(const vec3 &x, double t) {
  return {t * x[0], t * x[1], t * x[2]};
}

inline vec3 operator*(double t, const vec3 &x) { return x * t; }

inline vec3 operator/(const vec3 &x, double t) { return x * (1 / t); }

inline double dot(const vec3 &x, const vec3 &y) {
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

inline vec3 cross(const vec3 &x, const vec3 &y) {
  return {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2],
          x[0] * y[1] - x[1] * y[0]};
}

inline vec3 unit_vector(const vec3 &x) { return x / x.length(); }

vec3 reflect(const vec3 &v, const vec3 &n) { return v - 2 * dot(v, n) * n; }

vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat) {
  auto cos_theta = fmin(dot(-uv, n), 1.0);
  vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
  vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
  return r_out_perp + r_out_parallel;
}

vec3 random_in_unit_sphere() {
  while (true) {
    vec3 vec{vec3::random(-1, 1)};
    if (vec.length_squared() < 1) {
      return vec;
    }
  }
}

vec3 random_unit_vector() { return unit_vector(random_in_unit_sphere()); }

vec3 random_in_hemisphere(const vec3 &normal) {
  vec3 unit_in_sphere = random_in_unit_sphere();
  if (dot(unit_in_sphere, normal) > 0) {
    return unit_in_sphere;
  } else {
    return -unit_in_sphere;
  }
}

#endif
