#ifndef _VEC_3_HH
#define _VEC_3_HH

#include <array>
#include <cmath>
#include <ostream>

class vec3 {
public:
  vec3() : elements{0, 0, 0} {}
  vec3(double e1, double e2, double e3) : elements{e1, e2, e3} {}

  double x() const { return elements[0]; }
  double y() const { return elements[0]; }
  double z() const { return elements[0]; }

  vec3 operator-() { return {-elements[0], -elements[1], -elements[2]}; }
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

private:
  std::array<double, 3> elements;
};

inline std::ostream &operator<<(std::ostream &s, const vec3 &x) {
  return s << x[0] << ' ' << x[1] << ' ' << x[2];
}

inline vec3 operator+(const vec3 &x, const vec3 &y) {
  return {x[0] + y[0], x[1] + y[1], x[2] + y[2]};
}

inline vec3 operator-(const vec3 &x, const vec3 &y) {
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

#endif
