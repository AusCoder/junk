#include <math.h>
#include <stdio.h>

// The old school fast way to compute
// inverse sqrt.
float fast_approx_inverse_sqrt(float number) {
  int i;
  float x2, y;
  const float threehalfs = 1.5f;

  x2 = number * 0.5F;
  y = number;
  i = *(int *)&y;
  i = 0x5f3759df - (i >> 1);
  y = *(float *)&i;
  y = y * (threehalfs - (x2 * y * y));
  return y;
}

int main() {
  float f1 = 317.0 / 46;
  // Proper cast
  int i1 = (int)f1;
  // Cast ptr and dereference,
  // interprets bit pattern of f1 as an int
  int i2 = *(int *)&f1;
  // Similar to casting ptr, access
  // float bit pattern as ptr
  union {
    float f;
    int i;
  } u = {.f = f1};
  printf("Float is: %f. Int is: %d. Ptr cast int is: %d. Unioned float int is: "
         "%d\n",
         f1, i1, i2, u.i);

  float f2 = 0.156f;
  printf("Inverse sqrt: %f. Fast approx inverse sqrt: %f\n", 1 / sqrt(f2),
         fast_approx_inverse_sqrt(f2));
}
