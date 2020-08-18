#include <iostream>

/*
 * I don't think this actually does a double free,
 * I think the compiler is smart enough to not do it here,
 * see cppcon 2019 back to basics "The best parts of c++"
 */

struct DD {
  // Look up: builtin move of a scalar type is a copy?
  DD(const std::size_t size) : data(new double[size]) {}
  ~DD() {
    std::cout << "deleting a DD\n";
    delete[] data; }
  double *data;
};

DD get_data() {
  DD data{3};
  data.data[0] = 1.1; data.data[1] = 2.2; data.data[2] = 3.3;
  return data;  // move
}

double sum_data(const DD &d) {
  return d.data[0] + d.data[1] + d.data[2];
}

int main() {
  std::cout << sum_data(get_data()) << "\n";
}
