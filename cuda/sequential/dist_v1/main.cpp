#include <iostream>
#include <cmath>
#define N 64

using namespace std;

float scale(int i, int n)
{
  return ((float) i / (n - 1));
}

float distance(float x1, float x2)
{
  return sqrt((x2 - x1) * (x2 - x1));
}

int main()
{
  float out[N] = {0.0f};
  float ref = 0.5f;

  for (int i = 0; i < N; i++)
  {
    float x = scale(i, N);
    out[i] = distance(ref, x);
  }

  for (int i = 0; i < N; i++)
  {
    cout << out[i] << endl;
  }
  return 0;
}
