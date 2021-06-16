/*
  Find minimum of a rotated sorted array in O(log n) time.
Eg
[4,5,6,7,0,1,2]
 */

#include <stdio.h>
#include <assert.h>

int findMin(int *arr, int arrSize) {
  int l = 0;
  int r = arrSize - 1;
  // check for array sorted
  if (arr[l] <= arr[r]) {
    return arr[l];
  }
  for (;;) {
    assert(l < r);
    int m = l + (r - l) / 2;
    if (m == l) {
      return arr[l] < arr[r] ? arr[l] : arr[r];
    } else if (arr[l] < arr[m]) {
      l = m;
    } else {
      r = m;
    }
  }
}

int main(int argc, char *argv[]){
  /* int arr[] = {4,5,6,7,0,1,2}; */
  /* int arrSize = 7; */
  /* int arr[] = {4,5,6,7,0,1}; */
  /* int arrSize = 6; */
  int arr[] = {11,13,15,17};
  int arrSize = 4;
  printf("Min is %d\n", findMin(arr, arrSize));
  return 0;
}
