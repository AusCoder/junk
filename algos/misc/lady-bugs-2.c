// From http://orac.amt.edu.au/cgi-bin/train/hub.pl
#include <stdio.h>
#include <assert.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define ABS(X) ((X) < 0 ? -(X) : X)

int solve_1d_from(int *posns, int n, int k, int i, int dir) {
  if (dir == 0) { // left
    int l = i - k + 1 >= 0 ? posns[i] - posns[i - k + 1] : -1;
    //    printf("Solve left k: %d i: %d res: %d\n", k, i, l);
    return l;
  } else if (dir == 1) { // right
    int r = i + k - 1 < n ? posns[i + k - 1] - posns[i] : -1;
    //    printf("Solve right k: %d i: %d res: %d\n", k, i, r);
    return r;
  }
  assert(0);
}

int solve_from(int *posns, int n, int k, int q) {
  int i = 0;
  while (i < n) {
    if (posns[i] < q) {
      i++;
    } else {
      break;
    }
  }
  // i points at or after q
  //  printf("i:%d\n",i);

  int best = -1;

  // handle case of outside posns separately
  if (i == 0) {
    int s = solve_1d_from(posns, n, k, 0, 1);
    if (s >= 0) {
      int dist = s + posns[0] - q;
      if (best < 0) {
        best = dist;
      }
      best = MIN(best, dist);
    }
  }

  if (i == n) {
    int s = solve_1d_from(posns, n, k, n - 1, 0);
    if (s >= 0) {
      int dist = s + q - posns[n - 1];
      if (best < 0) {
        best = dist;
      }
      best = MIN(best, dist);
    }
  }

  // go right
  for (int j = i; j < n; j++) {
    int newk = MIN(n, k + (j - i));
    int s = solve_1d_from(posns, n, newk, j, 0);
    if (s >= 0) {
      int dist = s + posns[j] - q;
      //      printf("right search i:%d j:%d newk:%d s:%d dist:%d\n", i,j,newk,s,dist);
      if (best < 0) {
        best = dist;
      }
      best = MIN(best, dist);
    }
  }

  // go left
  int di = (i < n) && (q == posns[i]) ? 0 : 1;
  for (int j = 0; i - di >= 0; j++, di++) {
    int newk = MIN(n, k + j);
    int s = solve_1d_from(posns, n, newk, i - di, 1);
    if (s >= 0) {
      int dist = s + q - posns[i - di];
      //      printf("left search i:%d j:%d newk:%d s:%d dist:%d\n", i,j,newk,s,dist);
      if (best < 0) {
        best = dist;
      }
      best = MIN(best, dist);
    }
  }

  return best;
}

int main() {
  FILE *f_in = fopen("ladyin.txt", "r");
  FILE *f_out = fopen("ladyout.txt", "w");
  int n,k,q;
  fscanf(f_in, "%d %d %d", &n, &k, &q);
  int posns[n];
  for(int i = 0; i < n; i++) {
    fscanf(f_in, "%d", posns + i);
  }

  for (int i = 0; i < q; i++) {
    int pos;
    fscanf(f_in, "%d", &pos);
    fprintf(f_out, "%d", solve_from(posns, n, k, pos));
    if (i < q - 1) {
      fprintf(f_out, " ");
    }
  }
  fprintf(f_out, "\n");

  fclose(f_in);
  fclose(f_out);

  return 0;
}
