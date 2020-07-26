/* Implement 3 stacks in 1 array
Idea: think about the stacks living on a circle.
*/

#include <iostream>

#include <deque>
#include <memory>
#include <stdexcept>
#define N 3

using namespace std;

/* Cracking the coding interview: 3.1 */
template<class A>
class NStackArray
{
    public:
        NStackArray(int capacity_):
            capacity(capacity_), arr(new A[capacity_]), start(new int[N]), end(new int[N])
        {
            for (int i = 0; i < N; i++)
            {
                start[i] = i * (capacity / N);
                end[i] = start[i];
            }
        }

        void push(int stackIdx, A val)
        {
            if (collidesWithNextStack(stackIdx)) {
                shiftStack(nextStackIdx(stackIdx), stackIdx);
            }
            arr[end[stackIdx]] = val;
            end[stackIdx] = (end[stackIdx] + 1) % capacity;
        }

        A pop(int stackIdx)
        {
            if (start[stackIdx] == end[stackIdx]) {
                throw underflow_error("nothing pushed");
            }
            end[stackIdx] = (end[stackIdx] - 1) % capacity;
            return arr[end[stackIdx]];
        }

        void printAll()
        {
            for (int i = 0; i < capacity; i++)
            {
                cout << arr[i] << ", ";
            }
            cout << endl;
        }

    private:
        bool collidesWithNextStack(int stackIdx) {
            return end[stackIdx] == start[nextStackIdx(stackIdx)];
        }

        // Simpler version of shift stack that iterates backwards
        void shiftStack(int stackIdx, int origStackIdx) {
            if (stackIdx == origStackIdx) {
                throw overflow_error("arr full");
            }
            if (collidesWithNextStack(stackIdx)) {
                shiftStack(nextStackIdx(stackIdx), origStackIdx);
            }

            int e = end[stackIdx];
            int s = start[stackIdx];
            if (e < s) {
                e += capacity;
            }
            for (int i = e; i > s; i--) {
                arr[i % capacity] = arr[(i - 1) % capacity];
            }
            start[stackIdx] = (start[stackIdx] + 1) % capacity;
            end[stackIdx] = (end[stackIdx] + 1) % capacity;
        }

        // shift stack by pushing elems to a queue
        void shiftStackUsingQueue(int stackIdx, int origStackIdx) {
            if (stackIdx == origStackIdx) {
                throw overflow_error("arr full");
            }

            if (collidesWithNextStack(stackIdx)) {
                shiftStackUsingQueue(nextStackIdx(stackIdx), origStackIdx);
            }

            deque<A> q;
            q.push_back(arr[start[stackIdx]]);
            int idx = start[stackIdx];
            while (idx != end[stackIdx])
            {
                int nextIdx = (idx + 1) % capacity;
                q.push_back(arr[nextIdx]);
                arr[nextIdx] = q.front();
                q.pop_front();
                idx = (idx + 1) % capacity;
            }
            start[stackIdx] = (start[stackIdx] + 1) % capacity;
            end[stackIdx] = (end[stackIdx] + 1) % capacity;
        }

        inline int nextStackIdx(int stackIdx) {
            return (stackIdx + 1) % N;
        }

        int capacity;
        unique_ptr<A []> arr;
        unique_ptr<int []> start;
        unique_ptr<int []> end;
};

int main()
{
    NStackArray<int> stack(12);
    stack.printAll();

    stack.push(0, 1);
    stack.push(0, 2);

    stack.push(1, 1);
    stack.push(1, 2);

    stack.push(2, 1);
    stack.push(2, 2);
    stack.push(2, 3);
    stack.push(2, 4);

    stack.push(1, 3);
    stack.push(1, 4);
    stack.push(1, 5);
    stack.push(1, 6);

    stack.push(0, stack.pop(0) + 1);
    stack.push(1, stack.pop(1) + 1);
    stack.push(2, stack.pop(2) + 1);

    stack.printAll();
}
