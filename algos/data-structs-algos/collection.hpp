#include <memory>
#include <iostream>

using namespace std;

template<class Object>
class CollectionUniquePrt
{
    public:
        explicit CollectionUniquePrt(int capacity): ptr(new Object[capacity]), size(0), capacity(capacity)
        {}

        CollectionUniquePrt(const CollectionUniquePrt<Object> &col): ptr(new Object[col.capacity]), size(col.size), capacity(col.capacity)
        {
            for (int i = 0; i < size; i++) {
                ptr[i] = col.ptr[i];
            }
        }

        bool isEmpty() const
        {
            return size == 0;
        }

        void makeEmpty() {
            size = 0;
        }

        void insert(const Object &obj) {
            if (size >= capacity) {
                return;
            }
            ptr[size] = obj;
            size++;
        }

        void remove(const Object &obj) {
            int i = 0;
            for(; i < size; i++) {
                if (ptr[i] == obj) {
                    break;
                }
            }
            if (i == size) {
                return;
            }
            size--;
            for(; i < size; i++) {
                ptr[i] = ptr[i + 1];
            }
        }

        bool contains(const Object &obj) const {
            for (int i = 0; i < size; i++) {
                if (ptr[i] == obj) {
                    return true;
                }
            }
            return false;
        }

        void printAll() const {
            for (int i = 0; i < size; i++) {
                cout << ptr[i] << " ,";
            }
            cout << endl;
        }

    private:
        unique_ptr<Object []> ptr;
        int size;
        int capacity;
};


template<class Object>
class CollectionRaw
{
    public:
        explicit CollectionRaw(int capacity): ptr(new Object[capacity]), size(0), capacity(capacity)
        {}

        CollectionRaw(const CollectionRaw &col): ptr(new Object[col.capacity]), size(col.size), capacity(col.capacity)
        {
            for (int i = 0; i < size; i++) {
                ptr[i] = col.ptr[i];
            }
        }

        ~CollectionRaw()
        {
            delete[] ptr;
        }

        bool isEmpty() const
        {
            return size == 0;
        }

        void makeEmpty() {
            size = 0;
        }

        void insert(const Object &obj) {
            if (size >= capacity) {
                return;
            }
            ptr[size] = obj;
            size++;
        }

        void remove(const Object &obj) {
            int i = 0;
            for(; i < size; i++) {
                if (ptr[i] == obj) {
                    break;
                }
            }
            if (i == size) {
                return;
            }
            size--;
            for(; i < size; i++) {
                ptr[i] = ptr[i + 1];
            }
        }

        bool contains(const Object &obj) const {
            for (int i = 0; i < size; i++) {
                if (ptr[i] == obj) {
                    return true;
                }
            }
            return false;
        }

        void printAll() const {
            for (int i = 0; i < size; i++) {
                cout << ptr[i] << ", ";
            }
            cout << endl;
        }

    private:
        Object *ptr;
        int size;
        int capacity;
};
