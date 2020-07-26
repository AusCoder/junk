#ifndef _LIST_HPP
#define _LIST_HPP

#include <utility>
#include <iostream>

using namespace std;

template <typename Object>
class List
{
    private:
        struct Node
        {
            Object data;
            Node *prev;
            Node *next;

            Node(const Object &d = Object{}, Node *p = nullptr, Node *n = nullptr)
                : data{d}, prev{p}, next{n} {}

            Node(Object &&d, Node *p = nullptr, Node *n = nullptr)
                : data{std::move(d)}, prev{p}, next{n} {}
        };

    public:
        class const_iterator
        {
            public:
                const_iterator(): current{nullptr} {}

                const Object &operator*() const
                {
                    return retrieve();
                }

                const_iterator &operator++()
                {
                    current = current->next;
                    return *this;
                }

                const_iterator &operator++(int)
                {
                    const_iterator old = *this;
                    ++(*this);
                    return old;
                }

                bool operator==(const const_iterator &rhs) const
                {
                    return current == rhs.current;
                }
                bool operator!=(const const_iterator &rhs) const
                {
                    return !(*this == rhs);
                }

            protected:
                Node *current;

                Object &retrieve() const
                {
                    return current->data;
                }

                const_iterator(Node *p): current{p} {}

                friend class List<Object>;
        };

        class iterator: public const_iterator
        {
            public:
                iterator() {}

                Object &operator*()
                {
                    return const_iterator::retrieve();
                }

                const Object &operator*() const
                {
                    return const_iterator::operator*();
                }

            protected:
                iterator(Node *p): const_iterator{p} {}

                friend class List<Object>;
        };

    public:
        List()
        {
            cout << "default cstor" << endl;
            init();
        }
        ~List()
        {
            cout << "dstor" << endl;
            clear();
            delete head;
            delete tail;
        }
        List(const List &rhs)
        {
            cout << "copy cstor" << endl;
            init();
            for (auto &x: rhs)
            {
                push_back(x);
            }
        }
        List &operator= (const List &rhs)
        {
            cout << "assignment operator" << endl;
            List copy = rhs;
            // Q: What does this do?
            // A: it takes move params and moves from copy to this
            std::swap(*this, copy);
            return *this;
        }
        List(List &&rhs): theSize{rhs.theSize}, head{rhs.head}, tail{rhs.tail}
        {
            cout << "move cstor" << endl;
            rhs.theSize = 0;
            rhs.head = nullptr;
            rhs.tail = nullptr;
        }
        List &operator=(List &&rhs)
        {
            cout << "assignment move operator" << endl;
            // Q: is this correct?
            // std::swap(*this, rhs);

            std::swap(theSize, rhs.theSize);
            std::swap(head, rhs.head);
            std::swap(tail, rhs.tail);
            return *this;
        }

        iterator begin()
        {
            return {head->next};
        }
        const_iterator begin() const
        {
            return {head->next};
        }
        iterator end()
        {
            return {tail};
        }
        const_iterator end() const
        {
            return {tail};
        }

        int size() const
        {
            return theSize;
        }
        bool empty() const
        {
            return size() == 0;
        }

        Object &front()
        {
            return *begin();
        }
        const Object &front() const
        {
            return *begin();
        }
        Object &back()
        {
            return *--end();
        }
        const Object &back() const
        {
            return *--end();
        }
        void push_front(const Object &x)
        {
            insert(begin(), x);
        }
        void push_front(Object &&x)
        {
            insert(begin(), std::move(x));
        }
        void push_back(const Object &x)
        {
            insert(end(), x);
        }
        void push_back(Object &&x)
        {
            insert(end(), std::move(x));
        }
        void pop_front()
        {
            eraise(begin());
        }
        void pop_back()
        {
            eraise(--end());
        }

        void clear()
        {
            while (!empty())
            {
                pop_front();
            }
        }

        iterator insert(iterator itr, const Object &x)
        {
            Node *p = itr.current;
            Node *n = new Node{x, p->prev, p};
            p->prev->next = n;
            p->prev = n;
            theSize++;
            return {n};
        }
        iterator insert(iterator itr, Object &&x)
        {
            Node *p = itr.current;
            Node *n = new Node(std::move(x), p->prev, p);
            p->prev->next = n;
            p->prev = n;
            theSize++;
            return {n};
        }

        iterator eraise(iterator itr)
        {
            Node *p = itr.current;
            iterator retVal{p->next};
            p->prev->next = p->next;
            p->next->prev = p->prev;
            theSize--;
            delete p;
            return retVal;
        }
        iterator eraise(iterator from, iterator to)
        {
            for(iterator itr = from; itr != to;)
            {
                eraise(itr);
            }
            return to;
        }

    private:
        int theSize;
        Node *head;
        Node *tail;

        void init()
        {
            theSize = 0;
            head = new Node;
            tail = new Node;
            head->next = tail;
            tail->prev = head;
        }
};

#endif
