#include "list.hpp"

// Separating the declaration and definition of template functions
// requires you to explicitly instantiate the template for each
// type you require. See the lines like:
//      template void List<int>::init();

// template <typename Object>
// void List<Object>::init()
// {
//     theSize = 0;
//     head = new Node;
//     tail = new Node;
//     head->next = tail;
//     tail->prev = head;
// }
// template void List<int>::init();

// template <typename Object>
// List<Object>::List()
// {
//     init();
// }
// template List<int>::List();

// template <typename Object>
// List<Object>::~List()
// {
//     // clear();
//     delete head;
//     delete tail;
// }
// template List<int>::~List();
