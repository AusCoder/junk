/*
a binary sequence is alternating if adjacent entries differ
eg 010, 10101

a type 1 transform takes a digit from the front and puts it at the back
a type 2 transform swaps the digit at any index i

Given a binary sequence, find the minimum number of type 2 transforms to make it alternating.

111000 -> 101010  (2 type-2s)
111000 -> 110001 -> 010101 (2 type-2s)
001 -> 010 (0 type-2s) (only 1 type-1)
*/
