/*
MIT License

Copyright (c) 2021 Mikel Irazabal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef ASSOCIATIVE_RB_TREE_NAIVE
#define ASSOCIATIVE_RB_TREE_NAIVE

/*
    RB tree implemented a la CLRS

    Red-Black trees have 5 properties:
    1- Every node is Red or Black
    2- The root is black
    3- Every leaf is black
    4- If a node is red, then both its children are black
    5- For each node, all simple paths from the node to descendant leaves contain the same number of black nodes

    Lemma 1: a red-black tree with n internal nodes has a height at most 2lg(n+1)
*/

#include <stdbool.h>
#include <stddef.h>

typedef struct assoc_node_s assoc_node_t;

typedef void (*free_func_t)(void* key, void* value);

typedef struct {
  assoc_node_t* dummy; // node
  assoc_node_t* root; // node
  int (*comp)(const void*, const void*);
  size_t size;
  // size key
  size_t key_sz;
  free_func_t free_func;
  // size value
  // size_t val_sz; // Value is always a pointer to the type

} assoc_rb_tree_t;



/*
 * Init the rb ds
 * @param tree: the ds to initialize 
 * @param key_sz: Size of the key i.e., sizeof(T), that the rb_tree will store. 
 * @param comp: Compare function in the RB tree. 
 * @param t: Free function to free the value if needed 
*/
void assoc_rb_tree_init(assoc_rb_tree_t* tree, size_t key_sz, int (*comp)(const void*, const void*), free_func_t f);

/*
 * Free the rb ds
 * @param tree: the ds
*/
void assoc_rb_tree_free(assoc_rb_tree_t* tree);

// Modifiers
// The tree is responsible for freeing the void* key and value memory later

/*
 * Insert key-value 
 * @param tree: the ds
 * @param key a pointer to the key
 * @param key_sz  sizeof(key) 
 * @param value Pointer to the value. Ownership is transfere i.e., the RB tree will
 *              free it. It needs to be allocated in the heap 
*/
void assoc_rb_tree_insert(assoc_rb_tree_t* tree, void const* key, size_t key_sz, void* value);

/*
 * Extract value It returns the void* of value. the void* of the key is freed
 * @param tree: the ds
 * @param key a pointer to the key
 * @return value 
 * @pre @p key must exist in the ds
*/
void* assoc_rb_tree_extract(assoc_rb_tree_t* tree, void* key);


/*
 * Get the key from an iterator 
 * @param tree: the ds
 * @param it a pointer to the key
 * @return key from the ds. It does not free it. 
 * @pre @p it must exist in the ds
*/
void* assoc_rb_tree_key(assoc_rb_tree_t* tree, void* it);


/*
 * Get the value pointer form an iterator
 * @param tree: the ds
 * @param it a pointer to the key
 * @return key from the ds. It does not free it. 
 * @pre @p it must exist in the ds
*/
void* assoc_rb_tree_value(assoc_rb_tree_t* tree, void* it);

/*
 * Size of the current ds 
 * @param tree: the ds
 * @return number of elements in the ds 
*/
size_t assoc_rb_tree_size(assoc_rb_tree_t* tree);

// Forward Iterator Concept

/*
 * Get the iterator to the first element 
 * @param tree: the ds
 * @return iterator to the first element 
*/
void* assoc_rb_tree_front(assoc_rb_tree_t const* tree);


/*
 * Get the next iterator  
 * @param tree: the ds
 * @param it iterator 
 * @return iterator to the next element 
 * @pre iterator must be in the ds 
*/
void* assoc_rb_tree_next(assoc_rb_tree_t const* tree, void* it);


/*
 * Get the last iterator 
 * @param tree: the ds
 * @return iterator to one past the last element in the ds 
*/
void* assoc_rb_tree_end(assoc_rb_tree_t const* tree);

#endif

