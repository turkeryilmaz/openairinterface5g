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

#ifndef BIDIRECTIONAL_HASH_MAP
#define BIDIRECTIONAL_HASH_MAP

/*
 *  Naive bidirectional map. K1->K2 and K2->K1
 *  It is implemented with 2 RB trees 
 *  https://en.wikipedia.org/wiki/Bidirectional_map
 */

#include "assoc_rb_tree.h"

typedef int (*bi_map_cmp)(void const*, void const*);

typedef void (*free_func_t)(void* key, void* value);

typedef struct {
  assoc_rb_tree_t left;
  assoc_rb_tree_t right;
} bi_map_t;

typedef struct {
  void* it;
} bml_iter_t;

typedef struct {
  void* it;
} bmr_iter_t;

/*
 * Init the bimap ds
 * @param map the ds to initialize 
 * @param key_sz_1: Size of the first key i.e., sizeof(T), that the rb_tree will store. 
 * @param key_sz_2: Size of the second key i.e., sizeof(T), that the rb_tree will store. 
 * @param comp1 Compare function in the left RB tree. 
 * @param comp2 Compare function in the right RB tree. 
 * @param free1: Free function to free the left key 
 * @param free2: Free function to free the right key 
*/

void bi_map_init(bi_map_t* map, size_t key_sz_1, size_t key_sz_2, bi_map_cmp cmp1, bi_map_cmp cmp2, free_func_t free1, free_func_t free2);

/*
 * Free the bimap ds
 * @param map the ds 
 */
void bi_map_free(bi_map_t* map);

// Modifiers

/*
 * Insert K1 and K2 in the bimap ds
 * @param map the ds 
 * @param key_1: key1 pointer of the first key 
 * @param key_2: key2 pointer of the second key 
 * @param key_sz_1: Size of the first key i.e., sizeof(T) 
 * @param key_sz_2: Size of the second key i.e., sizeof(T) 
*/
void bi_map_insert(bi_map_t* map, void const* key1, size_t key_sz1, void const* key2, size_t key_sz2);

/*
 * It returns the void* of key2. the void* of the key1 is freed 
 * @param map the ds 
 * @param key_1: key1 pointer of the first key 
 * @param key_sz_1: Size of the first key i.e., sizeof(T) 
 * @return key2
*/

void* bi_map_extract_left(bi_map_t* map, void* key1, size_t key1_sz);


/*
 * It returns the void* of key1. the void* of the key2 is freed 
 * @param map the ds 
 * @param key_2: key2 pointer of the second key 
 * @param key_sz_2: Size of the first key i.e., sizeof(T) 
 * @return key1
*/

void* bi_map_extract_right(bi_map_t* map, void* key2, size_t key1_sz);

/*
 * It returns the void* of key2. the void* of the key1 is not freed 
 * @param map the ds 
 * @param bml_iter_t bimap left iterator 
 * @return key2
*/

void* bi_map_value_left(bi_map_t* map, bml_iter_t it);

/*
 * It returns the void* of key1. the void* of the key2 is not freed 
 * @param map the ds 
 * @param bml_iter_t valid bimap iterator 
 * @return key1
*/
void* bi_map_value_right(bi_map_t* map, bml_iter_t it);

// Capacity

/*
 * Number of elements in the ds 
 * @param map the ds 
 * @return number of elements in the ds
*/

size_t bi_map_size(bi_map_t* map);

// Forward Iterator Concept

/*
 * Front left Iterator front of the left rb tree 
 * @param map the ds 
 * @return iterator to the front left rb tree 
*/
bml_iter_t bi_map_front_left(bi_map_t* map);

/*
 * Next Iterator in the left rb tree 
 * @param map the ds 
 * @param it iterator to the next element from the left rb tree 
 * @return iterator to the next iterator in the left rb tree 
*/
bml_iter_t bi_map_next_left(bi_map_t* map, bml_iter_t it);

/*
 * End Iterator in the left rb tree 
 * @param map the ds 
 * @return iterator to the one past the end iterator in the left rb tree 
*/
bml_iter_t bi_map_end_left(bi_map_t* map);

/*
 * Front right Iterator front of the right rb tree 
 * @param map the ds 
 * @return iterator to the front right rb tree 
*/
bmr_iter_t bi_map_front_right(bi_map_t* map);

/*
 * Next Iterator in the right rb tree 
 * @param map the ds 
 * @param it iterator to the next element from the right rb tree 
 * @return iterator to the next iterator in the right rb tree 
*/
bmr_iter_t bi_map_next_right(bi_map_t* map, bmr_iter_t);

/*
 * End Iterator in the right rb tree 
 * @param map the ds 
 * @return iterator to the one past the end iterator in the right rb tree 
*/
bmr_iter_t bi_map_end_right(bi_map_t* map);

#endif
