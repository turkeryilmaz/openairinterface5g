#ifndef SEQ_CONTAINER_ARRAY
#define SEQ_CONTAINER_ARRAY

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>


/*
 * Dynamic array data structure that shrinks and expands as needed, similar to std::vector
 */

typedef struct seq_arr_s{
    uint8_t* data; 
    size_t size;
    const size_t elt_size;
    size_t cap;
} seq_arr_t;

typedef void (*seq_free_func)(void*); 

/*
 * Init the seq_array ds
 * @param arr: the ds to initialize 
 * @param sz: Size of the data type i.e., sizeof(T), that the seq_array_t will store. 
*/
void seq_arr_init(seq_arr_t* arr, size_t sz);

/*
 * Free the seq_array ds
 * @param arr: the ds  
 * @param seq_free_func: Function that every element will call before being freed e.g., usefull to free 
 *                       "deep" objects.
 * */
void seq_arr_free(seq_arr_t*, seq_free_func );

/*
 * Push back 
 * @param arr: the ds 
 * @param data: Pointer to the data that will ve copied. This a swallow copy i.e., memcpy
 * @param len: Lenght of the data to be stored i.e.,sizeof(T). This parameter is only used
 *              to guarantee that the object copied and the underlying array initialized at
 *              seq_arr_init have the same lenght
 * */
void seq_arr_push_back(seq_arr_t* arr, void* data, size_t len);


/*
 * Erase a half-open range [start_it, end_it)  
 * @param arr: the ds 
 * @param start_it Starting iterator of the range 
 * @param end_it Ending iterator of the range
 * @pre Precondition: the range is valid and @p end_it can be reached from @p start_it
 */
void seq_arr_erase(seq_arr_t*, void* start_it , void* end_it);

/*
 * Size   
 * @param arr: the ds 
 * @return the number of elements in the ds s
 * */

size_t seq_arr_size(seq_arr_t*);

/*
 * Front
 * @param arr: the ds
 * @return the iterator to the first element
 * */

void* seq_arr_front(seq_arr_t*);

/*
 * Next 
 * @param arr the ds 
 * @param it Iterator  
 * @return Iterator to the next element in the ds 
 * @pre Precondition: iterator @p it, is valid and not equal to the end iterator
 * */

void* seq_arr_next(seq_arr_t*, void* it);

/*
 * End 
 * @param arr: the ds 
 * @return one past the last element in the ds 
 * */

void* seq_arr_end(seq_arr_t*);

/*
 * End 
 * @param arr: the ds 
 * @param nth: Position of the element 
 * @return the iterator the nth element 
 * Precondition: @p nth is not greater than seq_arr_size()
*/

void* seq_arr_at(seq_arr_t*, uint32_t nth);

/*
 * Distance 
 * @param arr: the ds 
 * @param it_a Iterator 
 * @param it_b Iterator 
 * @return the of the amount of elements between it_a and it_b 
 * @pre Precondition: @p it_a and @p it_b are valid and @p it_a <  @p it_b 
*/

ptrdiff_t seq_arr_dist(seq_arr_t* arr, void* it_a, void* it_b);

/*
 * Equality 
 * @param arr: the ds 
 * @param it_a: iterator from the seq_array ds
 * @param it_b: iterator from the seq_array ds
 * @return true if both iterators point to the same element 
 * @pre Precondition: @p it_a and @p it_b are valid iterators 
*/

bool seq_arr_equal(seq_arr_t* arr, void* it_a ,void* it_b);


#endif

