/*
 * types.hpp
 *
 *  Created on: Dec 11, 2012
 *      Author: shaniaki
 */

#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_

// used for time measurement
#define NUM_RUNS 10
// internal buffer sizes
#define BUFFER_SIZE 24
// used in v4
#define CHUNK_LINES 32

// run which version of the parallel cuda kernel //
//#define KERNELVER compute_v1
//#define KERNELVER compute_v2
//#define KERNELVER compute_v3
#define KERNELVER compute_v4

// choose the input data set //
//#define DATASET 0
//#define DATASET 1
//#define DATASET 3
#define DATASET 4

#if DATASET==0
	#define WHITE_VALUE 255
	#define N 4
#elif DATASET==1
	#define WHITE_VALUE 255
	#define N 16
#elif DATASET==3 || DATASET==4
	#define WHITE_VALUE 255
	#define N 510
#endif

struct aoi
{
	int start;
	int end;
};


#endif /* GLOBALS_HPP_ */
