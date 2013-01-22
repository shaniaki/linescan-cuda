/*
 * types.hpp
 *
 *  Created on: Dec 11, 2012
 *      Author: shaniaki
 */

#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_

// run which version of the parallel cuda kernel //
#define KERNELVER compute_v1
//#define KERNELVER compute_v2
//#define KERNELVER compute_v3

// choose the input data set //
//#define DATASET 0
#define DATASET 1

#if DATASET==0
	#define WHITE_VALUE 255
	#define BUFFER_SIZE 24
	#define N 4
#elif DATASET==1
	#define WHITE_VALUE 255
	#define BUFFER_SIZE 24
	#define N 16
#endif

struct aoi
{
	int start;
	int end;
};


#endif /* GLOBALS_HPP_ */
