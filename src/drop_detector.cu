////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "globals.hpp"


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
float* readTiff(char* filename, unsigned int* w, unsigned int* h);

extern "C"
aoi* readAOIs(char* filename, unsigned int image_height, unsigned int num_threads);

extern "C"
float* readCoefs(char* filename, unsigned int image_height, unsigned int num_threads, unsigned int coefs_size);

extern "C"
int* readSWs(char* filename, unsigned int image_height, unsigned int num_threads);


extern "C"
void computeGold(float* reference,
				float* h_image_input,
				aoi* h_aoi_input,
				float* h_coeff_input,
				int* h_sw_input,
				unsigned int image_height,
				unsigned int image_width);

extern "C"
void compute_v1(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

extern "C"
void compute_v2(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

extern "C"
void compute_v3(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

/*
////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float *g_idata, float *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}
*/

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //int devID = findCudaDevice(argc, (const char **) (argv));

    unsigned int num_threads = N;
    float* h_coeff_input;
    float* h_image_input;
    int* h_sw_input;
    aoi* h_aoi_input;
    float *reference, *d_reference;
    unsigned int image_width, image_height;

    // read the input from the TIFF sizeof(aois) = inp_rows*image_row_width
    char input_name[20];
    sprintf(input_name,"%s/%d_input.tif",argv[1],DATASET);
	h_image_input = readTiff(input_name, &image_width, &image_height);

	// read the AOIs from the file sizeof(aois) = inp_rows*num_threads
	char aoi_name[20];
	sprintf(aoi_name,"%s/%d_aois.txt",argv[1],DATASET);
	h_aoi_input = readAOIs(aoi_name, image_height, num_threads);

	// read the coefs from the file sizeof(coefs) = inp_rows*num_threads*buffer_size
	char coefs_name[20];
	sprintf(coefs_name,"%s/%d_coefs.txt",argv[1],DATASET);
	h_coeff_input = readCoefs(coefs_name, image_height, num_threads, BUFFER_SIZE);

	// read the SWs from the file sizeof(sw) = inp_rows*num_threads
	char sws_name[20];
	sprintf(sws_name,"%s/%d_sws.txt",argv[1],DATASET);
	h_sw_input = readSWs(sws_name, image_height, num_threads);

	d_reference = (float*)malloc(image_height*num_threads*sizeof(float));

	double kernel_time=0;

	for (int i=0;i<=NUM_RUNS;i++)
	{
		struct timeval timerStart;
		gettimeofday(&timerStart, NULL);

		KERNELVER(d_reference, h_image_input, h_aoi_input, h_coeff_input, h_sw_input, image_height, image_width);

		struct timeval timerStop, timerElapsed;
		gettimeofday(&timerStop, NULL);
		timersub(&timerStop, &timerStart, &timerElapsed);
		if (i>0)
			kernel_time += timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
	}

	kernel_time /= NUM_RUNS;
	printf("Processing time: %f (ms)\n", kernel_time);

	// compute reference solution
	reference = (float*)malloc(image_height*num_threads*sizeof(float));
	computeGold(reference, h_image_input, h_aoi_input, h_coeff_input, h_sw_input, image_height, image_width);

	// check result
	for (int i=0;i<image_height;i++)
	{
		for (int j=0;j<N;j++)
		{
			if (abs(reference[i*N+j] - d_reference[i*N+j]) > 0.0001)
				printf("Error at image line %d for nozzle %d: reference=%f, calculated=%f\n", i, j, reference[i*N+j], d_reference[i*N+j]);
			//printf("%f ",d_reference[i*N+j]);
		}
		//printf("\n");
	}

    // cleanup memory
    free(h_image_input);
	free(h_aoi_input);
	free(h_coeff_input);
	free(h_sw_input);
	free(reference);
	free(d_reference);

    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
