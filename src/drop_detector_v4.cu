/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cuda.h>
#include <assert.h>
#include <stdio.h>

#include "globals.hpp"

////////////////////////////////////////////////////////////////////////////////
// export the callable function
extern "C"
void compute_v4(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

__device__
void thread_allocator_v4(float* input_image,
						aoi* aoi_coordinates,
						unsigned int image_width,
						unsigned int sn,
						aoi* thread_state,
						float* image_parts)
{
	// thread allocator
	unsigned int ta = blockIdx.x;
	unsigned int thx = threadIdx.x;
	/// output decoding
	unsigned int offset = ta*(image_width/N);
	unsigned int local_end = thread_state->end - thread_state->start;
	if (thx <= local_end)
	{
		image_parts[sn*image_width + offset + thx] =
				input_image[sn*image_width + thread_state->start + thx];
	}
	if (thx == local_end + 1)
		image_parts[sn*image_width + offset + thx] = -1; // to terminate the image part
	/// next state
	__syncthreads();
	if (thx==0)
		if ((aoi_coordinates + sn * N + ta)->start != -1)
			*thread_state = aoi_coordinates[sn * N + ta];
	__syncthreads();
	/// output decoding
}

__device__
float preproc_image_v4(float* image_parts, unsigned int image_width)
{
	unsigned int noz = blockIdx.x;
	float imgPreproc2DDFA = 0;
	float* imgpre = image_parts + (image_width / N) * noz;
	while (*(imgpre) != -1) {
		imgPreproc2DDFA += WHITE_VALUE - *(imgpre);
		imgpre++;
	}
	return imgPreproc2DDFA;
}

__device__
void auto_correlate_v4(float imgPreproc2DDFA,
						int* parallelSW,
						unsigned int sn,
						float* ac_samples,
						int* ac_sw,
						int ac_ignore_it,
						float* ac_sampWin,
						float* autoCorrToCombSubMul)
{
	unsigned int noz = blockIdx.x;
	unsigned int thx = threadIdx.x;
	//// output decoding
	if (thx<2*(*ac_sw))
		ac_sampWin[thx] = ac_samples[ac_ignore_it + thx];
	if (thx==0)
		ac_sampWin[2*(*ac_sw)] = ac_samples[ac_ignore_it + 2*(*ac_sw)];
	__syncthreads();
	if (thx<2* (*ac_sw))
		for (int c = 0; c <= (*ac_sw) * 2; c++) {
			int d = c + (thx-(*ac_sw)) + 1;
			int k = (thx-(*ac_sw)) + (*ac_sw);
			if ((d >= 0) && (d < (*ac_sw) * 2))
				autoCorrToCombSubMul[(*ac_sw) * 2 - k - 1] += ac_sampWin[c] * ac_sampWin[d];
		}
	__syncthreads();
	//// next state
	int ac_temp = parallelSW[sn * N + noz];
	if (ac_temp != -1)
		(*ac_sw) = ac_temp;

	__shared__ float ac_samples_temp[BUFFER_SIZE];
	if (thx<BUFFER_SIZE)
		ac_samples_temp[thx] = ac_samples[thx+1];
	__syncthreads();
	if (thx<BUFFER_SIZE)
		ac_samples[thx] = ac_samples_temp[thx];
	if (thx==0)
		ac_samples[BUFFER_SIZE] = imgPreproc2DDFA;
}

__device__
void cross_correlate_v4(int ac_ignore_it,
						int* ac_sw,
						float* ac_sampWin,
						float* cc_coefs,
						float* parallelCoeffs,
						unsigned int sn,
						float* xCorrToCombSubMul)
{
	unsigned int noz = blockIdx.x;
	unsigned int thx = threadIdx.x;
	//// output decoding
	__syncthreads();
	if (thx<2*(*ac_sw))
		for (int c = 0; c <= (*ac_sw) * 2; c++) {
			int d = c + (thx-(*ac_sw)) + 1;
			int k = (thx-(*ac_sw)) + (*ac_sw);
			if ((d >= 0) && (d < (*ac_sw) * 2))
				xCorrToCombSubMul[(*ac_sw) * 2 - k - 1] += ac_sampWin[c]
				                                                        * cc_coefs[ac_ignore_it + d];
		}
	__syncthreads();
	//// next state
	float* cc_temp = parallelCoeffs + sn * N * BUFFER_SIZE
			+ noz * BUFFER_SIZE;
	if (*cc_temp != -1)
		if (thx<BUFFER_SIZE)
				cc_coefs[thx] = cc_temp[thx];
}

__device__
void submul_v4(float* combSubMulToCombAvgSub,
			float* xCorrToCombSubMul,
			float* autoCorrToCombSubMul,
			unsigned int win_size)
{
	int i=threadIdx.x;
	if (i < win_size)
		combSubMulToCombAvgSub[i] = (xCorrToCombSubMul[i]
		                           - autoCorrToCombSubMul[i]) / autoCorrToCombSubMul[i];
}

__device__
void avgsub_v4(float* combSubMulToCombAvgSub,
			float* combAvgSubtoOutBlock,
			unsigned int win_size)
{
	int thx=threadIdx.x;
	__shared__ float as_average;
	if (thx==0)
	{
		as_average = 0;
		for (int i = 0; i < win_size; i++)
			as_average += combSubMulToCombAvgSub[i];
		as_average /= win_size;
	}
	__syncthreads();
	if (thx < win_size)
		combAvgSubtoOutBlock[thx] = combSubMulToCombAvgSub[thx] - as_average;
}

__device__
void out_block_v4(float* reference,
					unsigned int sn,
					float* out_buffer,
					float* combAvgSubtoOutBlock,
					unsigned int win_size)
{
	unsigned int noz = blockIdx.x;
	unsigned int thx = threadIdx.x;
	// output
	/// output decoding
	reference[sn * N + noz] = out_buffer[0];
	/// next state
	__shared__ float out_buffer_temp[BUFFER_SIZE];
	if (thx<BUFFER_SIZE)
			out_buffer_temp[thx] = out_buffer[thx+1];
	__syncthreads();
	if (thx<BUFFER_SIZE)
		out_buffer[thx] = out_buffer_temp[thx];
	out_buffer[BUFFER_SIZE] = 0;
	unsigned int out_ignore_it = (BUFFER_SIZE - win_size)
			/ 2;
	if (thx < BUFFER_SIZE - (2 * out_ignore_it))
		out_buffer_temp[thx + out_ignore_it] = out_buffer[thx + out_ignore_it]
				+ combAvgSubtoOutBlock[thx];
	__syncthreads();
	if (thx < BUFFER_SIZE - (2 * out_ignore_it))
		out_buffer[thx+ out_ignore_it] = out_buffer_temp[thx+ out_ignore_it];
}

__global__
void computeThreadAllocator_v4(float* input_image,
							aoi* aoi_coordinates,
							unsigned int image_height,
							unsigned int image_width,
							float* image_parts,
							aoi* g_thread_state
)
{
	unsigned int noz = blockIdx.x;
	unsigned int thx = threadIdx.x;

	// state variables
	/// thread allocator
	__shared__ aoi thread_state;

	//// initialization
	if (thx==0)
		thread_state = g_thread_state[noz];
	__syncthreads();

	for (unsigned int sn=0;sn<image_height;sn++)
	{
		// thread allocator
		thread_allocator_v4(input_image, aoi_coordinates, image_width, sn, &thread_state, image_parts);
	}
	if (thx==0)
			g_thread_state[noz] = thread_state;
	__syncthreads();
}

__global__
void computeThread_v4(float* reference,
						float* parallelCoeffs,
						int* parallelSW,
						unsigned int image_height,
						unsigned int image_width,
						float* image_parts,
						float* g_ac_samples,
						int* g_ac_sw,
						float* g_cc_coefs,
						float* g_out_buffer
)
{
	unsigned int noz = blockIdx.x;
	unsigned int thx = threadIdx.x;

	// state variables
	/// auto correlation
	__shared__ float ac_samples[BUFFER_SIZE + 1];
	__shared__ int ac_sw;
	/// cross correlation
	__shared__ float cc_coefs[BUFFER_SIZE + 1];
	/// output block
	__shared__ float out_buffer[BUFFER_SIZE + 1];
	//// initialization
	if (thx==0)
		ac_sw = g_ac_sw[noz];
	if (thx<BUFFER_SIZE+1)
	{
		ac_samples[thx] = g_ac_samples[noz*(BUFFER_SIZE+1) + thx];
		cc_coefs  [thx] = g_cc_coefs  [noz*(BUFFER_SIZE+1) + thx];
		out_buffer[thx] = g_out_buffer[noz*(BUFFER_SIZE+1) + thx];
	}

	// inter-block communication
	__shared__ int ac_ignore_it;
	__shared__ float imgPreproc2DDFA;
	__shared__ float autoCorrToCombSubMul[BUFFER_SIZE];
	__shared__ float ac_sampWin[(BUFFER_SIZE*2+1)];
	__shared__ float xCorrToCombSubMul[BUFFER_SIZE];
	__shared__ float combSubMulToCombAvgSub[BUFFER_SIZE];
	__shared__ float combAvgSubtoOutBlock[BUFFER_SIZE];

	for (unsigned int sn=0;sn<image_height;sn++)
	{
		// for all cuda blocks (nozzles):
		// pre-process image: inv and reduce
		imgPreproc2DDFA = preproc_image_v4(image_parts+sn*image_width, image_width);

		// single DDFA
		/// auto correlation
		if (thx<BUFFER_SIZE)
		{
			autoCorrToCombSubMul[thx] = 0;
			ac_sampWin[thx] = 0;
			ac_sampWin[BUFFER_SIZE+thx] = 0;
		}
		if (thx==0)
		{
			ac_ignore_it = BUFFER_SIZE / 2 - ac_sw;
			ac_sampWin[2*BUFFER_SIZE] = 0;
		}

		__syncthreads();
		auto_correlate_v4(imgPreproc2DDFA, parallelSW, sn, ac_samples, &ac_sw, ac_ignore_it, ac_sampWin, autoCorrToCombSubMul);

		/// cross correlation
		/// note: we use the ac_samples, ac_ignore_it, ac_sampWin and ac_sw from the auto correlation stage
		if (thx<BUFFER_SIZE)
			xCorrToCombSubMul[thx] = 0;
		__syncthreads();
		//// output decoding
		cross_correlate_v4(ac_ignore_it, &ac_sw, ac_sampWin, cc_coefs, parallelCoeffs, sn, xCorrToCombSubMul);

		// subtract and multiply ((x-y)/y)
		submul_v4(combSubMulToCombAvgSub, xCorrToCombSubMul, autoCorrToCombSubMul, ac_sw * 2);

		// average and subtract
		avgsub_v4(combSubMulToCombAvgSub, combAvgSubtoOutBlock, ac_sw * 2);

		// output
		/// output decoding
		out_block_v4(reference, sn, out_buffer, combAvgSubtoOutBlock, ac_sw * 2);
	}
	// write back the state variables to the global space
	if (thx==0)
		g_ac_sw[noz] = ac_sw;
	if (thx<BUFFER_SIZE+1)
	{
		g_ac_samples[noz*(BUFFER_SIZE+1) + thx] = ac_samples[thx];
		g_cc_coefs  [noz*(BUFFER_SIZE+1) + thx] = cc_coefs  [thx];
		g_out_buffer[noz*(BUFFER_SIZE+1) + thx] = out_buffer[thx];
	}
	__syncthreads();
}

void compute_v4(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width)
{
	// allocate/transfer data on/to to the device
	float* d_reference;
	assert(cudaSuccess == cudaMalloc((void **) &d_reference, N*CHUNK_LINES * sizeof(float)));

	float* d_input_image;
	assert(cudaSuccess == cudaMalloc((void **) &d_input_image, image_width*CHUNK_LINES * sizeof(float)));

	aoi* d_aoi_coordinates;
	assert(cudaSuccess == cudaMalloc((void **) &d_aoi_coordinates, N*CHUNK_LINES * sizeof(aoi)));

	float* d_parallelCoeffs;
	assert(cudaSuccess == cudaMalloc((void **) &d_parallelCoeffs, BUFFER_SIZE*N*CHUNK_LINES * sizeof(float)));

	int* d_parallelSW;
	assert(cudaSuccess == cudaMalloc((void **) &d_parallelSW, N*CHUNK_LINES * sizeof(int)));

	float* image_parts;
	assert(cudaSuccess == cudaMalloc((void **) &image_parts, image_width*CHUNK_LINES * sizeof(float)));

	// state variables
	/// thread allocator
	aoi* thread_state;
	assert(cudaSuccess == cudaMalloc((void **) &thread_state, N * sizeof(aoi)));
	assert(cudaSuccess == cudaMemset((void*)thread_state, 0, N * sizeof(aoi)));

	/// image preprocessor
	/// auto correlation
	float* ac_samples;
	assert(cudaSuccess == cudaMalloc((void **) &ac_samples, N * (BUFFER_SIZE + 1) * sizeof(float)));
	assert(cudaSuccess == cudaMemset((void*)ac_samples, 0, N * (BUFFER_SIZE + 1) * sizeof(float)));
	int* ac_sw;
	assert(cudaSuccess == cudaMalloc((void **) &ac_sw, N * sizeof(int)));
	assert(cudaSuccess == cudaMemset((void*)ac_sw, 0, N * sizeof(int)));
	/// cross correlation
	float* cc_coefs;
	assert(cudaSuccess == cudaMalloc((void **) &cc_coefs, N * (BUFFER_SIZE + 1) * sizeof(float)));
	assert(cudaSuccess == cudaMemset((void*)cc_coefs, 0, N * (BUFFER_SIZE + 1) * sizeof(float)));
	/// output block
	float* out_buffer;
	assert(cudaSuccess == cudaMalloc((void **) &out_buffer, N * (BUFFER_SIZE + 1) * sizeof(float)));
	assert(cudaSuccess == cudaMemset((void*)out_buffer, 0, N * (BUFFER_SIZE + 1) * sizeof(float)));

	// computation
	for (int i=0;i<image_height;i+=CHUNK_LINES)
	{
		// number of lines to be processes in this iteration
		unsigned int num_lines = min( CHUNK_LINES, image_height-i);
		cudaMemcpy(d_input_image, input_image + i*image_width, image_width*num_lines * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_aoi_coordinates, aoi_coordinates + i*N, N*num_lines * sizeof(aoi), cudaMemcpyHostToDevice);
		cudaMemcpy(d_parallelCoeffs, parallelCoeffs + i*BUFFER_SIZE*N, BUFFER_SIZE*N*num_lines * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_parallelSW, parallelSW + i*N, N*num_lines * sizeof(float), cudaMemcpyHostToDevice);

		unsigned int cuda_threads = ((image_width/N)/32+1)*32;
		computeThreadAllocator_v4<<<N,cuda_threads>>>(d_input_image, d_aoi_coordinates, num_lines, image_width, image_parts, thread_state);
		if ( cudaSuccess != cudaGetLastError() )
			printf( "Error in kernel call!\n" );

		cuda_threads = ((BUFFER_SIZE+1)/32+1)*32;
		computeThread_v4<<<N,cuda_threads>>>(d_reference, d_parallelCoeffs, d_parallelSW, num_lines, image_width, image_parts,
											ac_samples, ac_sw, cc_coefs, out_buffer);
		if ( cudaSuccess != cudaGetLastError() )
			printf( "Error in kernel call!\n" );

		assert(cudaSuccess == cudaMemcpy(reference + i*N, d_reference, N*num_lines * sizeof(float), cudaMemcpyDeviceToHost));
	}

	cudaFree(image_parts);

	cudaFree(d_parallelSW);
	cudaFree(d_parallelCoeffs);
	cudaFree(d_aoi_coordinates);
	cudaFree(d_input_image);
	cudaFree(d_reference);
}

