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
void compute_v1(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

__device__
void thread_allocator_v1(float* input_image,
						aoi* aoi_coordinates,
						unsigned int image_width,
						unsigned int sn,
						aoi* thread_state,
						float* image_parts)
{
	// thread allocator
	unsigned int ta = blockIdx.x;
	/// output decoding
	int index = 0;
	for (int taod = (thread_state + ta)->start;
			taod <= (thread_state + ta)->end; taod++) {
		image_parts[(image_width / N) * ta + index] =
				input_image[(sn * image_width) + taod];
		index++;
	}
	image_parts[(image_width / N) * ta + index] = -1; // to terminate the image part
	/// next state
	if ((aoi_coordinates + sn * N + ta)->start != -1)
		thread_state[ta] = aoi_coordinates[sn * N + ta];
}

__device__
float preproc_image_v1(float* image_parts, unsigned int image_width)
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
void auto_correlate_v1(float imgPreproc2DDFA,
						int* parallelSW,
						unsigned int sn,
						float* ac_samples,
						int* ac_sw,
						int ac_ignore_it,
						float* ac_sampWin,
						float* autoCorrToCombSubMul)
{
	unsigned int noz = blockIdx.x;
	//// output decoding
	for (int i=0;i<2*ac_sw[noz]+1;i++)
			ac_sampWin[i] = ac_samples[noz*(BUFFER_SIZE+1) + ac_ignore_it + i];
	for (int b = -ac_sw[noz]; b < ac_sw[noz]; b++)
		for (int c = 0; c <= ac_sw[noz] * 2; c++) {
			int d = c + b + 1;
			int k = b + ac_sw[noz];
			if ((d >= 0) && (d < ac_sw[noz] * 2))
				autoCorrToCombSubMul[ac_sw[noz] * 2 - k - 1] += ac_sampWin[c]
						* ac_sampWin[d];
		}
	//// next state
	int ac_temp = parallelSW[sn * N + noz];
	if (ac_temp != -1)
		ac_sw[noz] = ac_temp;

	for (int i=0;i<BUFFER_SIZE;i++)
		ac_samples[noz*(BUFFER_SIZE+1)+i] = ac_samples[noz*(BUFFER_SIZE+1)+(i+1)];
	ac_samples[noz*(BUFFER_SIZE+1) + BUFFER_SIZE] = imgPreproc2DDFA;
}

__device__
void cross_correlate_v1(int ac_ignore_it,
						int* ac_sw,
						float* ac_sampWin,
						float* cc_coefs,
						float* parallelCoeffs,
						unsigned int sn,
						float* xCorrToCombSubMul)
{
	unsigned int noz = blockIdx.x;
	//// output decoding
	for (int b = -ac_sw[noz]; b < ac_sw[noz]; b++)
		for (int c = 0; c <= ac_sw[noz] * 2; c++) {
			int d = c + b + 1;
			int k = b + ac_sw[noz];
			if ((d >= 0) && (d < ac_sw[noz] * 2))
				xCorrToCombSubMul[ac_sw[noz] * 2 - k - 1] += ac_sampWin[c]
				                                                        * cc_coefs[noz*(BUFFER_SIZE+1) + ac_ignore_it + d];
		}
	//// next state
	float* cc_temp = parallelCoeffs + sn * N * BUFFER_SIZE
			+ noz * BUFFER_SIZE;
	if (*cc_temp != -1)
		for (int i=0;i<BUFFER_SIZE;i++)
				cc_coefs[noz*(BUFFER_SIZE+1)+i] = cc_temp[i];
}

__device__
void submul_v1(float* combSubMulToCombAvgSub,
			float* xCorrToCombSubMul,
			float* autoCorrToCombSubMul,
			unsigned int win_size)
{
	for (int i = 0; i < win_size; i++)
		combSubMulToCombAvgSub[i] = (xCorrToCombSubMul[i]
		                           - autoCorrToCombSubMul[i]) / autoCorrToCombSubMul[i];
}

__device__
void avgsub_v1(float* combSubMulToCombAvgSub,
			float* combAvgSubtoOutBlock,
			unsigned int win_size)
{
	float as_average = 0;
	for (int i = 0; i < win_size; i++)
		as_average += combSubMulToCombAvgSub[i];
	as_average /= win_size;
	for (int i = 0; i < win_size; i++)
		combAvgSubtoOutBlock[i] = combSubMulToCombAvgSub[i] - as_average;
}

__device__
void out_block_v1(float* reference,
					unsigned int sn,
					float* out_buffer,
					float* combAvgSubtoOutBlock,
					unsigned int win_size)
{
	unsigned int noz = blockIdx.x;
	// output
	/// output decoding
	reference[sn * N + noz] = out_buffer[noz*(BUFFER_SIZE+1)];
	//printf("%f ", out_buffer[noz*(BUFFER_SIZE+1)]);
	/// next state
	for (int i=0;i<BUFFER_SIZE;i++)
			out_buffer[noz*(BUFFER_SIZE+1)+i] = out_buffer[noz*(BUFFER_SIZE+1)+(i+1)];
	out_buffer[noz*(BUFFER_SIZE+1) + BUFFER_SIZE] = 0;
	unsigned int out_ignore_it = (BUFFER_SIZE - win_size)
			/ 2;
	for (int i = 0; i < BUFFER_SIZE - (2 * out_ignore_it); i++) {
		out_buffer[noz*(BUFFER_SIZE+1)+i + out_ignore_it] = out_buffer[noz*(BUFFER_SIZE+1) + i + out_ignore_it]
				+ combAvgSubtoOutBlock[i];

		//for (int i=0;i<out_buffer[noz].size();i++) printf("%f ", out_buffer[noz][i]); printf("\n");

	}
}

__global__
void computeNozzles_v1(float* reference,
						float* input_image,
						aoi* aoi_coordinates,
						float* parallelCoeffs,
						int* parallelSW,
						unsigned int image_height,
						unsigned int image_width,
						aoi* thread_state,
						float* image_parts,
						int* ac_sw,
						float* ac_samples,
						float* cc_coefs,
						float* out_buffer
)
{
	for (unsigned int sn=0;sn<image_height;sn++)
	{
		// thread allocator
		thread_allocator_v1(input_image, aoi_coordinates, image_width, sn, thread_state, image_parts);

		// for all cuda blocks (nozzles):
		unsigned int noz = blockIdx.x;
		// pre-process image: inv and reduce
		float imgPreproc2DDFA;
		imgPreproc2DDFA = preproc_image_v1(image_parts, image_width);

		// single DDFA
		/// auto correlation
		float autoCorrToCombSubMul[BUFFER_SIZE];
		memset(autoCorrToCombSubMul, 0, BUFFER_SIZE*sizeof(float));
		int ac_ignore_it = BUFFER_SIZE / 2 - ac_sw[noz];
		float ac_sampWin[(BUFFER_SIZE*2+1)];
		memset(ac_sampWin, 0, (BUFFER_SIZE*2+1)*sizeof(float));
		auto_correlate_v1(imgPreproc2DDFA, parallelSW, sn, ac_samples, ac_sw, ac_ignore_it, ac_sampWin, autoCorrToCombSubMul);

		/// cross correlation
		/// note: we use the ac_samples, ac_ignore_it, ac_sampWin and ac_sw from the auto correlation stage
		float xCorrToCombSubMul[BUFFER_SIZE];
		memset(xCorrToCombSubMul, 0, BUFFER_SIZE*sizeof(float));
		//// output decoding
		cross_correlate_v1(ac_ignore_it, ac_sw, ac_sampWin, cc_coefs, parallelCoeffs, sn, xCorrToCombSubMul);

		// subtract and multiply ((x-y)/y)
		float combSubMulToCombAvgSub[BUFFER_SIZE];
		submul_v1(combSubMulToCombAvgSub, xCorrToCombSubMul, autoCorrToCombSubMul, ac_sw[noz] * 2);

		// average and subtract
		float combAvgSubtoOutBlock[BUFFER_SIZE];
		avgsub_v1(combSubMulToCombAvgSub, combAvgSubtoOutBlock, ac_sw[noz] * 2);

		// output
		/// output decoding
		out_block_v1(reference, sn, out_buffer, combAvgSubtoOutBlock, ac_sw[noz] * 2);
	}
	__syncthreads();
}

void compute_v1(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width)
{
	// allocate/transfer data on/to to the device
	float* d_reference;
	assert(cudaSuccess == cudaMalloc((void **) &d_reference, N*image_height * sizeof(float)));

	float* d_input_image;
	assert(cudaSuccess == cudaMalloc((void **) &d_input_image, image_width*image_height * sizeof(float)));
	cudaMemcpy(d_input_image, input_image, image_width*image_height * sizeof(float), cudaMemcpyHostToDevice);

	aoi* d_aoi_coordinates;
	assert(cudaSuccess == cudaMalloc((void **) &d_aoi_coordinates, N*image_height * sizeof(aoi)));
	cudaMemcpy(d_aoi_coordinates, aoi_coordinates, N*image_height * sizeof(aoi), cudaMemcpyHostToDevice);

	float* d_parallelCoeffs;
	assert(cudaSuccess == cudaMalloc((void **) &d_parallelCoeffs, BUFFER_SIZE*N*image_height * sizeof(float)));
	cudaMemcpy(d_parallelCoeffs, parallelCoeffs, BUFFER_SIZE*N*image_height * sizeof(float), cudaMemcpyHostToDevice);

	int* d_parallelSW;
	assert(cudaSuccess == cudaMalloc((void **) &d_parallelSW, N*image_height * sizeof(float)));
	cudaMemcpy(d_parallelSW, parallelSW, N*image_height * sizeof(float), cudaMemcpyHostToDevice);

	// state variables
	/// thread allocator
	aoi* thread_state;
	assert(cudaSuccess == cudaMalloc((void **) &thread_state, N * sizeof(aoi)));
	assert(cudaSuccess == cudaMemset((void*)thread_state, 0, N * sizeof(aoi)));
	float* image_parts;
	assert(cudaSuccess == cudaMalloc((void **) &image_parts, image_width * sizeof(float)));
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

	computeNozzles_v1<<<N,1>>>(d_reference, d_input_image, d_aoi_coordinates,
								d_parallelCoeffs, d_parallelSW, image_height, image_width,
								thread_state, image_parts, ac_sw, ac_samples,
								cc_coefs, out_buffer);
	if ( cudaSuccess != cudaGetLastError() )
	    printf( "Error in kernel call!\n" );

	cudaMemcpy(reference, d_reference, N*image_height * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(thread_state);
	cudaFree(image_parts);
	cudaFree(ac_samples);
	cudaFree(cc_coefs);
	cudaFree(out_buffer);
	cudaFree(ac_sw);

	cudaFree(d_parallelSW);
	cudaFree(d_parallelCoeffs);
	cudaFree(d_aoi_coordinates);
	cudaFree(d_input_image);
	cudaFree(d_reference);
}

