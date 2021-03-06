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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "globals.hpp"

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

void thread_allocator(float* input_image,
					aoi* aoi_coordinates,
					unsigned int image_width,
					unsigned int sn,
					aoi* thread_state,
					float* image_parts)
{
	// thread allocator
	for (unsigned int ta = 0; ta < N; ta++) {
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
}

float preproc_image(float* image_parts, unsigned int image_width, unsigned int noz)
{
	float imgPreproc2DDFA = 0;
	float* imgpre = image_parts + (image_width / N) * noz;
	while (*(imgpre) != -1) {
		imgPreproc2DDFA += WHITE_VALUE - *(imgpre);
		imgpre++;
	}
	return imgPreproc2DDFA;
}

void auto_correlate(float imgPreproc2DDFA,
					int* parallelSW,
					unsigned int sn,
					unsigned int noz,
					float* ac_samples,
					int* ac_sw,
					int ac_ignore_it,
					float* ac_sampWin,
					float* autoCorrToCombSubMul)
{
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
void cross_correlate(int ac_ignore_it,
					int* ac_sw,
					float* ac_sampWin,
					float* cc_coefs,
					unsigned int noz,
					float* parallelCoeffs,
					unsigned int sn,
					float* xCorrToCombSubMul)
{
	//// output decoding
	for (int b = -ac_sw[noz]; b < ac_sw[noz]; b++)
		for (int c = 0; c <= ac_sw[noz] * 2; c++) {
			int d = c + b + 1;
			int k = b + ac_sw[noz];
			if ((d >= 0) && (d < ac_sw[noz] * 2))
				xCorrToCombSubMul[ac_sw[noz] * 2 - k - 1] += ac_sampWin[c]
						* cc_coefs[noz*(BUFFER_SIZE+1) + ac_ignore_it + d];
		}
//	free(cc_coeffWin);
	//// next state
	float* cc_temp = parallelCoeffs + sn * N * BUFFER_SIZE
			+ noz * BUFFER_SIZE;
	if (*cc_temp != -1)
		for (int i=0;i<BUFFER_SIZE;i++)
				cc_coefs[noz*(BUFFER_SIZE+1)+i] = cc_temp[i];
}

void submul(float* combSubMulToCombAvgSub,
			float* xCorrToCombSubMul,
			float* autoCorrToCombSubMul,
			unsigned int win_size)
{
	for (int i = 0; i < win_size; i++)
		combSubMulToCombAvgSub[i] = (xCorrToCombSubMul[i]
				- autoCorrToCombSubMul[i]) / autoCorrToCombSubMul[i];
}
void avgsub(float* combSubMulToCombAvgSub,
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
void out_block(float* reference,
				unsigned int sn,
				unsigned int noz,
				float* out_buffer,
				float* combAvgSubtoOutBlock,
				unsigned int win_size)
{
	// output
	/// output decoding
	reference[sn * N + noz] = out_buffer[noz*(BUFFER_SIZE+1)];
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

void computeGold(float* reference,
				float* input_image,
				aoi* aoi_coordinates,
				float* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width)
{
    unsigned int num_threads = N;
	// state variables
	/// thread allocator
	aoi* thread_state = (aoi*)calloc(num_threads,sizeof(aoi));
	float* image_parts = (float*)malloc(image_width*sizeof(float));
	/// image preprocessor
	/// auto correlation
	float* ac_samples = (float*)calloc(num_threads * (BUFFER_SIZE + 1), sizeof(float));
	int* ac_sw = (int*)calloc(num_threads,sizeof(int));
	/// cross correlation
	float* cc_coefs = (float*)calloc(num_threads * (BUFFER_SIZE + 1), sizeof(float));
	/// output block
	float* out_buffer = (float*)calloc(num_threads * (BUFFER_SIZE + 1), sizeof(float));

	// computation
	for (unsigned int sn=0;sn<image_height;sn++) {
		// thread allocator
		thread_allocator(input_image, aoi_coordinates, image_width, sn, thread_state, image_parts);

		// for all threads (nozzles):
		for (unsigned int noz=0;noz<num_threads;noz++)
		{
			// pre-process image: inv and reduce
			float imgPreproc2DDFA;
			imgPreproc2DDFA = preproc_image(image_parts, image_width, noz);

			// single DDFA
			/// auto correlation
			float autoCorrToCombSubMul[BUFFER_SIZE];
			memset(autoCorrToCombSubMul, 0, BUFFER_SIZE*sizeof(float));
			int ac_ignore_it = BUFFER_SIZE / 2 - ac_sw[noz];
			float ac_sampWin[(BUFFER_SIZE*2+1)];
			memset(ac_sampWin, 0, (BUFFER_SIZE*2+1)*sizeof(float));
			auto_correlate(imgPreproc2DDFA, parallelSW, sn, noz, ac_samples, ac_sw, ac_ignore_it, ac_sampWin, autoCorrToCombSubMul);

			/// cross correlation
			/// note: we use the ac_samples, ac_ignore_it, ac_sampWin and ac_sw from the auto correlation stage
			float xCorrToCombSubMul[BUFFER_SIZE];
			memset(xCorrToCombSubMul, 0, BUFFER_SIZE*sizeof(float));
			//// output decoding
			cross_correlate(ac_ignore_it, ac_sw, ac_sampWin, cc_coefs, noz, parallelCoeffs, sn, xCorrToCombSubMul);

			// subtract and multiply ((x-y)/y)
			float combSubMulToCombAvgSub[BUFFER_SIZE];
			submul(combSubMulToCombAvgSub, xCorrToCombSubMul, autoCorrToCombSubMul, ac_sw[noz] * 2);

			// average and subtract
			float combAvgSubtoOutBlock[BUFFER_SIZE];
			avgsub(combSubMulToCombAvgSub, combAvgSubtoOutBlock, ac_sw[noz] * 2);

			// output
			/// output decoding
			out_block(reference, sn, noz, out_buffer, combAvgSubtoOutBlock, ac_sw[noz] * 2);
		}
	}
	free(thread_state);
	free(image_parts);
	free(ac_samples);
	free(cc_coefs);
	free(out_buffer);
	free(ac_sw);
}

