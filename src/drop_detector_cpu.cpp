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

#include "globals.hpp"

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(double* reference,
				double* input_image,
				aoi* aoi_coordinates,
				double* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width);

void thread_allocator(double* input_image,
					aoi* aoi_coordinates,
					unsigned int image_width,
					unsigned int sn,
					aoi* thread_state,
					double* image_parts)
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

double preproc_image(double* image_parts, unsigned int image_width, unsigned int noz)
{
	double imgPreproc2DDFA = 0;
	double* imgpre = image_parts + (image_width / N) * noz;
	while (*(imgpre) != -1) {
		imgPreproc2DDFA += WHITE_VALUE - *(imgpre);
		imgpre++;
	}
	return imgPreproc2DDFA;
}

void auto_correlate(double imgPreproc2DDFA,
					int* parallelSW,
					unsigned int sn,
					unsigned int noz,
					double* ac_samples,
					int* ac_sw,
					int ac_ignore_it,
					double* ac_sampWin,
					double* autoCorrToCombSubMul)
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
					double* ac_sampWin,
					double* cc_coefs,
					unsigned int noz,
					double* parallelCoeffs,
					unsigned int sn,
					double* xCorrToCombSubMul)
{
	//// output decoding
	double* cc_coeffWin = (double*)malloc((2*ac_sw[noz]+1)*sizeof(double));
	for (int i=0;i<2*ac_sw[noz]+1;i++)
		cc_coeffWin[i] =cc_coefs[noz*(BUFFER_SIZE+1) + ac_ignore_it + i];
	for (int b = -ac_sw[noz]; b < ac_sw[noz]; b++)
		for (int c = 0; c <= ac_sw[noz] * 2; c++) {
			int d = c + b + 1;
			int k = b + ac_sw[noz];
			if ((d >= 0) && (d < ac_sw[noz] * 2))
				xCorrToCombSubMul[ac_sw[noz] * 2 - k - 1] += ac_sampWin[c]
						* cc_coeffWin[d];
		}
	free(cc_coeffWin);
	//// next state
	double* cc_temp = parallelCoeffs + sn * N * BUFFER_SIZE
			+ noz * BUFFER_SIZE;
	if (*cc_temp != -1)
		for (int i=0;i<BUFFER_SIZE;i++)
				cc_coefs[noz*(BUFFER_SIZE+1)+i] = cc_temp[i];
}

void submul(double* combSubMulToCombAvgSub,
			double* xCorrToCombSubMul,
			double* autoCorrToCombSubMul,
			unsigned int win_size)
{
	for (int i = 0; i < win_size; i++)
		combSubMulToCombAvgSub[i] = (xCorrToCombSubMul[i]
				- autoCorrToCombSubMul[i]) / autoCorrToCombSubMul[i];
}
void avgsub(double* combSubMulToCombAvgSub,
			double* combAvgSubtoOutBlock,
			unsigned int win_size)
{
	double as_average = 0;
	for (int i = 0; i < win_size; i++)
		as_average += combSubMulToCombAvgSub[i];
	as_average /= win_size;
	for (int i = 0; i < win_size; i++)
		combAvgSubtoOutBlock[i] = combSubMulToCombAvgSub[i] - as_average;
}
void out_block(double* reference,
				unsigned int sn,
				unsigned int noz,
				double* out_buffer,
				double* combAvgSubtoOutBlock,
				unsigned int win_size)
{
	// output
	/// output decoding
	reference[sn * N + noz] = out_buffer[noz*(BUFFER_SIZE+1)];
	printf("%f ", out_buffer[noz*(BUFFER_SIZE+1)]);
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

void computeGold(double* reference,
				double* input_image,
				aoi* aoi_coordinates,
				double* parallelCoeffs,
				int* parallelSW,
				unsigned int image_height,
				unsigned int image_width)
{
    unsigned int num_threads = N;
	// state variables
	/// thread allocator
	aoi* thread_state = (aoi*)calloc(num_threads,sizeof(aoi));
	double* image_parts = (double*)malloc(image_width*sizeof(double));
	/// image preprocessor
	/// auto correlation
	double* ac_samples = (double*)calloc(num_threads * (BUFFER_SIZE + 1), sizeof(double));
	int* ac_sw = (int*)calloc(num_threads,sizeof(int));
	/// cross correlation
	double* cc_coefs = (double*)calloc(num_threads * (BUFFER_SIZE + 1), sizeof(double));
	/// output block
	double* out_buffer = (double*)calloc(num_threads * (BUFFER_SIZE + 1), sizeof(double));

	// computation
	for (unsigned int sn=0;sn<image_height;sn++) {
		// thread allocator
		thread_allocator(input_image, aoi_coordinates, image_width, sn, thread_state, image_parts);

		// for all threads (nozzles):
		for (unsigned int noz=0;noz<num_threads;noz++)
		{
			// pre-process image: inv and reduce
			double imgPreproc2DDFA;
			imgPreproc2DDFA = preproc_image(image_parts, image_width, noz);

			// single DDFA
			/// auto correlation
			double *autoCorrToCombSubMul = (double*)calloc(ac_sw[noz] * 2, sizeof(double));
			int ac_ignore_it = BUFFER_SIZE / 2 - ac_sw[noz];
			double *ac_sampWin = (double*)calloc(ac_sw[noz] * 2 + 1, sizeof(double));
			auto_correlate(imgPreproc2DDFA, parallelSW, sn, noz, ac_samples, ac_sw, ac_ignore_it, ac_sampWin, autoCorrToCombSubMul);

			/// cross correlation
			/// note: we use the ac_samples, ac_ignore_it, ac_sampWin and ac_sw from the auto correlation stage
			double *xCorrToCombSubMul = (double*)calloc(ac_sw[noz] * 2, sizeof(double));
			//// output decoding
			cross_correlate(ac_ignore_it, ac_sw, ac_sampWin, cc_coefs, noz, parallelCoeffs, sn, xCorrToCombSubMul);
			free(ac_sampWin);

			// subtract and multiply ((x-y)/y)
			double *combSubMulToCombAvgSub = (double*)calloc(ac_sw[noz] * 2, sizeof(double));
			submul(combSubMulToCombAvgSub, xCorrToCombSubMul, autoCorrToCombSubMul, ac_sw[noz] * 2);
			free(autoCorrToCombSubMul);
			free(xCorrToCombSubMul);

			// average and subtract
			double *combAvgSubtoOutBlock = (double*)calloc(ac_sw[noz] * 2, sizeof(double));
			avgsub(combSubMulToCombAvgSub, combAvgSubtoOutBlock, ac_sw[noz] * 2);
			free(combSubMulToCombAvgSub);

			// output
			/// output decoding
			out_block(reference, sn, noz, out_buffer, combAvgSubtoOutBlock, ac_sw[noz] * 2);
			free(combAvgSubtoOutBlock);
		}
		printf("\n");
	}
	free(thread_state);
	free(image_parts);
	free(ac_samples);
	free(cc_coefs);
	free(out_buffer);
	free(ac_sw);
}

