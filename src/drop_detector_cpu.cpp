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

#include <vector>
#include "globals.hpp"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// export C interface
//extern "C"
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
			*(image_parts + (image_width / N) * ta + index) =
					*(input_image + (sn * image_width) + taod);
			index++;
		}
		*(image_parts + (image_width / N) * ta + index) = -1; // to terminate the image part
		/// next state
		if ((aoi_coordinates + sn * N + ta)->start != -1)
			*(thread_state + ta) = *(aoi_coordinates + sn * N + ta);
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
					std::vector<std::vector<double> >& ac_samples,
					std::vector<int>& ac_sw,
					int& ac_ignore_it,
					std::vector<double>& ac_sampWin,
					std::vector<double>& autoCorrToCombSubMul)
{
	//// output decoding
	ac_ignore_it = BUFFER_SIZE / 2 - ac_sw[noz];
	ac_sampWin.assign(ac_samples[noz].begin() + ac_ignore_it,
			ac_samples[noz].end() - ac_ignore_it);
	for (int b = -ac_sw[noz]; b < ac_sw[noz]; b++)
		for (int c = 0; c <= ac_sw[noz] * 2; c++) {
			int d = c + b + 1;
			int k = b + ac_sw[noz];
			if ((d >= 0) && (d < ac_sw[noz] * 2))
				autoCorrToCombSubMul[ac_sw[noz] * 2 - k - 1] += ac_sampWin[c]
						* ac_sampWin[d];
		}
	//// next state
	int ac_temp = *(parallelSW + sn * N + noz);
	if (ac_temp != -1)
		ac_sw[noz] = ac_temp;

	ac_samples[noz].push_back(imgPreproc2DDFA);
	ac_samples[noz].erase(ac_samples[noz].begin());
}
void cross_correlate(int ac_ignore_it,
					const std::vector<int>& ac_sw,
					const std::vector<double>& ac_sampWin,
					std::vector<std::vector<double> >& cc_coefs,
					unsigned int noz,
					double* parallelCoeffs,
					unsigned int sn,
					std::vector<double>& xCorrToCombSubMul) {
	//// output decoding
	std::vector<double> cc_coeffWin;
	cc_coeffWin.assign(cc_coefs[noz].begin() + ac_ignore_it,
			cc_coefs[noz].end() - ac_ignore_it);
	for (int b = -ac_sw[noz]; b < ac_sw[noz]; b++)
		for (int c = 0; c <= ac_sw[noz] * 2; c++) {
			int d = c + b + 1;
			int k = b + ac_sw[noz];
			if ((d >= 0) && (d < ac_sw[noz] * 2))
				xCorrToCombSubMul[ac_sw[noz] * 2 - k - 1] += ac_sampWin[c]
						* cc_coeffWin[d];
		}
	//// next state
	double* cc_temp = parallelCoeffs + sn * N * BUFFER_SIZE
			+ noz * BUFFER_SIZE;
	if (*cc_temp != -1)
		cc_coefs[noz].assign(cc_temp, cc_temp + BUFFER_SIZE);
}

void submul(std::vector<double>& combSubMulToCombAvgSub,
			const std::vector<double>& xCorrToCombSubMul,
			const std::vector<double>& autoCorrToCombSubMul)
{
	for (int i = 0; i < combSubMulToCombAvgSub.size(); i++)
		combSubMulToCombAvgSub[i] = (xCorrToCombSubMul[i]
				- autoCorrToCombSubMul[i]) / autoCorrToCombSubMul[i];
}
void avgsub(const std::vector<double>& combSubMulToCombAvgSub,
			std::vector<double>& combAvgSubtoOutBlock) {
	double as_average = 0;
	for (int i = 0; i < combSubMulToCombAvgSub.size(); i++)
		as_average += combSubMulToCombAvgSub[i];
	as_average /= combSubMulToCombAvgSub.size();
	for (int i = 0; i < combSubMulToCombAvgSub.size(); i++)
		combAvgSubtoOutBlock[i] = combSubMulToCombAvgSub[i] - as_average;
}
void out_block(double* reference,
				unsigned int sn,
				unsigned int noz,
				std::vector<std::vector<double> >& out_buffer,
				const std::vector<double>& combAvgSubtoOutBlock)
{
	// output
	/// output decoding
	*(reference + sn * N + noz) = out_buffer[noz].front();
	printf("%f ", out_buffer[noz].front());
	/// next state
	out_buffer[noz].push_back(0);
	out_buffer[noz].erase(out_buffer[noz].begin());
	unsigned int out_ignore_it = (BUFFER_SIZE - combAvgSubtoOutBlock.size())
			/ 2;
	for (int i = 0; i < BUFFER_SIZE - (2 * out_ignore_it); i++) {
		out_buffer[noz][i + out_ignore_it] = out_buffer[noz][i + out_ignore_it]
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
	aoi* thread_state = (aoi*)malloc(num_threads*sizeof(aoi));
	memset(thread_state, 0, num_threads*sizeof(aoi));
	double* image_parts = (double*)malloc(image_width*sizeof(double));
	/// image preprocessor
	/// auto correlation
	std::vector<std::vector<double> > ac_samples(num_threads,std::vector<double>(BUFFER_SIZE + 1,0));
	std::vector<int> ac_sw(num_threads,0);
	/// cross correlation
	std::vector<std::vector<double> > cc_coefs(num_threads,std::vector<double>(BUFFER_SIZE + 1,0));
	/// output block
	std::vector<std::vector<double> > out_buffer(num_threads,std::vector<double>(BUFFER_SIZE + 1,0));

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
			std::vector<double> autoCorrToCombSubMul(ac_sw[noz] * 2, 0);
			int ac_ignore_it;
			std::vector<double> ac_sampWin;
			auto_correlate(imgPreproc2DDFA, parallelSW, sn, noz, ac_samples, ac_sw, ac_ignore_it, ac_sampWin, autoCorrToCombSubMul);

			/// cross correlation
			/// note: we use the ac_samples, ac_ignore_it, ac_sampWin and ac_sw from the auto correlation stage
			std::vector<double> xCorrToCombSubMul(ac_sw[noz] * 2, 0);
			//// output decoding
			cross_correlate(ac_ignore_it, ac_sw, ac_sampWin, cc_coefs, noz, parallelCoeffs, sn, xCorrToCombSubMul);

			// subtract and multiply ((x-y)/y)
			std::vector<double> combSubMulToCombAvgSub(BUFFER_SIZE);
			submul(combSubMulToCombAvgSub, xCorrToCombSubMul, autoCorrToCombSubMul);

			// average and subtract
			std::vector<double> combAvgSubtoOutBlock(BUFFER_SIZE);
			avgsub(combSubMulToCombAvgSub, combAvgSubtoOutBlock);

			// output
			/// output decoding
			out_block(reference, sn, noz, out_buffer, combAvgSubtoOutBlock);
		}
		printf("\n");
	}
	free(thread_state);
	free(image_parts);
}

