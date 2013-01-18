/*
 * file_io.cpp
 *
 *  Created on: Jan 14, 2013
 *      Author: shaniaki
 */

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>

#include <tiffio.h>

#include "globals.hpp"

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
double* readTiff(char* filename, unsigned int* w, unsigned int* h);

extern "C"
aoi* readAOIs(char* filename, unsigned int image_height, unsigned int num_threads);

extern "C"
double* readCoefs(char* filename, unsigned int image_height, unsigned int num_threads, unsigned int coefs_size);

extern "C"
int* readSWs(char* filename, unsigned int image_height, unsigned int num_threads);

/*
std::vector<std::vector<double> > readTiff(char* filename) {
	// read the input from the TIFF
	printf("Opening the input file: %s\n", filename);
	uint32 imagelength;
	TIFF* tif = TIFFOpen(filename, "r");
	if (tif)
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
	else
		printf("The Tiff image could not be opened. File name: %s\n", filename);

	tsize_t lineSize = TIFFScanlineSize(tif);
	std::vector < std::vector<double> > inputVector;
	tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
	for (uint32 row = 0; row < imagelength; row++) {
		std::vector < double > rowVector;
		TIFFReadScanline(tif, buf, row);
		unsigned char* smartBuf = (unsigned char*) (buf);
		for (tsize_t rowPos = 0; rowPos < lineSize; rowPos++)
			rowVector.push_back(smartBuf[rowPos]);
		inputVector.push_back(rowVector);
	}
	_TIFFfree(buf);
	TIFFClose(tif);
	return inputVector;
}
*/

double* readTiff(char* filename, unsigned int* w, unsigned int* h) {
	// read the input from the TIFF
	printf("Opening the input file: %s\n", filename);
	TIFF* tif = TIFFOpen(filename, "r");
	if (tif)
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, h);
	else
		printf("The Tiff image could not be opened. File name: %s\n", filename);

	*w = TIFFScanlineSize(tif);
	double* image;
	image = (double*)malloc(*w * *h * sizeof(double));
	tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
	for (uint32 row = 0; row < *h; row++) {
		double* rowVector = image + row * *w;
		TIFFReadScanline(tif, buf, row);
		unsigned char* smartBuf = (unsigned char*) (buf);
		for (tsize_t rowPos = 0; rowPos < *w; rowPos++)
			rowVector[rowPos] = smartBuf[rowPos];
	}
	_TIFFfree(buf);
	TIFFClose(tif);
	return image;
}

/*
std::vector<std::vector<aoi> > readAOIs(char* filename) {
	// read the coordinates for areas of interest
	printf("Opening the AOI coordinates file: %s\n", filename);
	std::ifstream aoiFile(filename);
	if (!aoiFile.is_open())
		printf("The AOI file could not be opened. File name: %s\n", filename);

	std::vector<std::vector<aoi> > result;
	char line[256]; // FIXME: calculate this!
	while (aoiFile.getline(line,256) > 0) {

		std::vector<aoi> lineResult;
		aoi single_tuple;
		unsigned int element_counter;

		std::string element;
		std::stringstream ss(line);

		bool push_tuple = false;

		while (ss >> element) {
			if (element == "$") {
				single_tuple.start = -1;
				element_counter = 0;
			} else if (element == "|") {
				if (push_tuple) {
					element_counter = 0;
					lineResult.push_back(single_tuple);
				} else {
					push_tuple = true;
					element_counter = 0;
				}
			} else {
				int value;
				std::stringstream ss_tmp(element);
				ss_tmp >> value;

				if (element_counter == 0) {
					single_tuple.start = value;
				} else {
					single_tuple.end = value;
					element_counter = 0;
				}
				element_counter++;
			}
		}
		element_counter = 0;
		result.push_back(lineResult);
	}

	aoiFile.close();
	return result;
}
*/

aoi* readAOIs(char* filename, unsigned int image_height, unsigned int num_threads) {
	// read the coordinates for areas of interest
	printf("Opening the AOI coordinates file: %s\n", filename);
	std::ifstream aoiFile(filename);
	if (!aoiFile.is_open())
		printf("The AOI file could not be opened. File name: %s\n", filename);

	aoi* result = (aoi*)malloc(image_height * num_threads * sizeof(aoi));
	char *line = (char*)malloc(num_threads * 10 + 2);
	unsigned int line_number = 0;
	while (aoiFile.getline(line,num_threads * 10 + 2) > 0) {
		aoi* lineResult = result + line_number * num_threads;
		aoi single_tuple;
		unsigned int element_counter;

		std::string element;
		std::stringstream ss(line);

		bool push_tuple = false;

		while (ss >> element) {
			if (element == "$") {
				single_tuple.start = -1;
				element_counter = 0;
			} else if (element == "|") {
				if (push_tuple) {
					element_counter = 0;
					*lineResult = single_tuple; lineResult++;
				} else {
					push_tuple = true;
					element_counter = 0;
				}
			} else {
				int value;
				std::stringstream ss_tmp(element);
				ss_tmp >> value;

				if (element_counter == 0) {
					single_tuple.start = value;
				} else {
					single_tuple.end = value;
					element_counter = 0;
				}
				element_counter++;
			}
		}
		element_counter = 0;
		line_number++;
	}

	aoiFile.close();
	free(line);
	return result;
}

/*
std::vector<std::vector<std::vector<double> > > readCoefs(char* filename) {
	// read the coefficients
	printf("Opening the coefficients file: %s\n", filename);
	std::ifstream coefFile(filename);
	if (!coefFile.is_open())
		printf("The coefficients file could not be opened. File name: %s\n", filename);

	std::vector<std::vector<std::vector<double> > > result;
	char line[256]; // FIXME: calculate this!
	while (coefFile.getline(line,256) > 0) {

		std::vector<std::vector<double> > lineVector;
		std::vector<double> subVector;

		std::string element;
		std::stringstream ss(line);

		while (ss >> element) {
			if (element == "|") {
				if (subVector.size() != 0) {
					lineVector.push_back(subVector);
					subVector.clear();
				}
			} else if (element == "$") {
				lineVector.push_back(std::vector<double>());
			} else {
				double result;
				std::stringstream ss_tmp(element);
				ss_tmp >> result;
				subVector.push_back(result);
			}
		}
		result.push_back(lineVector);
	}

	coefFile.close();
	return result;
}
*/

double* readCoefs(char* filename, unsigned int image_height, unsigned int num_threads, unsigned int coefs_size) {
	// read the coefficients
	printf("Opening the coefficients file: %s\n", filename);
	std::ifstream coefFile(filename);
	if (!coefFile.is_open())
		printf("The coefficients file could not be opened. File name: %s\n", filename);

	double* result = (double*)malloc(image_height * num_threads * coefs_size * sizeof(double));
	char *line = (char*)malloc(num_threads * coefs_size * 4 + 2);
	unsigned int line_number = 0;
	while (coefFile.getline(line,num_threads * coefs_size * 4 + 2) > 0) {

		double* lineVector = result + line_number * num_threads * coefs_size;
		bool emptyCoefs = true;

		std::string element;
		std::stringstream ss(line);

		while (ss >> element) {
			if (element == "|") {
				if (!emptyCoefs) {
					emptyCoefs = true;
				}
			} else if (element == "$") {
				*lineVector = -1;
				lineVector += coefs_size;
			} else {
				double value;
				std::stringstream ss_tmp(element);
				ss_tmp >> value;
				*lineVector = value;
				lineVector++;
			}
		}
		line_number++;
	}

	coefFile.close();
	free(line);
	return result;
}

/*
std::vector<std::vector<int> > readSWs(char* filename) {
	// read the SWs
	printf("Opening the SW file: %s\n", filename);
	std::ifstream swFile(filename);
	if (!swFile.is_open())
		printf("The SW file could not be opened. File name: %s\n", filename);

	std::vector<std::vector<int> > result;
	char line[256]; // FIXME: calculate this!
	while (swFile.getline(line,256) > 0) {

		std::vector<int> abstVector;

		std::string element;
		std::stringstream ss(line);

		while (ss >> element) {
			if (element != "*") {
				int result;
				std::istringstream is(element);
				is >> result;
				abstVector.push_back(result);
			} else {
				abstVector.push_back(-1);
			}
		}
		result.push_back(abstVector);
	}

	swFile.close();
	return result;
}
*/

int* readSWs(char* filename, unsigned int image_height, unsigned int num_threads) {
	// read the SWs
	printf("Opening the SW file: %s\n", filename);
	std::ifstream swFile(filename);
	if (!swFile.is_open())
		printf("The SW file could not be opened. File name: %s\n", filename);

	int* result = (int*)malloc(image_height * num_threads * sizeof(int));
	char *line = (char*)malloc(num_threads * 5 + 2);
	unsigned int line_number = 0;
	while (swFile.getline(line,num_threads * 5 + 2) > 0) {

		int* abstVector = result + line_number * num_threads;

		std::string element;
		std::stringstream ss(line);

		while (ss >> element) {
			if (element != "*") {
				int value;
				std::istringstream is(element);
				is >> value;
				*abstVector = value;
			} else {
				*abstVector = -1;
			}
			abstVector++;
		}
		line_number++;
	}

	swFile.close();
	free(line);
	return result;
}


