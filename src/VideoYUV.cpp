//
// Copyright(c) Multimedia Signal Processing Group (MMSPG),
//              Ecole Polytechnique Fédérale de Lausanne (EPFL)
//              http://mmspg.epfl.ch
// All rights reserved.
// Author: Philippe Hanhart (philippe.hanhart@epfl.ch)
//
// Permission is hereby granted, without written agreement and without
// license or royalty fees, to use, copy, modify, and distribute the
// software provided and its documentation for research purpose only,
// provided that this copyright notice and the original authors' names
// appear on all copies and supporting documentation.
// The software provided may not be commercially distributed.
// In no event shall the Ecole Polytechnique Fédérale de Lausanne (EPFL)
// be liable to any party for direct, indirect, special, incidental, or
// consequential damages arising out of the use of the software and its
// documentation.
// The Ecole Polytechnique Fédérale de Lausanne (EPFL) specifically
// disclaims any warranties.
// The software provided hereunder is on an "as is" basis and the Ecole
// Polytechnique Fédérale de Lausanne (EPFL) has no obligation to provide
// maintenance, support, updates, enhancements, or modifications.
//

#include "VideoYUV.hpp"

VideoYUV::VideoYUV(const char *f, int h, int w, int nbf, int chroma_format)
{
	chf = chroma_format;
	if(strcmp(f, "-") == 0)
		file = stdin;
	else
		file = fopen(f, "rb");
		
	if (!file) {
		fprintf(stderr, "readOneFrame: cannot open input file (%s)\n", f);
		exit(EXIT_FAILURE);
	}
	height = h;
	width  = w;
	nbframes = nbf;

	comp_height[0] = height;
	comp_width [0] = width;
	if (chroma_format == CHROMA_SUBSAMP_400) {
		comp_height[2] = comp_height[1] = 0;
		comp_width [2] = comp_width [1] = 0;
	}
	else if (chroma_format == CHROMA_SUBSAMP_420) {
		// Check size
		if (height % 2 == 1 || width % 2 == 1) {
			fprintf(stderr, "YUV420: 'height' and 'width' have to be even numbers.\n");
			exit(EXIT_FAILURE);
		}

		comp_height[2] = comp_height[1] = height >> 1;
		comp_width [2] = comp_width [1] = width >> 1;
	}
	else if (chroma_format == CHROMA_SUBSAMP_422) {
		// Check size
		if (w % 2 == 1) {
			fprintf(stderr, "YUV422: 'width' has to be an even number.\n");
			exit(EXIT_FAILURE);
		}

		comp_height[2] = comp_height[1] = height;
		comp_width [2] = comp_width [1] = width >> 1;
	}
	else {
		comp_height[2] = comp_height[1] = height;
		comp_width [2] = comp_width [1] = width;
	}
	comp_size[0] = comp_height[0]*comp_width[0];
	comp_size[1] = comp_height[1]*comp_width[1];
	comp_size[2] = comp_height[2]*comp_width[2];
	
	size = static_cast<size_t>(comp_size[0]+comp_size[1]+comp_size[2]);
	
	data = new imgpel[size];
	luma = data;
	chroma[0] = data+comp_size[0];
	chroma[1] = data+comp_size[0]+comp_size[1];
	yuv_data = new imgpel[height * width * 3];
}

VideoYUV::~VideoYUV()
{
	delete[] data;
	fclose(file);
}

bool VideoYUV::readOneFrame()
{
	imgpel *ptr_data = data;

	if (fread(ptr_data, 1, size, file) != size) {
		fprintf(stderr, "readOneFrame: cannot read %zu bytes from input file, unexpected EOF.\n", size);
    return false;
  }
	yuv_ready = false;

	return true;
}

imgpel *VideoYUV::getYUV()
{
	if (yuv_ready) return yuv_data;

	imgpel *ptr = yuv_data;
	imgpel *lptr = luma;
	imgpel *c0 = chroma[0];
	imgpel *c1 = chroma[1];

	if (chf == CHROMA_SUBSAMP_400) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				*ptr++ = *lptr++;
				*ptr++ = 0;
				*ptr++ = 0;
			}
		}
	} else if (chf == CHROMA_SUBSAMP_420) {
		imgpel *next_line_ptr = yuv_data + width;
		imgpel *next_line_lptr = luma + width;

		for (int y = 0; y < height; y += 2) {
			for (int x = 0; x < width; x += 2) {
				*ptr++ = *lptr++;
				*ptr++ = *c0;
				*ptr++ = *c1;

				*ptr++ = *lptr++;
				*ptr++ = *c0;
				*ptr++ = *c1;

				*next_line_ptr++ = *next_line_lptr++;
				*next_line_ptr++ = *c0;
				*next_line_ptr++ = *c1;

				*next_line_ptr++ = *next_line_lptr++;
				*next_line_ptr++ = *c0++;
				*next_line_ptr++ = *c1++;
			}

			ptr += width;
			lptr += width;
			next_line_ptr += width;
			next_line_lptr += width;
		}
	} else if (chf == CHROMA_SUBSAMP_422) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; x += 2) {
				*ptr++ = *lptr++;
				*ptr++ = *c0;
				*ptr++ = *c1;

				*ptr++ = *lptr++;
				*ptr++ = *c0++;
				*ptr++ = *c1++;
			}
		}
	} else if (chf == CHROMA_SUBSAMP_444) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				*ptr++ = *lptr++;
				*ptr++ = *c0++;
				*ptr++ = *c1++;
			}
		}
	}

	yuv_ready = true;

	return yuv_data;
}

void VideoYUV::getLuma(cv::Mat& local_luma, int type)
{
	cv::Mat tmp(height, width, CV_8UC1, luma);
	if (type == CV_8UC1) {
		tmp.copyTo(local_luma);
	}
	else {
		tmp.convertTo(local_luma, type);
	}
}

void VideoYUV::getYUV(cv::Mat& yuv)
{
	cv::Mat tmp(height, width, CV_8UC3, getYUV());
	tmp.convertTo(yuv, yuv.type());
}

void VideoYUV::getU(cv::Mat& u)
{
	cv::Mat tmp(comp_height[1], comp_width[1], CV_8UC1, chroma[0]);
	tmp.convertTo(u, u.type());
}

void VideoYUV::getV(cv::Mat& v)
{
	cv::Mat tmp(comp_height[2], comp_width[2], CV_8UC1, chroma[1]);
	tmp.convertTo(v, v.type());
}
