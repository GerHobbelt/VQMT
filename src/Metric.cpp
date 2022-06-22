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

#include "Metric.hpp"

Metric::Metric(int h, int w, int t) : gb_tmp(h, w, t)
{
	height = h;
	width = w;
}

Metric::~Metric()
{

}

void Metric::applyGaussianBlur(const cv::Mat& src, cv::Mat& dst, int ksize, double sigma)
{
	int invalid = (ksize-1)/2;
	cv::GaussianBlur(src, gb_tmp, cv::Size(ksize,ksize), sigma);
	gb_tmp(cv::Range(invalid, gb_tmp.rows-invalid), cv::Range(invalid, gb_tmp.cols - invalid)).copyTo(dst);
}

void Metric::applyBlur(const cv::Mat& src, cv::Mat& dst, int ksize)
{
	cv::blur(src, gb_tmp, cv::Size(ksize,ksize), cv::Point(0, 0));
	gb_tmp(cv::Range(0, gb_tmp.rows - ksize), cv::Range(0, gb_tmp.cols - ksize)).copyTo(dst);
}
