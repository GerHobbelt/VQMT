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

#include "PSNR.hpp"

PSNR::PSNR(int h, int w) : Metric(h, w)
{
}

float PSNR::compute(const cv::Mat& original, const cv::Mat& processed)
{
	cv::Mat tmp(original.rows, original.cols, original.type());
	cv::subtract(original, processed, tmp);
	cv::multiply(tmp, tmp, tmp);

	cv::Scalar tmp_mean = cv::mean(tmp);
	double res = tmp_mean.val[0];
	for (int i = 1; i < original.channels(); ++i) {
		res += tmp_mean.val[1];
	}
	res /= original.channels();

	return float(10.0 * log10(255.0 * 255.0 / res));
}
