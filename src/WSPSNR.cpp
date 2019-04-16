//
// Copyright(c) Multimedia Signal Processing Group (MMSPG),
//              Ecole Polytechnique Fédérale de Lausanne (EPFL)
//              http://mmspg.epfl.ch
// All rights reserved.
// Author: Roberto Azevedo (roberto.azevedo@epfl.ch)
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

#include "WSPSNR.hpp"
#include "math.h"

WSPSNR::WSPSNR(int h, int w) : Metric(h, w)
{
}

float WSPSNR::compute(const cv::Mat& original, const cv::Mat& processed)
{
	cv::Mat tmp (height, width,  CV_32F);
    cv::Mat weights (height, width, CV_32F);

  for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            weights.at<float>(j, i) = cos ((j + 0.5 - (height / 2.0)) * M_PI / height);
        }
  }

	cv::subtract(original, processed, tmp);

	cv::multiply(tmp, tmp, tmp);
    cv::multiply(tmp, weights, tmp);

	return float(10*log10(255*255/cv::mean(tmp).val[0]));
}
