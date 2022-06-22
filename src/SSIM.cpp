//
// Copyright(c) Multimedia Signal Processing Group (MMSPG),
//              Ecole Polytechnique Fédérale de Lausanne (EPFL)
//              http://mmspg.epfl.ch
//              Zhou Wang
//              https://ece.uwaterloo.ca/~z70wang/
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

//
// This is an OpenCV implementation of the original Matlab implementation
// from Nikolay Ponomarenko available from http://live.ece.utexas.edu/research/quality/.
// Please refer to the following papers:
// - Z. Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli, "Image quality
//   assessment: from error visibility to structural similarity," IEEE
//   Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, April 2004.
//

#include "SSIM.hpp"

const float SSIM::C1 = 6.5025f;
const float SSIM::C2 = 58.5225f;

enum {
	SSIM_SIZE = 8,
	GK_SIZE = 11,
};

SSIM::SSIM(int h, int w, int t) : Metric(h, w, t),
  mu1(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  mu2(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  mu1_sq(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  mu2_sq(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  mu1_mu2(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  img1_sq(h, w, t),
  img2_sq(h, w, t),
  img1_img2(h, w, t),
  sigma1_sq(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  sigma2_sq(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t),
  sigma12(h - (GK_SIZE - 1), w - (GK_SIZE - 1), t)

#if defined(HAVE_SSIM_BLUR_8)
  ,
  bmu1(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bmu2(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bmu1_sq(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bmu2_sq(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bmu1_mu2(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bsigma1_sq(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bsigma2_sq(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t),
  bsigma12(h - (SSIM_SIZE - 1), w - (SSIM_SIZE - 1), t)
#endif
{
}

float SSIM::compute(const cv::Mat& original, const cv::Mat& processed)
{
	cv::Scalar res = computeSSIM(original, processed);
	return float(res.val[0]);
}

#if defined(HAVE_SSIM_BLUR_8)
float SSIM::compute_x8(const cv::Mat& img1, const cv::Mat& img2)
{
	// mu1 = filter2(window, img1, 'valid');
	applyBlur(img1, bmu1, SSIM_SIZE);

	// mu2 = filter2(window, img2, 'valid');
	applyBlur(img2, bmu2, SSIM_SIZE);

	// mu1_sq = mu1.*mu1;
	cv::multiply(bmu1, bmu1, bmu1_sq);
	// mu2_sq = mu2.*mu2;
	cv::multiply(bmu2, bmu2, bmu2_sq);
	// mu1_mu2 = mu1.*mu2;
	cv::multiply(bmu1, bmu2, bmu1_mu2);

	cv::multiply(img1, img1, img1_sq);
	cv::multiply(img2, img2, img2_sq);
	cv::multiply(img1, img2, img1_img2);

	// sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
	applyBlur(img1_sq, bsigma1_sq, SSIM_SIZE);
	bsigma1_sq -= bmu1_sq;

	// sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
	applyBlur(img2_sq, bsigma2_sq, SSIM_SIZE);
	bsigma2_sq -= bmu2_sq;

	// sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
	applyBlur(img1_img2, bsigma12, SSIM_SIZE);
	bsigma12 -= bmu1_mu2;

	// cs_map = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);
	// tmp1 = 2*sigma12 + C2;
	cv::Mat& tmp1 = bsigma12;
	tmp1 *= 2;
	tmp1 += C2;

	//tmp2 = sigma1_sq + sigma2_sq + C2;
	cv::Mat& tmp2 = bsigma1_sq;
	tmp2 += bsigma2_sq;
	tmp2 += C2;

	cv::divide(tmp1, tmp2, tmp1);

	// ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
	//tmp3 = 2*mu1_mu2 + C1;
	cv::Mat& tmp3 = bmu1_mu2;
	tmp3 *= 2;
	tmp3 += C1;

	//tmp4 = mu1_sq + mu2_sq + C1;
	cv::Mat& tmp4 = bmu1_sq;
	tmp4 += bmu2_sq;
	tmp4 += C1;

	cv::multiply(tmp3, tmp1, tmp3);
	cv::divide(tmp3, tmp4, tmp3);
	cv::Mat& ssim_map = tmp3;

	// mssim = mean2(ssim_map);
	cv::Scalar ssim_mean = cv::mean(ssim_map);
	return float(ssim_mean.val[0]);
}
#endif

cv::Scalar SSIM::computeSSIM(const cv::Mat& img1, const cv::Mat& img2)
{
	// mu1 = filter2(window, img1, 'valid');
	applyGaussianBlur(img1, mu1, GK_SIZE, 1.5);

	// mu2 = filter2(window, img2, 'valid');
	applyGaussianBlur(img2, mu2, GK_SIZE, 1.5);

	// mu1_sq = mu1.*mu1;
	cv::multiply(mu1, mu1, mu1_sq);
	// mu2_sq = mu2.*mu2;
	cv::multiply(mu2, mu2, mu2_sq);
	// mu1_mu2 = mu1.*mu2;
	cv::multiply(mu1, mu2, mu1_mu2);

	cv::multiply(img1, img1, img1_sq);
	cv::multiply(img2, img2, img2_sq);
	cv::multiply(img1, img2, img1_img2);

	// sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
	applyGaussianBlur(img1_sq, sigma1_sq, GK_SIZE, 1.5);
	sigma1_sq -= mu1_sq;

	// sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
	applyGaussianBlur(img2_sq, sigma2_sq, GK_SIZE, 1.5);
	sigma2_sq -= mu2_sq;

	// sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
	applyGaussianBlur(img1_img2, sigma12, GK_SIZE, 1.5);
	sigma12 -= mu1_mu2;

	// cs_map = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);
	cv::Mat& tmp1 = sigma12;
	tmp1 *= 2;
	tmp1 += C2;

	//tmp2 = sigma1_sq + sigma2_sq + C2;
	cv::Mat& tmp2 = sigma1_sq;
	tmp2 += sigma2_sq;
	tmp2 += C2;

	cv::divide(tmp1, tmp2, tmp1);
	cv::Mat& cs_map = tmp1;

	// ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
	cv::Mat& tmp3 = mu1_mu2;
	tmp3 *= 2;
	tmp3 += C1;

	//tmp4 = mu1_sq + mu2_sq + C1;
	cv::Mat& tmp4 = mu1_sq;
	tmp4 += mu2_sq;
	tmp4 += C1;

	cv::multiply(tmp3, tmp1, tmp3);
	cv::divide(tmp3, tmp4, tmp3);
	cv::Mat& ssim_map = tmp3;

	// mssim = mean2(ssim_map);
	cv::Scalar ssim_mean = cv::mean(ssim_map);
	double mssim = ssim_mean.val[0];
	// mcs = mean2(cs_map);
	cv::Scalar cs_mean = cv::mean(cs_map);
	double mcs = cs_mean.val[0];

	for (int i = 1; i < img1.channels(); ++i) {
		mssim += ssim_mean.val[i];
		mcs += cs_mean.val[i];
	}
	mssim /= img1.channels();
	mcs /= img1.channels();

	cv::Scalar res(mssim, mcs);

	return res;
}
