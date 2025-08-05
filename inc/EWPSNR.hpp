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

/**************************************************************************

 Calculation of the Peak Signal-to-Noise Ratio (PSNR) image quality measure.

**************************************************************************/

#ifndef EWPSNR_hpp
#define EWPSNR_hpp

#include "Metric.hpp"
#include <unordered_map>
#include <string>
#include <vector>


class EWPSNR : protected Metric {
public:

	EWPSNR(int height, int width);
	// Compute the PSNR index of the processed image
	float compute(const cv::Mat& original, const cv::Mat& processed);

	bool match_eye_track_data(std::string filename);

    void set_frame_no(unsigned int no) { m_frame_no = no; };

private:
	float WPSNR(const cv::Mat& original, const cv::Mat& processed, const cv::Mat& w);
    void compute_eye_weight(cv::Mat& w);
	bool load_eye_track_data();
    double retina_gaussian (int x,int y, float x_e, float y_e, int sigma_x, int sigma_y);



	std::string m_id;
	std::string m_path;
    unsigned int m_frame_no;

    std::vector<std::vector<std::pair<float, float>>> m_gazes;

	const std::unordered_map<std::string, std::string> m_eye_track_data = {
			{"bus", "/data/SFU_etdb/CSV/bus-Screen.csv"},
			{"city", "/data/SFU_etdb/CSV/city-Screen.csv"},
			{"crew", "/data/SFU_etdb/CSV/crew-Screen.csv"},
			{"flower", "/data/SFU_etdb/CSV/flower-Screen.csv"},
			{"foreman", "/data/SFU_etdb/CSV/foreman-Screen.csv"},
			{"hall", "/data/SFU_etdb/CSV/hall-Screen.csv"},
			{"harbour", "/data/SFU_etdb/CSV/harbour-Screen.csv"},
			{"mobile", "/data/SFU_etdb/CSV/mobile-Screen.csv"},
			{"mother", "/data/SFU_etdb/CSV/mother-Screen.csv"},
			{"soccer", "/data/SFU_etdb/CSV/soccer-Screen.csv"},
			{"stefan", "/data/SFU_etdb/CSV/stefan-Screen.csv"},
			{"tempete", "/data/SFU_etdb/CSV/tempete-Screen.csv"}
	};
};

#endif
