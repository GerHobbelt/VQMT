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

#include "EWPSNR.hpp"
#include <cctype>
#include <algorithm>
#include <fstream>
#include <iostream>

#define PI 3.14159265

EWPSNR::EWPSNR(int h, int w) : Metric(h, w)
{
//	m_eye_track_data["bus"] = "/data/SFU_etdb/CSV/bus-Screen.csv";
//    m_eye_track_data["city"] = "/data/SFU_etdb/CSV/city-Screen.csv";
//    m_eye_track_data["crew"] = "/data/SFU_etdb/CSV/crew-Screen.csv";
//    m_eye_track_data["flower"] = "/data/SFU_etdb/CSV/flower-Screen.csv";
//    m_eye_track_data["foreman"] = "/data/SFU_etdb/CSV/foreman-Screen.csv";
//    m_eye_track_data["hall"] = "/data/SFU_etdb/CSV/hall-Screen.csv";
//    m_eye_track_data["harbour"] = "/data/SFU_etdb/CSV/harbour-Screen.csv";
//    m_eye_track_data["mobile"] = "/data/SFU_etdb/CSV/mobile-Screen.csv";
//    m_eye_track_data["mother"] = "/data/SFU_etdb/CSV/mother-Screen.csv";
//    m_eye_track_data["soccer"] = "/data/SFU_etdb/CSV/soccer-Screen.csv";
//    m_eye_track_data["stefan"] = "/data/SFU_etdb/CSV/stefan-Screen.csv";
//    m_eye_track_data["tempete"] = "/data/SFU_etdb/CSV/tempete-Screen.csv";
}

float EWPSNR::compute(const cv::Mat& original, const cv::Mat& processed)
{
    cv::Mat w(original.size(), CV_32FC1);
    compute_eye_weight(w);
	return WPSNR(original, processed, w);
}

float EWPSNR::WPSNR(const cv::Mat& original, const cv::Mat& processed, const cv::Mat& w)
{
	cv::Mat tmp(height,width,CV_32F);
	cv::subtract(original, processed, tmp);
	cv::multiply(tmp, tmp, tmp);
	cv::multiply(tmp, w, tmp);
	return float(10*log10(255*255/(cv::mean(tmp).val[0]*original.cols*original.rows)));
}

void EWPSNR::compute_eye_weight(cv::Mat &w)
{
    float sum = 0;
    for (int i=0; i<w.rows; i++) {
        for (int j=0; j<w.cols; j++) {
            float* data = w.ptr<float>(i, j);
            *data = 0;
            for (auto &p: m_gazes[m_frame_no]) {
                *data += static_cast<float>(retina_gaussian(j, i, p.first, p.second, 64, 64));
            }
            sum += *data;
        }
    }
    w /= sum;
}

double  EWPSNR::retina_gaussian(int x, int y, float x_e, float y_e, int sigma_x, int sigma_y)
{
    auto sq = [](float _x) { return _x*_x; };
    return (1/(2*PI*sigma_x*sigma_y))
           * exp(-(sq(static_cast<float>(x)-x_e)/(2*sq(static_cast<float>(sigma_x)))
                   +sq(static_cast<float>(y)-y_e)/(2*sq(static_cast<float>(sigma_y)))));
}

bool EWPSNR::match_eye_track_data(std::string filename)
{
    std::transform(filename.begin(), filename.end(), filename.begin(), tolower);
    for(auto &i : m_eye_track_data) {
        if(filename.find(i.first) != std::string::npos) {
            m_id = i.first;
            m_path = i.second;
            if ( load_eye_track_data() ) {
                return true;
            } else {
                return false;
            }
        }
    }
    return false;
}

bool EWPSNR::load_eye_track_data()
{
    try {
        std::ifstream csv(m_path);
        std::string buffer;
        std::getline(csv, buffer);
        std::getline(csv, buffer);
        float x, y;
        char ch;
        while (!csv.eof() )
        {
            std::vector< std::pair<float, float>> this_frame;
            for (int i=0; i<15; i++) {
                csv >> x ;
                csv >> ch;
                csv >> y;
                csv >> ch;
                this_frame.push_back(std::make_pair(x, y));
                csv >> x ;
                csv >> ch;
                csv >> y;
                csv >> ch;
            }
            m_gazes.push_back(this_frame);
        }
    }catch (...) {
        std::cout << "Error: cannot load eye tracking file." << std::endl;
        return false;
    }
    return true;
}