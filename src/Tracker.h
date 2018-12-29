//
// Created by yao on 2018/9/17.
//

#ifndef CFNET_TRACKER_H
#define CFNET_TRACKER_H

#include <armadillo>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <json/json.h>

using namespace std;
using namespace Json;
using namespace arma;

class Tracker {
public:
    static void tracker(Value hp, Value run, Value design, vector<string> frame_list, double pos_x, double pos_y,
                        double target_w, double target_h, int final_score_sz, int start_frame, double &speed, double bboxes[][4]);


    static void hanning(mat &han,int &m)
    {
        const long double Pi(3.14159265358979323846264338327950288419716939937510);
        for (int i=0;i<m;i++) {
            han(0,i)= 0.5 - 0.5*cos(2.0*Pi*i/(m-1));
        }
    }

    static void find_max_coordinate(mat &matrix, int p[])
    {
        double maxx(-10000000.0);
        int x(matrix.n_rows), y(matrix.n_cols), i, j;
        for (i=0; i<x; ++i)
            for (j=0; j<y; ++j)
                if (matrix(i,j) > maxx) {
                    maxx = matrix(i,j);
                    p[0] = i;
                    p[1] = j;
                }
    }
};


#endif //CFNET_TRACKER_H
