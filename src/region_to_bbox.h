//
// Created by yao on 2018/9/14.
//

#ifndef CFNET_REGION_TO_BBOX_H
#define CFNET_REGION_TO_BBOX_H
#include<vector>
#include<iostream>
#include<cmath>
using namespace std;

class region_to_bbox {
public:
    static void bbox(double &pos_x, double &pos_y, double &target_w, double &target_h, vector<double> region, bool center=true) ;
};


#endif //CFNET_REGION_TO_BBOX_H
