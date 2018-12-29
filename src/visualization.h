//
// Created by yao on 2018/9/27.
//

#ifndef CFNET_VISUALIZATION_H
#define CFNET_VISUALIZATION_H
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class visualization {
public:
    static void show_frame(string filename,double bbox[]);

};


#endif //CFNET_VISUALIZATION_H
