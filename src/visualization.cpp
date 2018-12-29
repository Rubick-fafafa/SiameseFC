//
// Created by yao on 2018/9/27.
//

#include "visualization.h"

void visualization::show_frame(string filename,double bbox[])
{
    Mat image;
    image = imread(filename);
    rectangle(image, Point(bbox[0],bbox[1]), Point(bbox[0]+bbox[2],bbox[1]+bbox[3]), Scalar(0,0,255));
    imshow("Tracking", image);
    waitKey(1);

}