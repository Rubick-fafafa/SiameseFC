//
// Created by yao on 2018/9/14.
//

#include "region_to_bbox.h"

double mean(vector<double> region, int p, int q)
{
    double sum(0);
    int num(0);
    while (p < region.size()) {
        sum += region[p];
        p += q;
        num++;
    }
    return sum/num;
}

double maxx(vector<double> region, int p, int q)
{
    double m(-10000000);
    while (p < region.size()) {
        if (region[p]>m) m = region[p];
        p += q;
    }
    return m;
}

double minn(vector<double> region, int p, int q)
{
    double m(10000000);
    while (p < region.size()) {
        if (region[p]<m) m = region[p];
        p += q;
    }
    return m;
}

double linalg_norm(vector<double> region, int p1, int q1, int p2, int q2)
{
    double temp[10],ans(0);
    int n(0);
    while (p1<q1) temp[n++]=region[p1++]-region[p2++];
    for (int i=0; i<=n; ++i) ans += temp[i]*temp[i];
    return sqrt(ans);
}

void region_to_bbox::bbox(double &pos_x, double &pos_y, double &target_w, double &target_h, vector<double> region, bool center)
{
    int len = region.size();
    if (len == 4) {
        if (center) {
            target_w = region[2];
            target_h = region[3];
            pos_x = region[0] + target_w/2;
            pos_y = region[1] + target_h/2;
        } else {
            pos_x = region[0];
            pos_y = region[1];
            target_w = region[2];
            target_h = region[3];
        }


    } else if (len == 8) {
        double cx(mean(region,0,2));
        double cy(mean(region,1,2));
        double x1(minn(region,0,2));
        double x2(maxx(region,0,2));
        double y1(minn(region,1,2));
        double y2(maxx(region,1,2));
        double A1(linalg_norm(region,0,2,2,4) * linalg_norm(region,2,4,4,6));
        double A2((x2 - x1) * (y2 - y1));
        double s(sqrt(A1/A2));
        target_w = s * (x2 - x1) + 1;
        target_h = s * (y2 - y1) + 1;
        if (center){
            pos_x = cx;
            pos_y = cy;
        } else {
            pos_x = cx - target_w/2;
            pos_y = cy - target_h/2;
        }
    } else {
        cout<<"GT region format is invalid, should have 4 or 8 entries."<<endl;
        exit(0);
    }
}