#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <armadillo>
#include <cstring>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "src/parse_arguments.h"
#include "src/region_to_bbox.h"
#include "src/Tracker.h"
using namespace std;
using namespace Json;
using namespace cv;
using namespace boost;
using namespace arma;

#define MAX_FRAME 1000 // Max number of frames


filesystem::path dataset_folder;

inline double maxx(double a,double b) { if (a<b) return b; else return a;}
inline double minn(double a,double b) { if (a>b) return b; else return a;}

void _get_filename(Value &env, Value &evaluation, vector<string> &video_list)
{
    //acquire all datasets' filename
    video_list.clear();
    dataset_folder = "../" + env["root_dataset"].asString() + "/" + evaluation["dataset"].asString();
    if (filesystem::is_directory(dataset_folder)) {
        for (filesystem::directory_iterator iter(dataset_folder); iter != filesystem::directory_iterator(); ++iter) {
            filesystem::path p = *iter;
            if (filesystem::is_directory(p)) { video_list.emplace_back(p.leaf().string()); }
        }

    }
    else {  cout << "Dataset path doesn't exist!\n"; exit(0);  }
    sort (video_list.begin(), video_list.end());
}


void _init_video(Value &env, Value &evaluation, string &pic, vector<string> &frame_list, vector<vector<double>> &gt)
{
    // acquire all frames and groundtruth in a dataset file
    frame_list.clear();
    gt.clear();
    filesystem::path frames = dataset_folder.string() + "/" + pic;
    for (filesystem::directory_iterator iter(frames); iter != filesystem::directory_iterator(); ++iter) {
        filesystem::path p = *iter;
        if (p.leaf().string().find(".jpg") != -1) { frame_list.emplace_back(frames.string() + "/" + p.leaf().string()); }
    }
    sort (frame_list.begin(), frame_list.end());

    // read groundtruth
    ifstream in(frames.string() + "/groundtruth.txt");
    string line;
    if(in) {
        while (getline (in, line)) {
            vector<double> coordinate;
            string s="";
            for (auto i:line) {
                if (i == ',') {
                    coordinate.emplace_back(stod(s));
                    s = "";
                }
                else { s += i; }
            }
            coordinate.emplace_back(stod(s));
            gt.emplace_back(coordinate);
        }
        if (gt.size() != frame_list.size()) {  cout<<"Number of frames and number of GT lines should be equal."<<endl; exit(0);    }
    }
    else {  cout <<"groundtruth doesn't exist!" << endl; exit(0); }

}


inline double _compute_distance(double boxA[], double boxB[])
{
    double x1(boxA[0] + boxA[2]/2);
    double y1(boxA[1] + boxA[3]/2);
    double x2(boxB[0] + boxB[2]/2);
    double y2(boxB[1] + boxB[3]/2);
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

double _compute_iou(double boxA[], double boxB[])
{
    // determine the (x, y)-coordinates of the intersection rectangle
    double xA(maxx(boxA[0], boxB[0]));
    double yA(maxx(boxA[1], boxB[1]));
    double xB(minn(boxA[0] + boxA[2], boxB[0] + boxB[2]));
    double yB(minn(boxA[1] + boxA[3], boxB[1] + boxB[3]));
    double iou;
    if (xA < xB && yA < yB) {
        // compute the area of intersection rectangle
        double interArea = (xB - xA) * (yB - yA);
        // compute the area of both the prediction and ground-truth
        // rectangles
        double boxAArea = boxA[2] * boxA[3];
        double boxBArea = boxB[2] * boxB[3];
        // compute the intersection over union by taking the intersection
        // area and dividing it by the sum of prediction + ground-truth
        // areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea);
    } else {
        iou = 0;
    }

    if (!(iou>=0 && iou<=1.01)) cout<<"iou is out of range\n"<<endl;
    return iou;

}

void _compile_results(vector<vector<double>> &gt, int &start_frame, int total_frame, double bboxes[][4], int dist_threshold, double &precision, double &precisions_auc, double &iou, int &length)
{
    int len(total_frame - start_frame);
    int n_thresholds(50),i,j;
    double new_distances[MAX_FRAME];
    double new_ious[MAX_FRAME];
    double gt4[4];
    vec precisions_ths(n_thresholds);

    int sum1(0);
    double sum2(0);

    for (i=0; i<len; ++i) {
        region_to_bbox::bbox(gt4[0],gt4[1],gt4[2],gt4[3],gt[start_frame + i], false);

        new_distances[i] = _compute_distance(bboxes[i],gt4);
        new_ious[i] = _compute_iou(bboxes[i],gt4);

        sum2 += new_ious[i];
        if (new_distances[i] < dist_threshold) ++sum1;
    }

    // what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum1*1.0/len*100;

    // find above result for many thresholds, then report the AUC
    vec thresholds = linspace<vec>(0, 25, n_thresholds + 1);
    //reverse it so that higher values of precision goes at the beginning
    thresholds = reverse(thresholds);

    for (i=0; i< n_thresholds; ++i) {
        precisions_ths[i] = 0;
        for (j=0; j<len; ++j)
            if (new_distances[j] < thresholds[i]) ++precisions_ths[i];
        precisions_ths[i] /= len;
    }

    // integrate over the thresholds
    precisions_auc = as_scalar(trapz(precisions_ths));

    // per frame averaged intersection over union (OTB metric)
    iou = sum2/len*100;
    length = len;

}

int main()
{
    ios::sync_with_stdio(false);
    double pos_x, pos_y, target_w, target_h;
    int final_score_sz,i,j;
    vector<string> video_list, frame_list;
    vector<vector<double>> gt;
    Value hp, evaluation, run, env, design;
    parse_arguments::Parse(hp, evaluation, run, env, design);  //read parameters from json file
    final_score_sz = hp["response_up"].asInt() * (design["score_sz"].asInt() -1) + 1;
    if (evaluation["video"].asString() == "all") {
        _get_filename(env, evaluation, video_list);

        unsigned long total = video_list.size() * evaluation["n_subseq"].asInt();
        double bboxes[MAX_FRAME][4];
        double *speed = new double[total];
        double *precision = new double[total];
        double *precisions_auc = new double[total];
        double *ious = new double[total];
        int *length = new int[total];
        int idx(0);
        int n_subseq(evaluation["n_subseq"].asInt());
        int video_size(video_list.size());

        for (i=0; i< video_size; ++i) {       //you can set a small size to test
            _init_video(env, evaluation, video_list[i], frame_list, gt);
            vec starts = linspace<vec>(0, frame_list.size()-1, n_subseq + 1);
            for (j=0; j< 3; ++j) {
                int st(rint(starts[j]));
                region_to_bbox::bbox(pos_x, pos_y, target_w, target_h, gt[st]);
                Tracker::tracker(hp, run, design, frame_list, pos_x, pos_y, target_w, target_h, final_score_sz, st, speed[idx], bboxes);
                _compile_results(gt, st, frame_list.size(), bboxes, evaluation["dist_threshold"].asInt(), precision[idx], precisions_auc[idx], ious[idx], length[idx]);
                cout<<fixed<<setprecision(2);
                cout<<i<<" -- "<< video_list[i]<<" -- Precision: "<< precision[idx]<< " -- Precisions AUC: "<< precisions_auc[idx]
                    <<" -- IOU: "<<ious[idx]<< " -- Speed: "<<speed[idx]<<" --\n";
                ++idx;
            }
        }

        int tot_frames(0);
        double mean_precision(0), mean_precision_auc(0), mean_iou(0), mean_speed(0);
        for (j=0; j<idx; ++j) tot_frames += length[j];
        for (j=0; j<idx; ++j) {
            mean_precision += length[j] * precision[j];
            mean_precision_auc += length[j] * precisions_auc[j];
            mean_iou += length[j] * ious[j];
            mean_speed += length[j] * speed[j];
        }
        mean_precision /= tot_frames;
        mean_precision_auc /= tot_frames;
        mean_iou /= tot_frames;
        mean_speed /= tot_frames;

        cout<<"\n-- Overall stats (averaged per frame) on "<<  video_list.size() << " videos (" << tot_frames << " frames) --\n";
        cout<<" -- Precision "<< "(" << evaluation["dist_threshold"].asFloat() <<" px)" << ": "<<mean_precision
            <<" -- Precisions AUC: "<< mean_precision_auc
            <<" -- IOU: " << mean_iou
            <<" -- Speed: " << mean_speed << " --\n";

        delete []speed;
        delete []precision;
        delete []precisions_auc;
        delete []ious;

    } else {

    }
    return 0;
}
