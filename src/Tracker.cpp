//
// Created by yao on 2018/9/17.
//

#include "Tracker.h"
#include "visualization.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/array_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;
using namespace chrono;



inline void _update_target_position(double &pos_x, double &pos_y, mat &score_, int &final_score_sz, int &tot_stride, int &search_sz, int &response_up, double &x_sz)
{
    int p[2];
    double disp_in_area[2],disp_in_xcrop[2],disp_in_frame[2];
    double temp1(double(tot_stride) / response_up),temp2(x_sz / search_sz);
    // find location of score maximizer
    Tracker::find_max_coordinate(score_, p);
    // displacement from the center in search area final representation ...
    double center((final_score_sz - 1) / 2.0);
    disp_in_area[0] = p[0] - center;
    disp_in_area[1] = p[1] - center;
    // displacement from the center in instance crop
    disp_in_xcrop[0] = disp_in_area[0] * temp1;
    disp_in_xcrop[1] = disp_in_area[1] * temp1;
    // displacement from the center in instance crop (in frame coordinates)
    disp_in_frame[0] = disp_in_xcrop[0] * temp2;
    disp_in_frame[1] = disp_in_xcrop[1] * temp2;
    // *position* within frame in frame coordinates
    pos_x += disp_in_frame[1];
    pos_y += disp_in_frame[0];
}



void Tracker::tracker(Value hp, Value run, Value design, vector<string> frame_list, double pos_x, double pos_y,
                      double target_w, double target_h, int final_score_sz, int start_frame, double &speed ,double bboxes[][4])
{


    // get all parameters we need
    float z_lr(hp["z_lr"].asFloat());
    int response_up(hp["response_up"].asInt());
    int search_sz(design["search_sz"].asInt());
    int exemplar_sz(design["exemplar_sz"].asInt());
    int tot_stride(design["tot_stride"].asInt());
    int i,j,a,b,c,d,dim_z1,dim_z2,dim_z3,sf,size2;
    int scale_exponent[3] = {-2, 0, 2};
    double scale_lr(hp["scale_lr"].asDouble());
    double scale_step(hp["scale_step"].asDouble());
    double scale_penalty(hp["scale_penalty"].asDouble());
    double window_influence(hp["window_influence"].asDouble());
    double one_sub_lr = 1 - scale_lr;
    double one_sub_zlr = 1 - z_lr;
    double scale_factors[3],scaled_exemplar[3],scaled_search_area[3],scaled_target_w[3],scaled_target_h[3];


    // search the target in three scales (a litte bit smaller, the same size, a little bit bigger )
    scale_factors[0] = pow(scale_step, scale_exponent[0]);
    scale_factors[1] = pow(scale_step, scale_exponent[1]);
    scale_factors[2] = pow(scale_step, scale_exponent[2]);

    mat hann_1d(1,final_score_sz);

    // cosine window to penalize large displacements
    Tracker::hanning(hann_1d, final_score_sz);
    mat penalty(hann_1d.t() * hann_1d);
    penalty /= accu(penalty);

    double context = design["context"].asDouble()*(target_w+target_h);
    double z_sz = sqrt((target_w+context)*(target_h+context));
    double x_sz = double(search_sz) / exemplar_sz * z_sz;

    bboxes[0][0] = pos_x - target_w/2;
    bboxes[0][1] = pos_y - target_h/2;
    bboxes[0][2] = target_w;
    bboxes[0][3] = target_h;

    // import tensorflow graph
    string graph_path("../pretrained/Score.pb");
    Status status;
    Session* session;
    GraphDef graph_def;
    Scope root = Scope::NewRootScope();
    ClientSession cli_sess(root);
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_path, &graph_def));
    TF_CHECK_OK(NewSession(SessionOptions(), &session));
    TF_CHECK_OK(session->Create(graph_def));

    vector<pair<string, Tensor>> input;
    vector<Tensor> output;

    // These tensors are supposed to be feeded
    Tensor Pos_x(DT_DOUBLE, TensorShape());
    Tensor Pos_y(DT_DOUBLE, TensorShape());
    Tensor Z_sz(DT_DOUBLE, TensorShape());
    Tensor X_sz0(DT_DOUBLE, TensorShape());
    Tensor X_sz1(DT_DOUBLE, TensorShape());
    Tensor X_sz2(DT_DOUBLE, TensorShape());
    Tensor Filename(DT_STRING, TensorShape());

    // Initialize the tensor
    Pos_x.scalar<double>()() = pos_x;
    Pos_y.scalar<double>()() = pos_y;
    Z_sz.scalar<double>()() = z_sz;
    Filename.scalar<string>()() = frame_list[start_frame];


    input.emplace_back(string("pos_x_ph"), Pos_x);
    input.emplace_back(string("pos_y_ph"), Pos_y);
    input.emplace_back(string("z_sz_ph"), Z_sz);
    input.emplace_back(string("filename"), Filename);

    vector<string> output_layer ={"templates_z:0"};
    status = session->Run(input, output_layer, {}, &output);
    Tensor templates_z_(output[0]);
    Tensor new_templates_z_(templates_z_);


    auto t_start = system_clock::now();

    // Get an image from the queue
    int fz(frame_list.size());
    for (i=start_frame+1; i<fz; ++i) {

        sf = i - start_frame;
        scaled_exemplar[0] = z_sz * scale_factors[0];
        scaled_exemplar[1] = z_sz * scale_factors[1];
        scaled_exemplar[2] = z_sz * scale_factors[2];

        scaled_search_area[0] = x_sz * scale_factors[0];
        scaled_search_area[1] = x_sz * scale_factors[1];
        scaled_search_area[2] = x_sz * scale_factors[2];

        scaled_target_w[0] = target_w * scale_factors[0];
        scaled_target_w[1] = target_w * scale_factors[1];
        scaled_target_w[2] = target_w * scale_factors[2];


        scaled_target_h[0] = target_h * scale_factors[0];
        scaled_target_h[1] = target_h * scale_factors[1];
        scaled_target_h[2] = target_h * scale_factors[2];


        Pos_x.scalar<double>()() = pos_x;
        Pos_y.scalar<double>()() = pos_y;
        X_sz0.scalar<double>()() = scaled_search_area[0];
        X_sz1.scalar<double>()() = scaled_search_area[1];
        X_sz2.scalar<double>()() = scaled_search_area[2];
        Filename.scalar<string>()() = frame_list[i];


        //auto temp = Squeeze(root, templates_z_);
        //TF_CHECK_OK(cli_sess.Run({temp}, &output));
        //templates_z_ = output[0];


        input.clear();
        input.emplace_back(string("pos_x_ph"), Pos_x);
        input.emplace_back(string("pos_y_ph"), Pos_y);
        input.emplace_back(string("x_sz0_ph"), X_sz0);
        input.emplace_back(string("x_sz1_ph"), X_sz1);
        input.emplace_back(string("x_sz2_ph"), X_sz2);
        input.emplace_back(string("templates_z"), templates_z_);
        input.emplace_back(string("filename"), Filename);

        status = session->Run(input, {"ScoreMap:0"}, {}, &output);

        {
            auto temp = Squeeze(root, output[0]);
            TF_CHECK_OK(cli_sess.Run({temp}, &output));
        }

        Tensor scores_(output[0]);



        // penalize change of scale and find scale with hightest peak (after penalty)
        dim_z1 = scores_.shape().dim_size(1);
        dim_z2 = scores_.shape().dim_size(2);
        int new_scale_id(1);
        double peak(-10000000.0);
        auto  scores_map = scores_.tensor<float,3>();
        for (a = 0; a< dim_z1; ++a) {
            for (b = 0; b < dim_z2; ++b) {
                scores_map(0, a, b) *= scale_penalty;
                scores_map(2, a, b) *= scale_penalty;
                if (scores_map(0, a, b) > peak) {
                    peak = scores_map(0, a, b);
                    new_scale_id = 0;
                }
                if (scores_map(2, a, b) > peak) {
                    peak = scores_map(2, a, b);
                    new_scale_id = 2;
                }
                if (scores_map(1, a, b) >= peak) {
                    peak = scores_map(1, a, b);
                    new_scale_id = 1;
                }
            }
        }

        // update scaled sizes
        x_sz *= one_sub_lr;
        x_sz += scale_lr * scaled_search_area[new_scale_id];
        target_w *= one_sub_lr;
        target_w += scale_lr * scaled_target_w[new_scale_id];
        target_h *= one_sub_lr;
        target_h += scale_lr * scaled_target_h[new_scale_id];

        // select response with new_scale_id
        mat score_(dim_z1,dim_z2);

        for (a = 0; a< dim_z1; ++a) {
            for (b = 0; b < dim_z2; ++b) {
                score_(a,b) = scores_map(new_scale_id, a, b);
            }
        }
        score_ -= score_.min();
        score_ /= accu(score_);

        // apply displacement penalty
        score_ *= 1 - window_influence;
        score_ += window_influence * penalty;

        _update_target_position(pos_x, pos_y, score_, final_score_sz, tot_stride, search_sz, response_up, x_sz);

        // convert <cx,cy,w,h> to <x,y,w,h> and save output
        bboxes[sf][0] = pos_x - target_w/2;
        bboxes[sf][1] = pos_y - target_h/2;
        bboxes[sf][2] = target_w;
        bboxes[sf][3] = target_h;

        // update the target representation with a rolling average
        if (z_lr > 0 ) {
            Pos_x.scalar<double>()() = pos_x;
            Pos_y.scalar<double>()() = pos_y;
            Z_sz.scalar<double>()() = z_sz;
            Filename.scalar<string>()() = frame_list[i];

            input.clear();
            input.emplace_back(string("pos_x_ph"), Pos_x);
            input.emplace_back(string("pos_y_ph"), Pos_y);
            input.emplace_back(string("z_sz_ph"), Z_sz);
            input.emplace_back(string("filename"), Filename);

            status = session->Run(input, {"templates_z:0"}, {}, &output);
            new_templates_z_ = output[0];


            {
                auto map1 = new_templates_z_.tensor<float,4>();
                auto map2 = templates_z_.tensor<float,4>();
                dim_z1 = new_templates_z_.shape().dim_size(1);
                dim_z2 = new_templates_z_.shape().dim_size(2);
                dim_z3 = new_templates_z_.shape().dim_size(3);

                for (b = 0; b < dim_z1; ++b) {
                    for (c = 0; c < dim_z2; ++c) {
                        for (d = 0; d < dim_z3; ++d) {
                            map2(0, b, c, d) *= one_sub_zlr;
                            map2(0, b, c, d) += map1(0, b, c, d) * z_lr;
                            map2(1, b, c, d) *= one_sub_zlr;
                            map2(1, b, c, d) += map1(1, b, c, d) * z_lr;
                            map2(2, b, c, d) *= one_sub_zlr;
                            map2(2, b, c, d) += map1(2, b, c, d) * z_lr;

                        }
                    }
                }

            }


        }

        // update template patch size
        z_sz *= one_sub_lr;
        z_sz += scale_lr * scaled_exemplar[new_scale_id];

        // show tracking result
        if (run["visualization"].asBool()) {
            visualization::show_frame(frame_list[i] ,bboxes[i]);
        }


/*
        Tensor result = scores_;
        int a = result.shape().dim_size(0);
        int b = result.shape().dim_size(1);
        int c = result.shape().dim_size(2);
        int d = result.shape().dim_size(3);
        cout<<a<<" "<<b<<" "<<c<<" "<<d<<endl;
        auto result_map = result.tensor<float,3>();
        cout<<"result: "<<result_map(0,0,1)<<endl;
*/
    }

    // caculate the speed (fps)
    auto t_end = system_clock::now();
    auto duration = duration_cast<microseconds>(t_end - t_start);
    double spend_time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    speed = (frame_list.size() - start_frame)/spend_time;

}