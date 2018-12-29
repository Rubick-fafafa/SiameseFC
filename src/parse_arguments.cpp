//
// Created by yao on 2018/9/14.
//

#include "parse_arguments.h"
#include <fstream>
#include <iostream>

Value load(string path)
{
    Reader reader;
    Value root;

    ifstream in(path, ios::binary);
    if( !in.is_open() ) {
        cout << "Error opening file " <<path<<endl;

    } else{
        if (!reader.parse(in,root)) {
            cout<<"Parse error\n";
        }
    }
    return root;

}

void parse_arguments::Parse(Value & hp,Value & evaluation, Value & run, Value & env, Value & design)
{
    hp = load("../parameters/hyperparams.json");
    evaluation = load("../parameters/evaluation.json");
    run = load("../parameters/run.json");
    env = load("../parameters/environment.json");
    design = load("../parameters/design.json");
}

