//
// Created by yao on 2018/9/14.
//

#ifndef CFNET_PARSE_ARGUMENTS_H
#define CFNET_PARSE_ARGUMENTS_H
#include <json/json.h>

using namespace std;
using namespace Json;

class parse_arguments {
public:
    static void Parse(Value & hp,Value & evaluation, Value & run, Value & env, Value & design);
};


#endif //CFNET_PARSE_ARGUMENTS_H
