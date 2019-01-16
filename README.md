# SiameseFC
C++ implementation version

This code doesn't include training part and it is base on [*torrvision/siamfc-tf*](https://github.com/torrvision/siamfc-tf)

I used the python version to generate a **"Score.pb"** model which include all parameters in the tensor graph. Then we invoke it in tensorflow C++ to track

# Run

* "libs" includes tensorFlow C++ environment that I build from the tensorflow source code
* The whole structure is the same as the python version. You can refer to its README
* run "run_tracker.cpp" to track


# Result

C++ implementation is not as fast as I predict. SiameseFC is unable to track online. About 2.3 fps on Macbook Pro 2017. 

# Reference
https://github.com/torrvision/siamfc-tf

https://www.octadero.com/2017/08/27/tensorflow-c-environment

http://blog.blitzblit.com/2017/06/11/creating-tensorflow-c-headers-and-libraries

https://spockwangs.github.io/blog/2018/01/13/train-using-tensorflow-c-plus-plus-api/


# License
This code can be freely used for personal, academic, or educational purposes.
Please contact me for commercial use.
