cmake_minimum_required(VERSION 3.12)
project(SiameseFC)

find_package(OpenCV)
find_package(JsonCPP)
find_package(Boost 1.67.0 REQUIRED COMPONENTS system filesystem date_time)
find_package(Armadillo REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${JsonCPP_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${ARMADILLO_INCLUDE_DIRS} )
include_directories(
        /usr/local/include/eigen3
        ./libs/include
        ./libs/include/tensorflow
        ./libs/include/bazel-genfiles
        ./libs/include/protobuf-inc
        ./libs/include/nsync-public)

set(CMAKE_CXX_STANDARD 14)

link_directories(./libs/lib)

add_executable(SiameseFC run_tracker.cpp src/region_to_bbox.cpp src/region_to_bbox.h src/parse_arguments.cpp src/parse_arguments.h src/Tracker.cpp src/Tracker.h src/visualization.cpp src/visualization.h)

target_link_libraries( SiameseFC ${OpenCV_LIBS} )
target_link_libraries( SiameseFC JsonCPP)
target_link_libraries( SiameseFC ${Boost_LIBRARIES})
target_link_libraries( SiameseFC tensorflow_cc tensorflow_framework)
target_link_libraries( SiameseFC ${ARMADILLO_LIBRARIES})