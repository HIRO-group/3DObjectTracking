cmake_minimum_required(VERSION 3.5)

add_executable(evaluate_rbot_dataset evaluate_rbot_dataset.cpp rbot_evaluator.cpp rbot_evaluator.h)
target_link_libraries(evaluate_rbot_dataset PUBLIC m3t)

add_executable(parameter_study_rbot_dataset parameter_study_rbot_dataset.cpp rbot_evaluator.cpp rbot_evaluator.h)
target_link_libraries(parameter_study_rbot_dataset PUBLIC m3t)

add_executable(evaluate_choi_dataset evaluate_choi_dataset.cpp choi_evaluator.cpp choi_evaluator.h)
target_link_libraries(evaluate_choi_dataset PUBLIC m3t)

add_executable(parameter_study_choi_dataset parameter_study_choi_dataset.cpp choi_evaluator.cpp choi_evaluator.h)
target_link_libraries(parameter_study_choi_dataset PUBLIC m3t)

add_executable(evaluate_opt_dataset evaluate_opt_dataset.cpp opt_evaluator.cpp opt_evaluator.h)
target_link_libraries(evaluate_opt_dataset PUBLIC m3t)

add_executable(parameter_study_opt_dataset parameter_study_opt_dataset.cpp opt_evaluator.cpp opt_evaluator.h)
target_link_libraries(parameter_study_opt_dataset PUBLIC m3t)

add_executable(evaluate_ycb_dataset evaluate_ycb_dataset.cpp ycb_evaluator.cpp ycb_evaluator.h)
target_link_libraries(evaluate_ycb_dataset PUBLIC m3t)

add_executable(parameter_study_ycb_dataset parameter_study_ycb_dataset.cpp ycb_evaluator.cpp ycb_evaluator.h)
target_link_libraries(parameter_study_ycb_dataset PUBLIC m3t)

add_executable(evaluate_rtb_dataset evaluate_rtb_dataset.cpp rtb_evaluator.cpp rtb_evaluator.h)
target_link_libraries(evaluate_rtb_dataset PUBLIC m3t)

add_executable(parameter_study_rtb_dataset parameter_study_rtb_dataset.cpp rtb_evaluator.cpp rtb_evaluator.h)
target_link_libraries(parameter_study_rtb_dataset PUBLIC m3t)

add_executable(run_on_recorded_sequence run_on_recorded_sequence.cpp)
target_link_libraries(run_on_recorded_sequence PUBLIC m3t)

add_executable(run_generated_tracker run_generated_tracker.cpp)
target_link_libraries(run_generated_tracker PUBLIC m3t)

add_executable(run_pen_paper_demo run_pen_paper_demo.cpp)
target_link_libraries(run_pen_paper_demo PUBLIC m3t)

add_executable(constraint_convergence constraint_convergence.cpp)
target_link_libraries(constraint_convergence PUBLIC m3t)

add_executable(optimization_time optimization_time.cpp)
target_link_libraries(optimization_time PUBLIC m3t)

if (USE_AZURE_KINECT)
    add_executable(record_camera_sequence record_camera_sequence.cpp)
    target_link_libraries(record_camera_sequence PUBLIC m3t)

    add_executable(run_on_camera_sequence run_on_camera_sequence.cpp)
    target_link_libraries(run_on_camera_sequence PUBLIC m3t)
endif ()
# Append these lines to 3DObjectTracking/M3T/examples/CMakeLists.txt
# Debian/Ubuntu dependencies: libzmq3-dev cppzmq-dev nlohmann-json3-dev
find_path(CPPZMQ_INCLUDE_DIR zmq.hpp)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
find_library(ZMQ_LIBRARY NAMES zmq)
if(NOT CPPZMQ_INCLUDE_DIR OR NOT NLOHMANN_JSON_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
    message(FATAL_ERROR "run_newton_stream requires cppzmq, nlohmann-json and libzmq")
endif()
add_executable(run_newton_stream run_newton_stream.cpp)
target_include_directories(run_newton_stream PRIVATE
    ${CPPZMQ_INCLUDE_DIR} ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(run_newton_stream PRIVATE m3t ${ZMQ_LIBRARY})
