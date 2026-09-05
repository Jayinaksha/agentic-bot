cmake_minimum_required(VERSION 3.8)
project(r2d2_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds config launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
