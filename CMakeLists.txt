cmake_minimum_required(VERSION 2.8.3)
project(shadowpiano)

add_compile_options(-std=c++11)
add_compile_options(-lpthread)

find_package(catkin REQUIRED COMPONENTS	roscpp shadowlibs)

# Eigen
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
	message(STATUS "Eigen3 found")
else ()
	message(FATAL_ERROR "Eigen3 not found")
endif ()

include_directories(
		include
		${catkin_INCLUDE_DIRS}
		${EIGEN3_INCLUDE_DIRS}
)

catkin_python_setup()

catkin_package(
		INCLUDE_DIRS include
		CATKIN_DEPENDS
		roscpp shadowlibs
)
#-------------------------------------------------------------------------#

# Shadow Alignment Task
add_executable(piano_player src/shadow_piano.cpp)
target_link_libraries(piano_player ${catkin_LIBRARIES})