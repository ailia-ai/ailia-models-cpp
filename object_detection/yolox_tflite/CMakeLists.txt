cmake_minimum_required(VERSION 3.1)

set (PROJECT_NAME yolox_tflite)
set (SRC_FILES ${PROJECT_NAME}.cpp)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

project(${PROJECT_NAME} CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${OpenCV_LIBRARY_DIRS} ${LIBRARY_PATH})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_link_libraries(${PROJECT_NAME} ailia_tflite ${OpenCV_LIBRARIES})
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
