cmake_minimum_required(VERSION 3.10)

set (PROJECT_NAME clap)
set (SRC_FILES ${PROJECT_NAME}.cpp clap_utils.cpp)
set (SRC_FILES ${SRC_FILES} ../../util/utils.cpp)
set (SRC_FILES ${SRC_FILES} ../../util/wave_reader.cpp)


project(${PROJECT_NAME} CXX)

include_directories(${INCLUDE_PATH})
link_directories(${LIBRARY_PATH})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_link_libraries(${PROJECT_NAME} ailia ailia_tokenizer ailia_audio)
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
