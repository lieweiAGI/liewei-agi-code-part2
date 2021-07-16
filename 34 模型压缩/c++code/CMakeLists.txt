cmake_minimum_required(VERSION 3.15)
project(DeepLearningStudy)

set(CMAKE_PREFIX_PATH D:/lib/libtorch)
set(OpenCV_DIR D:/lib/opencv/build/x64/vc15/lib)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(DeepLearningStudy  main.cpp)
target_link_libraries(DeepLearningStudy ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET DeepLearningStudy PROPERTY CXX_STANDARD 11)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET DeepLearningStudy
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:DeepLearningStudy>)

    file(GLOB OPENCV_DLLS "D:/lib/opencv/build/x64/vc15/bin/*.dll")
    add_custom_command(TARGET DeepLearningStudy
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:DeepLearningStudy>)
endif (MSVC)