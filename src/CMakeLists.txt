# 告诉 CMake “我要把这些源文件编译并链接成一个可执行程序，名字叫 hello_cmake”
add_executable(hello_engine main.cc)

# 把可执行文件输出到顶层 build 目录
set_target_properties(hello_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

find_package(Threads REQUIRED)
target_link_libraries(hello_engine PRIVATE Threads::Threads)