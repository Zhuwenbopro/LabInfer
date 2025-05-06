rm -rf build/
mkdir build && cd build
cmake ..
cmake --build .
cd ..
./hello_engine