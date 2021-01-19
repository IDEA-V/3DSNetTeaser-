git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus && mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=3.6 .. && make teaserpp_python
cd python && pip install .
cd ../../../
git clone https://github.com/zgojcic/3DSmoothNet.git
cd 3DSmoothNet
cmake -DCMAKE_BUILD_TYPE=Release .
make