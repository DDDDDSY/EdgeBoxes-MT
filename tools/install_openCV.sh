cores=$1
if [ -z $1 ]
then
    echo "Please specify a core count to use for make -j# ie. ./install_openCV.sh 4"
    exit
fi

# Install prereqs
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev python3-dev
sudo apt-get install python3-pip
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev

# Unified directory for building
mkdir open_cv
cd open_cv

# Download source
wget https://codeload.github.com/opencv/opencv/zip/3.4.0 -O opencv-3.4.0.zip
wget https://codeload.github.com/opencv/opencv_contrib/zip/3.4.0 -O opencv_contrib-3.4.0.zip

# unzip
unzip opencv-3.4.0.zip
unzip opencv_contrib-3.4.0.zip

# python prereqs
pip3 install numpy

# cmake
cd opencv-3.4.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
      -D WITH_FFMPEG=1 \
      -D BUILD_EXAMPLES=ON  ..

echo "Does the cmake output look good? y/n"
read input
if [ $input != "y" ]
then
    exit
fi

make -j$cores
echo "Does the make output look good? y/n"
read input2
if [ $input2 != "y" ]
then
    exit
fi
sudo make install
sudo ldconfig

echo "Installation complete! Feel free to delete open_cv/"
