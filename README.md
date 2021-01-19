# 3DSNetTeaser++
## Install
If you don't have PCL, install it by
```
./install_pcl.sh
```
For C dependencies
```
./install_c.sh
```
Install 3DSmoothNet and TEASER-plusplus and compile them
```
./install.sh
```
Create environment with all python dependencies with conda and switch to it
```
conda env create -f env.yaml
conda activate 3dsmooth
```
Now you are ready to use

## Usage
Run pipeline.py with number of threads you want.
```
 OMP_NUM_THREADS=16 python pipeline.py
```

## Tips
Always remember to transform your .ply file to ascii form, otherwise 3DSmoothNet can't read them.
Most of time, you don't generate keypoints more than 20000, otherwise it either is too slow or just cracks.
Too less threads lead to abortion or segmentation fault.
