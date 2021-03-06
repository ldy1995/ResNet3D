# ResNet3D
This is a ResNet-based post-processing 3D network for reducing the low-dose noise and artifacts.

## Requirements

* Linux Platform
* NVIDIA GPU + CUDA 
* PyTorch == 1.0.1
* Python3.6
* Numpy1.16
* Scipy1.2

## Dataset Descriptions

* CT scanning data of ten patients from [Mayo Clinic](http://www.aapm.org/GrandChallenge/LowDoseCT/) 

## Results

The following three images show current results of the network for restoring the low-dose CT images at  transverse, coronal and sagittal planes, respectively.

![Fig. 1. Results of transverse planes.](./img/ResNet3D_trans.png)

![Fig. 2. Results of coronal planes.](./img/ResNet3D_Coron.png)

![Fig. 3. Results of sagittal planes](./img/ResNet3D_Sagit.png)

## Contact

If you have any question, please feel free to contact Danyang Li (Email: lidanyang1995@smu.edu.cn).
