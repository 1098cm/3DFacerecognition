# 3DFacerecognition
Pytorch implementation of "Toward three-dimensional face recognition via adaptive optimization fusing monocular images and geometric structures"  

We propose a method to generate the geometric structure maps of human faces through the parametric face reconstruction model and achieve a high accuracy rate of 3D face recognition based on the adaptive optimization of the feature fusion of monocular images and geometric shape maps, providing a new perspective for the research of 3D face recognition technology.  

## Environment
1.New environment  
`conda create -n 3DFaceRecog python=3.8`  

2.Install PyTorch  
`conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch`

3.Insatll [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)   

4.face-alignment  

5.Other libraries  
`pip install tqdm bcolz-zipline prettytable menpo mxnet`  

If you still don't know how to install it, please refer to the installation environments of [DECA](https://github.com/yfeng95/DECA), [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch), [AdaFace](https://github.com/mk-minchul/AdaFace) and [CosFace](https://github.com/yule-li/CosFace) methods.  

## Prepare data
1.Download the training set: CASIA-WebFace  

Testing set: Texas3DFRD, Casia3D, Facescrub  

2.The depth map and detailed shape map are obtained by using the [DECA](https://github.com/yfeng95/DECA) method.  

## Evaluation
1.Download the trained weights and place them in the designated folder; Similarly, place the original image and the depth map or detail shape map generated by its reconstruction in the specified folder. The trained weights can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/11D98URQAu-ZIfZ70M43CLg) 提取码: ikqd.  

2.Modify the content of weight loading and data loading in the code and run the following program:  
`python eval.py`  

## Training
The training code is coming soon.  

## Acknowledgements
Here are some great resources we benefit:  

[face-alignment](https://github.com/1adrianb/face-alignment) for cropping  

[DECA](https://github.com/yfeng95/DECA) for 3D data  

[AdaFace](https://github.com/mk-minchul/AdaFace), [CosFace](https://github.com/yule-li/CosFace) and [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) for loss function  
