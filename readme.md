# SAR: Spatial-Aware Regression for 3D Hand Pose and Mesh Reconstruction from a Monocular RGB Image
![sar](https://github.com/zxz267/SAR/blob/main/assets/SAR.png)
## Introduction
This is the PyTorch implementation of ISMAR 2021 paper "SAR: Spatial-Aware Regression for 3D Hand Pose and Mesh Reconstruction from a Monocular RGB Image".
We provide our research code for training and testing our proposed method on FreiHAND dataset.

## Installation
### Requirements
- Python-3.7.11
- PyTorch-1.7.1
- torchvision-0.8.2
- pyrender-0.1.45 (for rendering mesh, please follow the [official installation guide](https://pyrender.readthedocs.io/en/latest/install/).) 

### Setup with Conda
We suggest to create a new conda environment and install all the relevant dependencies.

        conda create -n SAR python=3.7
        conda activate SAR
        pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r requirements.txt

### PyTorch MANO layer
- For the MANO layer, we used [manopth](https://github.com/hassony2/manopth). The repo is already included in `./manopth`.
- The MANO model file `MANO_RIGHT.pkl` from [this link](https://mano.is.tue.mpg.de/) is already included in `./manopth/mano/models`.

## Dataset
- Download FreiHAND dataset from [this link](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
- Download root joint coordinates from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/tree/7b9aabf93535deca01b988c9d6ef02f002ffe7c8) from 
[this link](https://drive.google.com/drive/folders/1OzwQG2ZutJ4Lzg5psilStsv_MO6-cysA).

You need to follow directory structure of the  ` data `  as below:

    ${ROOT}  
    |-- data  
    |   |-- FreiHAND
    |   |   |-- training
    |   |   |   |-- rgb
    |   |   |   |-- mask
    |   |   |-- evaluation
    |   |   |   |-- rgb
    |   |   |-- evaluation_K.json
    |   |   |-- evaluation_scale.json
    |   |   |-- training_K.json
    |   |   |-- training_scale.json
    |   |   |-- training_mano.json
    |   |   |-- training_xyz.json
    |   |   |-- training_verts.json
    |   |   |-- bbox_root_freihand_output.json
        
## Training
1. Modify `./config.py` to specify the model and parameters for the training.
2. Run code `python train.py`.

We provide a [training log example](https://raw.githubusercontent.com/zxz267/SAR/main/output/log/train_SAR_resnet34_Stage2_Batch64_lr0.0003_Size256_Epochs50.log).

## Evaluation
1. Modify `./config.py` to specify the path of the trained model's weights in "checkpoint" and the the corresponding model parameters.
2. Run code `python test.py` (if visualize mesh, `PYOPENGL_PLATFORM=osmesa python test.py
`).
3. Zip `./output/pred.json` and submit the prediction zip file to 
[FreiHAND Leaderboard](https://competitions.codalab.org/competitions/21238) to obtain the evaluation scores.

To reprodece our results, we provide the [pretrained model](https://drive.google.com/file/d/1j9gUbXor-FuX_YH1fptSTE1DVSjjnZRp/view?usp=sharing) 
(using ResNet-34 as backbone and two stages) and the corresponding [prediction file](https://drive.google.com/file/d/16oQKiDwEKOFjn6Gq7u8AiJjvpAvCA75w/view?usp=sharing). 
This pretrained model should generate the following results:

    Evaluation 3D KP results:
    auc=0.229, mean_kp3d_avg=6.14 cm
    Evaluation 3D KP ALIGNED results:
    auc=0.871, mean_kp3d_avg=0.65 cm

    Evaluation 3D MESH results:
    auc=0.228, mean_kp3d_avg=6.14 cm
    Evaluation 3D MESH ALIGNED results:
    auc=0.866, mean_kp3d_avg=0.67 cm

    F-scores
    F@5.0mm = 0.130 	F_aligned@5.0mm = 0.724
    F@15.0mm = 0.392 	F_aligned@15.0mm = 0.981

## Acknowledgement
We borrowed a part of the open-source code of [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/tree/7b9aabf93535deca01b988c9d6ef02f002ffe7c8). 



        

        
    
    
    


