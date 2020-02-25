# SRFNet
A neural network architecture for super resolving extremely low-resolution optical flow for ego motion estimation.

Corresponding publication can be found [here](https://ieeexplore.ieee.org/abstract/document/8962229).

## Repo Breakdown

    - scripts: Contains matlab and python scripts for processing the different
            raw datasets we work with

        -- data: Scripts for generating rotation-compensated windows of kitti,
            euroc, and advio sequences

        -- visualization: Scripts for visualizing derotated windows

    - svo_matlab: Matlab implementation of inverse-compositional lucas kanade
        optical flow, which was originally intended to work in tandem with
        a gradient super resolution network

    - sr-pwc: Python (3.5.5) package for all the code used in our paper. Runs on
        Pytorch 1.0.1.post2, CudNN 7402, numpy version 1.14.6

        -- correlation_package : NVIDIA CUDA code for cost volume computation
            in the network

        -- networks.py : Network architectures including SRFNet (SRPWCNet), 
            SRResNet, PWCNet, etc.

        -- layers.py : Layer primitives used for defining network architectures

        -- data_utils.py : Code for loading datasets for training

        -- flow_utils.py : Code for handling different optical flow representations
                and generating corresponding visualizations

        -- test_models.py : Wrapper code around various networks used to define
                our baselines during evaluation

        -- scripts: Training and evaluation scripts, as well as trained models and results
            
            Naming Convention: 
                -- train_(model).py : Trains a network from scratch

                -- finetune_(model)_(dataset).py : Continues to train (model) from some
                        previous state on a new (dataset)

                -- evaluate_(dataset/models).py : Scripts for generating the tables 
                        in the experimental section of the paper. Corresponding
                        results listed in *_results/ folders. Summaries generated
                        with summarize_results.py/


            -- states: Folder containing trained models for each network. Each
                subfolder denotes the dataset the model was trained on. 
                
                Naming Convention: (model)_(epoch).pkl



## Installation

    Setup correlation_package:

    cd sr-pwc/correlation_package
    python setup.py install
