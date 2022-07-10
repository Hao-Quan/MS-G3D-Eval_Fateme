# ****** evaluates MS-G3D by Heirarchical Prediction Model

******: A Large-scale Dataset for Human Activity Recognition In The Wild

## Dependencies

- Python >= 3.6
- PyTorch >= 1.2.0
- [NVIDIA Apex](https://github.com/NVIDIA/apex) (auto mixed precision training)
- PyYAML, tqdm, tensorboardX

### Download Datasets

- ****** Skeleton

### Data Preprocessing

#### Directory Structure

Put downloaded data into the following directory structure:

```
-******/
    -data_reduce_valid/
      - train_data_joint.npy
      - train_label.pkl
    -data_reduce_valid_crop/
      - train_data_joint.npy
      - train_label.pkl
```


## Pretrained Models

- Download pretrained models for producing the final results on ******: [[Dropbox](https://www.dropbox.com/sh/5ov5l0bsmu54cld/AADHI54otwfxgZtxnXwj_skwa?dl=0)]


- Put the folder of pretrained models at repo root:

```
- MS-G3D/
  - pretrained-models/
  - main.py
  - ...
```


## Training & Testing

- The general training template command:
```
python3 main.py
  --config <config file>
  --work-dir <place to keep things (weights, checkpoints, logs)>
  --device <GPU IDs to use>
  --half   # Mixed precision training with NVIDIA Apex (default O1) for GPUs ~11GB memory
  [--base-lr <base learning rate>]
  [--batch-size <batch size>]
  [--weight-decay <weight decay>]
  [--forward-batch-size <batch size during forward pass, useful if using only 1 GPU>]
  [--eval-start <which epoch to start evaluating the model>]
```

- The general testing template command:
```
python3 main.py
  --config <config file>
  --work-dir <place to keep things>
  --device <GPU IDs to use>
  --weights <path to model weights>
  [--test-batch-size <...>]
```

- Use the corresponding config files from `./config` to train/test different datasets

- Examples
  - Train on -****** Joint
    - Train with 1 GPU:
      - `python3 main.py --config ./config/-******/train_joint_remote.yaml`
    - Train with 2 GPUs:
      - `python3 main.py --config ./config/-******/train_joint_remote.yaml --batch-size 32 --forward-batch-size 32 --device 0 1`
  - Test on -****** Joint
    - `python3 main.py --config ./config/-******/test_joint_remote.yaml`
    

## Acknowledgements

This repo is based on
  - [MS-G3D](https://github.com/kenziyuliu/MS-G3D)

Thanks to the original authors for their work!


## Citation

Please cite this work if you find it useful.


## Contact
Please email for further questions
