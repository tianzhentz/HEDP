# MDEP
This repository is our PyTorch implementation of HDEP.

## Requirements

Note that only NVIDIA GPUs are supported for now, and we use NVIDIA RTX 3090. 

```shell
#Create the virtual environment
conda env create -f coda_env.yaml
```

Download the following three datasets to the ./data folder:

[CDDB](https://github.com/Coral79/CDDB) 

[CORe50](https://vlomonaco.github.io/core50/index.html#dataset)  

[DomainNet](http://ai.bu.edu/M3SDA/)  

```
CDDB
├── biggan
│   ├── train
│   └── val
├── gaugan
│   ├── train
│   └── val
├── san
│   ├── train
│   └── val
├── whichfaceisreal
│   ├── train
│   └── val
├── wild
│   ├── train
│   └── val
... ...
```

```
core50
└── core50_128x128
    ├── labels.pkl
    ├── LUP.pkl
    ├── paths.pkl
    ├── s1
    ├── s2
    ├── s3
    ...
```

```
domainnet
├── clipart
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── clipart_test.txt
├── clipart_train.txt
├── infograph
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── infograph_test.txt
├── infograph_train.txt
├── painting
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
... ...
```


## How to train

You can train the HDEP with the following commands:

```shell
# CDDB-Hard
nohup python main.py --config configs/train/cddb-hard.json >> logs/train/cddb.log 2>&1 &


# CORe50
nohup python main.py --config configs/train/core50.json >> logs/train/core50.log 2>&1 &


# Domainnet
nohup python main.py --config configs/train/domainnet.json >> logs/train/domainnet.log 2>&1 &

```

## How to eval

You can eval the HDEP with the following commands. Add the path of the trained model to the evaluation configuration file, paying attention to distinguishing between known and unknown domains:

```shell
# CDDB-Hard
nohup python main.py --config configs/eval/known/cddb-hard.json >> logs/eval/known/cddb.log 2>&1 &


# CORe50
nohup python main.py --config configs/eval/unknown/core50.json >> logs/eval/unknown/core50.log 2>&1 &


# Domainnet
nohup python main.py --config configs/eval/known/domainnet.json >> logs/eval/known/domainnet.log 2>&1 &

```

