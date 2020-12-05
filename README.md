## Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Numpy
- Pytorch 0.4+

Preparation 1: create folder for dataset.

first, download Market-1501 dataset from the links below

google drive: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing

baidu disk: https://pan.baidu.com/s/1ntIi2Op

second,
```bash
mkdir data
unzip Market-1501-v15.09.15.zip
ln -s Market-1501-v15.09.15 market
``` 
then, get the directory structure
```
├── feature_disentangle
	　　　　　　   ├── data
	　　　　　　　        ├── market
	　　　　　　　        ├── Market-1501-v15.09.15
```


Preparation 2: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```

Finally, conduct training, testing and evaluating with one command
```bash
python run.py
```

This code is related to our paper _A Feature Disentangling Approach for Person Re-Identification via Self-Supervised Data Augmentation_.

If you use this code, please cite our paper as:

```

```
