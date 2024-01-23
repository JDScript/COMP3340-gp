# COMP3340 Group Project - Image Classifier on Flowers17
This is the repository for the group project of COMP3340 Applied Machine Learning @ HKU.
## Environment Setup
```
conda create -n comp3340gp python=3.11
conda activate comp3340gp
pip install -r requirements.txt
```

## Configuration
Please kindly check the configuration files in `config.yaml`. You can change the backbone and other parameters there. Please note that command line parameters will have higher priority than the configuration file. If you want to train on cuda, you can either change the device parameter in `config.yaml` or simply use the following command to start training.
```
python main.py device=cuda
```