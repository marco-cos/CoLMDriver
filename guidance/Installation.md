Two environments are needed: 'vllm' and 'colmdriver', we recommend creating two separate environments to avoid conflicts.

## vllm environment
Prepare an environment for running MLLMs.
Please refer to https://github.com/vllm-project/vllm.
Name this environment as vllm.
```Shell
conda activate vllm
```

## For colmdriver
### Step 1: Basic Installation for colmdriver
Get code and create pytorch environment.
```Shell
git clone https://github.com/cxliu0314/CoLMDriver.git
conda create --name colmdriver python=3.7 cmake=3.22.1
conda activate colmdriver
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

pip install -r opencood/requirements.txt
pip install -r simulation/requirements.txt
```

### Step 2: Download and setup CARLA 0.9.10.1.
```Shell
chmod +x simulation/setup_carla.sh
./simulation/setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
mkdir external_paths
ln -s ${PWD}/carla/ external_paths/carla_root
# If you already have a Carla, just create a soft link to external_paths/carla_root
```

The file structure should be:
```Shell
|--CoLMDriver
    |--external_paths
        |--carla_root
            |--CarlaUE4
            |--Co-Simulation
            |--Engine
            |--HDMaps
            |--Import
            |--PythonAPI
            |--Tools
            |--CarlaUE4.sh
            ...
```

Note: we choose the setuptools==41 to install because this version has the feature `easy_install`. After installing the carla.egg you can install the lastest setuptools to avoid No module named distutils_hack.

Steps 3,4,5 are for perception module.

### Step 3: Install Spconv (1.2.1)
We use spconv 1.2.1 to generate voxel features in perception module.

To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

### Step 4: Set up
```Shell
# Set up
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
```

### Step 5: Install pypcd
```Shell
# go to another folder
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```
