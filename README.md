# Web Application for Pose Transfer
Code for an university project based on [PInet_PG](https://github.com/Zhangjinso/PINet_PG)

## Requirements

First one has to create an environment with all depencies.
We used `venv` [(How to install venv)](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv) but `conda` or `pipenv` should work too. However the scripts and examples are based on `venv` and that the environment is named *venv*.

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
## if there are problems with the requirements try and install missing depencies manually
pip install -r requirements.txt --no-deps
```

Install Pytorch according to the instructions on [pytorch.org](https://pytorch.org/).

## Models

For generating the training data and for running inference two additional pretrained models are required:

 - [Grapy-ML](https://github.com/Charleshhy/Grapy-ML) for human parsing can be found on their [Google Drive](https://drive.google.com/drive/folders/1eQ9IV4QmcM5dLCuVMSVE3ogVpL6qUQL5)
 - A pose estimator [based on HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation) can be found [here](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA).

Download both `CIHP_trained.pth` and `pose_hrnet_w48_256x192.pth` into `assets/pretrains/`.

## Training Data

Download the images from the *In-shop Clothes Retrieval Benchmark* from the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) and split them according to the list provided in `fashion_data` into the subfolders `fashion_data/train/`, `fashion_data/test/` and `fashion_data/val/`.
Generate additional training data by running `scripts/generate_train_data.sh`.

See `scripts/train_star.sh` for examples of configuring a training run.

# Run the webapp

After installing the requirements and auxiliary models as seen above, the next step is to install the node modules (Javascript dependencies). We are using `npm` for this which is shipped with `node.js` [(Install node.js)](https://nodejs.org/en/).
After installing `node` and `npm` one can install the node modules:
```bash
## switch to the webapp/static folder
cd webapp/static
## install the dependencies from the package-lock.json
npm install
```

Additional we need to install redis a database for the task queue:
```bash
sudo apt install redis
```

To run the app we need to download some additional files (i.e. the models and video files):
(https://drive.google.com/drive/folders/1oOC3G_CMR8hQPDY9ob65meFww77LIdqZ)
```bash
## change to top-level directory
cd PINet_PG
## create the folders for the transfer model
mkdir -p checkpoints/fashion_PInet_cycle
## download `latest_net_netG.pth` to `PINet_PG/checkpoints/fashion_PInet_cycle
## download and extract `assets.tar.gz` at top-level
cd PINet_PG
tar -xf assets.tar.gz
## download and extract `videos.tar.gz` in the PINet_PG/webapp/static folder
cd PINet_PG/webapp/static
tar -xf videos.tar.gz
```

The last step is to run the `start_webapp.sh` script:
```bash
## switch to the top level folder
cd PINet_PG
## run the script
bash start_webapp.sh

## to stop the webapp
ctrl+C
```

# Acknowledgments

Our model and code is heavily based on [PINet_PG](https://github.com/Zhangjinso/PINet_PG), their code is based on [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer).  
We use some additional code from the [Grapy-ML](https://github.com/Charleshhy/Grapy-ML) for running the human parsing model.
