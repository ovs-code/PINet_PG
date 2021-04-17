# PINet_PG
Code for our PG paper [Human Pose Transfer by Adaptive Hierarchical Deformation](http://cic.tju.edu.cn/faculty/likun/pg2020.pdf)

This is Pytorch implementation for pose transfer on DeepFashion dataset. The code is extremely borrowed from [Pose Transfer](https://github.com/tengteng95/Pose-Transfer). Thanks for their work!

# Requirement
```
conda create -n tip python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate 
```



# Data

Data preparation for images and keypoints can follow [Pose Transfer](https://github.com/tengteng95/Pose-Transfer)
Parsing data can be found from [baidu](https://pan.baidu.com/s/1Ic8sIY-eYhGnIoZdDlhgxA) (fetch code:abcd) or [Google drive](https://drive.google.com/file/d/1xwm5cOrj2LSAp8H1wA4_YqK32pCtnXir/view?usp=sharing)

# Test
You can directly download our test results from [baidu](https://pan.baidu.com/s/15tcKgRV12NGByIrr4qoqDw) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1r8ebcw3IW7-3AKGkJtcW03ckMVWeAYGZ/view?usp=sharing).<br>
Pre-trained checkpoint can be found from [baidu](https://pan.baidu.com/s/1Orvpt42lV-R2uzI-10q3_A) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1xwm5cOrj2LSAp8H1wA4_YqK32pCtnXir/view?usp=sharing) and put it in the folder (-->checkpoints-->fashion_PInet_PG).

**Test by yourself** <br>

```python
python test.py --dataroot ./fashion_data/ --name fashion_PInet_PG --model PInet --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PInet --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results
```

# Run the webapp
First one has to create an environment with all depencies.
We used `venv` [How to install venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv) but `conda` or `pipenv` should work too. However the scripts and examples are based on `venv`.
```bash
## switch to project folder
cd PINet_PG/
## create new virtual environment
python3 -m venv [venvname]
## activate the virtual environment
source [venvname]/bin/activate
## install the dependencies from requirements.txt
pip install -r requirements.txt
## if there are problems with the requirements try and install missing depencies manually
pip install -r requirements.txt --no-deps
```

The next step is to install the node modules (Javascript dependencies). We are using `npm` for this which is shipped with `node.js` [Install node.js](https://nodejs.org/en/).
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



# Citation

If you use this code, please cite our paper.

```
@article {10.1111:cgf.14148,
journal = {Computer Graphics Forum},
title = {{Human Pose Transfer by Adaptive Hierarchical Deformation}},
author = {Zhang, Jinsong and Liu, Xingzi and Li, Kun},
year = {2020},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14148}
}
```



# Acknowledgments

Our code is based on [Pose Transfer](https://github.com/tengteng95/Pose-Transfer).
