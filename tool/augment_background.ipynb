{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "billion-bible",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import os\n",
    "from PIL import Image, ImageOps\n",
    "import sys\n",
    "\n",
    "\n",
    "def background_swap(image, segmentation, background):\n",
    "    assert image.shape == background.shape, print(\"image and background dimensions don't match: \",'\\n',\"image: \",image.shape,\"\\n background: \", background.shape)\n",
    "    i = 0\n",
    "    newnew = np.zeros(image.shape)\n",
    "    for col in range(segmentation.shape[0]):\n",
    "        for row in range(segmentation.shape[1]):\n",
    "            if segmentation[col,row] == 0:\n",
    "                newnew[col,row,0] = background[col,row,0]\n",
    "                newnew[col,row,1] = background[col,row,1]\n",
    "                newnew[col,row,2] = background[col,row,2]\n",
    "            else:\n",
    "                newnew[col,row,0] = image[col,row,0]\n",
    "                newnew[col,row,1] = image[col,row,1]\n",
    "                newnew[col,row,2] = image[col,row,2]\n",
    "\n",
    "    newnew = Image.fromarray(np.uint8(newnew)).convert('RGB')\n",
    "    return newnew\n",
    "\n",
    "\n",
    "\n",
    "# referencing list of test images from deepfashion dataset: relative file references will have to change\n",
    "inputs_list = []\n",
    "lst = open('../fashion/test_data/test.lst', 'r')\n",
    "inputs_list = lst.read().splitlines()\n",
    "inpath = '../fashion/test_data/test'\n",
    "\n",
    "# this is the input image from the deepfashion dataset\n",
    "pic = np.asarray(Image.open(os.path.join(inpath,inputs_list[1900])).resize((176,256))) \n",
    "\n",
    "# res is the output of the segmentation model (function is called segment in the parsing_compiled_final.ipynb file)\n",
    "new_res = res.reshape(256,176,1) \n",
    "\n",
    "_, ax = plt.subplots(3, 5, figsize=(15,15))\n",
    "i=0\n",
    "j=0\n",
    "\n",
    "for filename in os.listdir('../backgrounds'):\n",
    "    if filename.endswith(\".jpg\") or filename.endswith(\".py\"): \n",
    "        background = np.asarray(Image.open(os.path.join(os.path.abspath('../backgrounds'),filename)).resize((176,256))) # this is from the backgrounds folder\n",
    "        ax[j,i].imshow(background_swap(pic, new_res, background))\n",
    "        if i == 4:\n",
    "            i = 0\n",
    "            j +=1\n",
    "        else:\n",
    "            i +=1\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
