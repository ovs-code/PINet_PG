import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.PINet20 import create_model
from util.visualizer import Visualizer
from util import html
import time

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# TODO: add DataLoader for using the model without ground truth
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

# opt.how_many = 999999
fps = 0
num = 0
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many))
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    fps += endTime-startTime
    num += 1
    print(endTime-startTime)
    visuals = model.get_reduced_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]
    print(img_path)
    visualizer.save_images(webpage, visuals, img_path)
print(fps)
print(num)
#print(fps/num)
webpage.save()
