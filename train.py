from collections import OrderedDict, defaultdict
import time
import copy

import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.PINet20 import create_model
from util.visualizer import Visualizer

import torch
torch.backends.cudnn.benchmark = True


opt = TrainOptions().parse()
val_opt = copy.copy(opt)
val_opt.serial_batches = True
val_opt.phase = 'val'
val_opt.resize_or_crop = 'no'
val_opt.no_flip = True
val_opt.batchSize = 16
val_data_loader = CreateDataLoader(val_opt)
data_loader = CreateDataLoader(opt)
val_dataset = val_data_loader.load_data()
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
print('#validation images = %d' % len(val_dataset))

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0


for epoch in range(opt.epoch_count, opt.sepiter + opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    model.isTrain = True
    model.train()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)

        if epoch <= opt.sepiter:
            model.optimize_parameters_seperate()
        else:
            model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps %  opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % 10 == 0 and epoch>500:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.sepiter + opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.sepiter:
        model.update_learning_rate()

    # TODO: validate
    val_errors = OrderedDict()
    model.isTrain = False
    model.eval()
    for i, data in enumerate(val_dataset):
        model.set_input(data)
        with torch.no_grad():
            model.validate()

        errors = model.get_current_errors()
        for k in errors:
            try:
                val_errors[k] = val_errors.get(k, []) + [errors[k].item()]
            except AttributeError:
                val_errors[k] = val_errors.get(k, []) + [errors[k]]
    visualizer.plot_validation_errors(epoch, {k: sum(v)/len(v) for k, v in val_errors.items()})
    # TODO: print/log validation losses
