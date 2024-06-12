import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from tqdm import tqdm

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from math import log10

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def metrics(img1, img2):
    img1_np = img1.cpu().detach().numpy()
    img2_np = img2.cpu().detach().numpy()

    # Calculate PSNR using PyTorch
    mse = F.mse_loss(img1, img2).item()
    psnr = 10 * log10(1 / mse)

    # Calculate SSIM using scikit-image
    ssim_val = ssim(img1_np, img2_np, multichannel=True)
    return psnr, ssim_val

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    if opt.max_dataset_size == -1:
        opt.max_dataset_size = float("inf")
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_deb_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()
    for i, data in enumerate(tqdm(dataset)):
        # breakpoint()
        if (i < opt.delta): continue
        if (opt.num_test != -1) and (i >= opt.num_test + opt.delta):  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        sizeA = model.get_sizeA()

        # breakpoint()
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, sizeA, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
