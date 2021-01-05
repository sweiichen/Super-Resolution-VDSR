import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time, math
import warnings
import os
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpus",
                    default="1",
                    type=str,
                    help="gpu ids (default: 0)")
parser.add_argument("--gt", action="store_true", help="Use ground truth?")
parser.add_argument('--model',
                    default='model/model_epoch_8.pth',
                    type=str,
                    help='path to model weight')
parser.add_argument('--input',
                    default='testing_lr_images/',
                    type=str,
                    help='path to input images folder')
parser.add_argument('--output',
                    default='result/',
                    type=str,
                    help='path to output images folder')
parser.add_argument("--factor", type=int, default=3, help="upscale factor")
parser.add_argument('--gt_folder',
                    default='val/',
                    type=str,
                    help='path to groud truth images folder')

opt = parser.parse_args()


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border,
                shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border,
            shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:, :, 0] = y
    img[:, :, 1] = ycbcr[:, :, 1]
    img[:, :, 2] = ycbcr[:, :, 2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


if opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

model = torch.load(opt.model, map_location="cpu")["model"]
if opt.cuda:
    model = model.cuda()
else:
    model = model.cpu()

save_to = opt.output
test_dir = opt.input
if opt.gt:
    gt = opt.gt_folder
avg_bi = 0
avg_vdsr = 0
os.makedirs(save_to, exist_ok=True)
for file in os.listdir(test_dir):
    im_b = Image.open(os.path.join(test_dir, file)).convert("RGB")
    im_b = im_b.resize((im_b.width * opt.factor, im_b.height * opt.factor),
                       Image.BICUBIC)
    im_b_ycbcr = np.array(im_b.convert("YCbCr"))
    im_b_y = im_b_ycbcr[:, :, 0].astype(float)
    im_input = im_b_y / 255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(
        1, -1, im_input.shape[0], im_input.shape[1])
    if opt.cuda:
        im_input = im_input.cuda()
    out = model(im_input)
    out = out.cpu()
    im_h_y = out.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0, :, :]
    if opt.gt:
        print(f"Evaluation of {file}:")
        im_gt = Image.open(os.path.join(gt, file)).convert("RGB")
        im_gt_ycbcr = np.array(im_gt.convert("YCbCr"))
        im_gt_y = im_gt_ycbcr[:, :, 0].astype(float)
        psnr_bicubic = PSNR(im_gt_y, im_b_y)
        print('psnr for bicubic is {}dB'.format(psnr_bicubic))
        avg_bi += psnr_bicubic
        psnr_predicted = PSNR(im_gt_y, im_h_y)
        print('psnr for vdsr is {}dB'.format(psnr_predicted))
        avg_vdsr += psnr_predicted

    im_h = colorize(im_h_y, im_b_ycbcr)
    im_h.save(os.path.join(save_to, file))
if opt.gt:
    avg_vdsr = avg_vdsr / len(os.listdir(test_dir))
    avg_bi = avg_bi / len(os.listdir(test_dir))
    print(f"Average PSNR of VDSR:{avg_vdsr}")
    print(f"Average PSNR of Bicubic:{avg_bi}")