import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from PIL import Image

sys.path.append('content\LAM_Demo')

from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath


from ModelZoo.networks.hat_model import HATModel
import ModelZoo.NN.HAT_arch
import yaml

opt = None
with open('HAT_SRx4.yml') as f:
  opt = yaml.safe_load(f)
  opt['is_train'] = False
  opt['dist'] = False
  # opt['network_g']['img_size'] = 256
  # opt['network_g']['img_range'] = 255.
  print(opt)
model = HATModel(opt)


# module_name = input('enter module name:\n(It must be placed in: content/LAM_Demo/ModelZoo/networks) (e.g.: network_swinir)\n')
# class_name = input('enter class name (json file must have the same name) (e.g.: SwinIR):\n')
# pretrained_model_name = input('enter pretrained model name:\n(It must be placed in: content/LAM_Demo/ModelZoo/models) (e.g.: 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth)\n')

# model = load_model(module_name, class_name, pretrained_model_name)
# model = load_model('SwinIR@Base')
model.eval()

window_size = 8  # Define windoes_size of D
img_lr, img_hr = prepare_images('content/LAM_Demo/test_images/7.png')  # Change this image name
tensor_lr = PIL2Tensor(img_lr)[:3]
tensor_hr = PIL2Tensor(img_hr)[:3]
cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2)
cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

# ------------------------
img_tensor = tensor_lr.unsqueeze(0).to(torch.device('cpu'))
out = img_tensor
with torch.no_grad():
    out = model(img_tensor)
Tensor2PIL(out).show()
# pil_image = Tensor2PIL(out)
# plt.imshow(pil_image)
# plt.title("output")
# plt.axis('off')
# plt.show()
# ------------------------



# plt.imshow(cv2_hr)
# plt.show()
# plt.imshow(cv2_lr)
# plt.show()

w = 110  # The x coordinate of your select patch, 125 as an example
h = 150  # The y coordinate of your select patch, 160 as an example
         # And check the red box
         # Is your selected patch this one? If not, adjust the `w` and `h`.


draw_img = pil_to_cv2(img_hr)
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img)
# position_pil
position_pil.show()

sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)


interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=False)

print(result_numpy)

grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)

abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)

print('--------------------')
print(abs_normed_grad_numpy)

saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)

pil = make_pil_grid(
    [position_pil,
     saliency_image_abs,
     blend_abs_and_input,
     blend_kde_and_input,
     Tensor2PIL(torch.clamp(result, min=0., max=1.))]
)

# pil
pil.show()
# plt.show(pil)


gini_index = gini(abs_normed_grad_numpy)
print(f"The gini index of this case is {gini_index}")
diffusion_index = (1 - gini_index) * 100
print(f"The diffusion index of this case is {diffusion_index}")