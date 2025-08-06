import json
import os
import torch
import importlib

MODEL_DIR = 'new_LAM/content/LAM_Demo/ModelZoo/models'

NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN',
    'SAN',
    'SwinIR',
    'HAT',
    'MAT'
]

MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'SwinIR': {
        # 'Base': '004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth'
        'Base': '001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
        #   'Base': '004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth'
        #   'Base': '006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth'
    },
    'HAT': {
        'Base': 'HAT_SRx4.pth'
    },
    'MAT': {
        'Base': 'MAT_x4.pth'
    }
}


def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(module_name, class_name, pretrained_model_name, package_path=".networks"):
    # Construct the full module path
    module_path = f"{package_path}.{module_name}"
    # Dynamically import the module
    module = importlib.import_module(module_path, package=__package__)
    # Get the class from the module
    model_class = getattr(module, class_name)
    # loading parameters in json file with the same name as class name
    with open(f'new_LAM/content/LAM_Demo/ModelZoo/networks/{class_name}.json', 'r') as file:
        params = json.load(file)

    net = model_class(**params)
    # net = model_class(pretrained=f'content/LAM_Demo/ModelZoo/models/{pretrained_model_name}')

    # device = torch.device('cpu')
    #
    # pretrained_dict = torch.load(f'content/LAM_Demo/ModelZoo/models/{pretrained_model_name}', map_location='cpu')
    # net.load_state_dict(pretrained_dict, strict=True)
    # net = net.to(device)
    #
    # print_network(net, class_name)
    return net


def load_model(module_name, class_name, pretrained_model_name):
    net = get_model(module_name, class_name, pretrained_model_name)
    state_dict_path = os.path.join(MODEL_DIR, pretrained_model_name)
    print(f'Loading model {state_dict_path} for {class_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    net.load_state_dict(state_dict)
    return net

# def get_model(model_name, factor=4, num_channels=3):
#     """
#     All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
#     :param model_name:
#     :param factor:
#     :param num_channels:
#     :return:
#     """
#     print(f'Getting SR Network {model_name}')
#     if model_name.split('-')[0] in NN_LIST:
#
#         if model_name == 'RCAN':
#             from .NN.rcan import RCAN
#             net = RCAN(factor=factor, num_channels=num_channels)
#
#         elif model_name == 'CARN':
#             from .CARN.carn import CARNet
#             net = CARNet(factor=factor, num_channels=num_channels)
#
#         elif model_name == 'RRDBNet':
#             from .NN.rrdbnet import RRDBNet
#             net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)
#
#         elif model_name == 'SAN':
#             from .NN.san import SAN
#             net = SAN(factor=factor, num_channels=num_channels)
#
#         elif model_name == 'RNAN':
#             from .NN.rnan import RNAN
#             net = RNAN(factor=factor, num_channels=num_channels)
#
#         elif model_name == 'SwinIR':
#             from .NN.network_swinir import SwinIR
#             net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
#                          depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
#
#         elif model_name=='HAT':
#             from .NN.HAT_arch import HAT
#             net = HAT(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
#                       num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
#
#         elif model_name=='MAT':
#             from .NN.MAT_arch import MAT
#             net = MAT(upsampler="pixelshuffle")
#
#         else:
#             raise NotImplementedError()
#
#         print_network(net, model_name)
#         return net
#     else:
#         raise NotImplementedError()


# def load_model(model_loading_name):
#     """
#     :param model_loading_name: model_name-training_name
#     :return:
#     """
#     splitting = model_loading_name.split('@')
#     if len(splitting) == 1:
#         model_name = splitting[0]
#         training_name = 'Base'
#     elif len(splitting) == 2:
#         model_name = splitting[0]
#         training_name = splitting[1]
#     else:
#         raise NotImplementedError()
#     assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
#     net = get_model(model_name)
#
#     state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
#     print(f'Loading model {state_dict_path} for {model_name} network.')
#     state_dict = torch.load(state_dict_path, map_location='cpu')
#     net.load_state_dict(state_dict)
#     return net
