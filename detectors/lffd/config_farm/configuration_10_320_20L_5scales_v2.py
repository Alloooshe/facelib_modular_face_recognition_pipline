# -*- coding: utf-8 -*-

import sys
import datetime
import os
import math
import logging



# add mxnet python path to path env if need
mxnet_python_path = '/home/heyonghao/libs/incubator-mxnet/python'
sys.path.append(mxnet_python_path)
import mxnet

'''
init logging
'''
param_log_mode = 'w'
param_log_file_path = '../log/%s_%s.log' % (os.path.basename(__file__)[:-3], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

'''
    data setting
'''
# pick file path for train set
param_trainset_pickle_file_path = os.path.join(os.path.dirname(__file__), '../data_provider_farm/data_folder/train_data_gt_9.pkl')
# pick file path for test set
param_valset_pickle_file_path = ''

'''
    training setting
'''

# batchsize for training
param_train_batch_size = 16

# the ratio of neg image in a batch
param_neg_image_ratio = 0.1

# GPU index for training (single machine multi GPU)
param_GPU_idx_list = [0]

# input height for network
param_net_input_height = 640

# input width for network
param_net_input_width = 640

# the number of train loops
param_num_train_loops = 2000000

# the number of threads used for train dataiter
param_num_thread_train_dataiter = 4

# the number of threads used for val dataiter
param_num_thread_val_dataiter = 1

# training start index
param_start_index = 0

# the evaluation frequency for current model
param_validation_interval = 10000

# batchsize for validation
param_val_batch_size = 20

# the number of loops for each evaluation
param_num_val_loops = 0

# the path of pre-trained model
param_pretrained_model_param_path = ''

# the frequency of display, namely displaying every param_display_interval loops
param_display_interval = 100

# the frequency of metric update, less updates will boost the training speed (should less than param_display_interval)
param_train_metric_update_frequency = 20

# set save prefix (auto)
param_save_prefix = '../saved_model/' + os.path.basename(__file__)[:-3] + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + \
                    '/' + os.path.basename(__file__)[:-3].replace('configuration', 'train')

# the frequency of model saving, namely saving the model params every param_model_save_interval loops
param_model_save_interval = 100000

# hard nagative mining ratio, needed by loss layer
param_hnm_ratio = 5

# init learning rate
param_learning_rate = 0.1
# weight decay
param_weight_decay = 0.00001
# momentum
param_momentum = 0.9

# learning rate scheduler -- MultiFactorScheduler
scheduler_step_list = [500000, 1000000, 1500000]
# multiply factor of scheduler
scheduler_factor = 0.1

# construct the learning rate scheduler
param_lr_scheduler = mxnet.lr_scheduler.MultiFactorScheduler(step=scheduler_step_list, factor=scheduler_factor)
# param_optimizer_name = 'adam'
# param_optimizer_params = {'learning_rate': param_learning_rate,
#                           'wd': param_weight_decay,
#                           'lr_scheduler': param_lr_scheduler,
#                           'begin_num_update': param_start_index}
param_optimizer_name = 'sgd'
param_optimizer_params = {'learning_rate': param_learning_rate,
                          'wd': param_weight_decay,
                          'lr_scheduler': param_lr_scheduler,
                          'momentum': param_momentum,
                          'begin_num_update': param_start_index}
'''
    data augmentation
'''

# trigger for horizon flip
param_enable_horizon_flip = True

# trigger for vertical flip
param_enable_vertical_flip = False

# trigger for brightness
param_enable_random_brightness = True
param_brightness_factors = {'min_factor': 0.5, 'max_factor': 1.5}

# trigger for saturation
param_enable_random_saturation = True
param_saturation_factors = {'min_factor': 0.5, 'max_factor': 1.5}

# trigger for contrast
param_enable_random_contrast = True
param_contrast_factors = {'min_factor': 0.5, 'max_factor': 1.5}

# trigger for blur
param_enable_blur = False
param_blur_factors = {'mode': 'random', 'sigma': 1}
param_blur_kernel_size_list = [3]

# negative image resize interval
param_neg_image_resize_factor_interval = [0.5, 3.5]

'''
    algorithm
'''
# the number of image channels
param_num_image_channel = 3

# the number of output scales (loss branches)
param_num_output_scales = 5

# feature map size for each scale
param_feature_map_size_list = [159, 79, 39, 19, 9]

# bbox lower bound for each scale
param_bbox_small_list = [10, 20, 40, 80, 160]
assert len(param_bbox_small_list) == param_num_output_scales

# bbox upper bound for each scale
param_bbox_large_list = [20, 40, 80, 160, 320]
assert len(param_bbox_large_list) == param_num_output_scales

# bbox gray lower bound for each scale
param_bbox_small_gray_list = [math.floor(v * 0.9) for v in param_bbox_small_list]
# bbox gray upper bound for each scale
param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in param_bbox_large_list]

# the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
param_receptive_field_list = param_bbox_large_list
# RF stride for each scale
param_receptive_field_stride = [4, 8, 16, 32, 64]
# the start location of the first RF of each scale
param_receptive_field_center_start = [3, 7, 15, 31, 63]

# the sum of the number of output channels, 2 channels for classification and 4 for bbox regression
param_num_output_channels = 6

