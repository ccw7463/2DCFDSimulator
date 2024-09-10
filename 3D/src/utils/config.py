import ml_collections
import numpy as np

def get_transformer_config():
    '''Transformer configurations'''
    config = ml_collections.ConfigDict()
    config.model_name = "CFDUnet"
    config.num_layers = 12
    config.hidden_size = 256 # 256, 512
    config.num_heads = 8
    config.mlp_dim = 1024 # 512, 1024
    config.rate = 0.2
    config.dimension_setting = [120,60,64]
    config.maximum_position_encoding = 840
    return config

def simulaion_config():
    '''simulation dimension confiurations'''
    config = ml_collections.ConfigDict()
    config.x_dim=120
    config.y_dim=60
    config.z_dim=64
    return config

def operating_condition_max_value_config(operating_data):
    '''운전조건 속성별 최대값'''
    
    config = ml_collections.ConfigDict()
    iv_max = np.max(operating_data[:,1,0])
    it_max = np.max(operating_data[:,1,1])
    wt_max = np.max(operating_data[:,1,2])
    angle_max = np.max(operating_data[:,1,3])

    config.iv_max=iv_max
    config.it_max=it_max
    config.wt_max=wt_max
    config.angle_max=angle_max

    return config


