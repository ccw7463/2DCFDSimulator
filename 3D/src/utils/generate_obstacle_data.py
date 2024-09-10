import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import ml_collections

def get_obstacle_config(data):
    '''테스트 위한 obstacle data 만들때 사용'''
    config = ml_collections.ConfigDict()

    # 시뮬레이션 고정값
    config.x_dim= data["x_dim"]
    config.y_dim= data["y_dim"]
    config.z_dim= data["z_dim"]
    config.x_max = data["x_max"]
    config.y_max = data["y_max"]
    config.z_max = data["z_max"]

    # 장애물 개수 설정
    config.obs_count = data["obs_count"]

    # 장애물 위치바꾸는 부분
    config.obs_x_min = data["obs_x_min"]
    config.obs_x_max = data["obs_x_max"]
    config.obs_y_min = data["obs_y_min"]
    config.obs_y_max = data["obs_y_max"]
    config.obs_z_min = data["obs_z_min"]
    config.obs_z_max = data["obs_z_max"]

    # 입구/출구 좌표값
    config.input_pos = data["input_pos"]
    config.output_pos = data["output_pos"]

    return config

def make_null_dataframe(config):
    df_x_lst = []
    df_y_lst = []
    df_z_lst = []
    for a in np.linspace(0,config.y_max,config.y_dim):
        for b in np.linspace(0,config.x_max,config.x_dim):
            for c in np.linspace(0,config.z_max,config.z_dim):
                df_x_lst.append(b)
                df_y_lst.append(a)
                df_z_lst.append(c)
    df=pd.DataFrame({"x":df_x_lst,"y":df_y_lst,"z":df_z_lst})
    return df

def generate_obstacle_data(config):

    arr_lst = []

    # 장애물 개수만큼 반복문
    obs_count = config.obs_count

    for i in range(obs_count):
        # 좌표값 (m)를 배열번호로 바꿈

        arr_x_min = round( config.obs_x_min[i] * config.x_dim / config.x_max ) 
        arr_x_max = round( config.obs_x_max[i] * config.x_dim / config.x_max )
        arr_y_min = round( config.obs_y_min[i] * config.y_dim / config.y_max )
        arr_y_max = round( config.obs_y_max[i] * config.y_dim / config.y_max )
        arr_z_min = round( config.obs_z_min[i] * config.z_dim / config.z_max )
        arr_z_max = round( config.obs_z_max[i] * config.z_dim / config.z_max )
        x_range = np.arange(arr_x_min,arr_x_max,1)
        y_range = np.arange(arr_y_min,arr_y_max,1)

        # 배열생성과정
        boundaries =[]
        for _ in range(arr_z_max):
            boundary = np.ones([config.x_dim,config.y_dim])
            for i in x_range:
                for j in y_range:
                    boundary[i,j] = 0
            boundaries.append(boundary)
        for _ in range(config.z_dim-arr_z_max):
            boundary = np.ones([config.x_dim,config.y_dim])
            boundaries.append(boundary)
        arr = np.array(boundaries).transpose(2,1,0)
        arr = arr.reshape(config.x_dim*config.y_dim*config.z_dim)
        arr_lst.append(arr)

    return arr_lst

