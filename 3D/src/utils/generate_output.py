import numpy as np
import os
from tqdm import tqdm
from tqdm.contrib import tzip
from scipy.interpolate import griddata
import pandas as pd

def load_data():
    '''데이터 로드'''
    no_obstacle_data=[i for i in os.listdir("./dataset/raw/no_obstacle/") if i!=".ipynb_checkpoints"]
    yes_obstacle_data=[i for i in os.listdir("./dataset/raw/yes_obstacle/") if i!=".ipynb_checkpoints"]
    return no_obstacle_data,yes_obstacle_data

def generate_no_obstacle_output(no_obstacle_data,dimension_setting,option):
    '''장애물 없는 데이터 output 생성'''
    output_data = []
    for i in tqdm(no_obstacle_data):
        out = process_no_obstacle(i,dimension_setting,option)
        output_data.append(out)
    output_data = np.array(output_data)

    if option == "temperature":
        output_data = calculate_celsius(output_data,dimension_setting)
        
    np.save(f"./dataset/processed/output/no_obstacle/{option}_label.npy",output_data)
    
def generate_yes_obstacle_output(yes_obstacle_data,dimension_setting,option):
    '''장애물 있는 데이터 output 생성'''
    output_data=[]
    obstacle_value_lst=np.load(open("./src/data/obstacle_value_lst.npy",'rb'))
    for i,j in tzip(yes_obstacle_data,obstacle_value_lst):
        j = j.reshape(dimension_setting[0],dimension_setting[1],dimension_setting[2])
        out = process_yes_obstacle(i,dimension_setting,option,j)
        output_data.append(out)
    output_data = np.array(output_data) # 108,120,60,64

    if option == "temperature":
        output_data = calculate_celsius(output_data,dimension_setting)

    np.save(f"./dataset/processed/output/yes_obstacle/{option}_label.npy",output_data)
    
def process_no_obstacle(i,dimension_setting,option):
    '''장애물 없는 데이터 전처리'''
    with open("dataset/raw/no_obstacle/"+i,"r") as txt_file:
            df = text_to_df(txt_file)
    aligned_df = manipulate_dataframe(df,dimension_setting,option)
    value = aligned_df[option].values.reshape(dimension_setting[0],dimension_setting[1],dimension_setting[2])    
    return value

def process_yes_obstacle(i,dimension_setting,option,j):
    '''장애물 있는 데이터 전처리'''
    with open("dataset/raw/yes_obstacle/"+i,"r") as txt_file:
            df = text_to_df(txt_file)
    aligned_df = manipulate_dataframe(df,dimension_setting,option)
    value = aligned_df[option].values.reshape(dimension_setting[0],dimension_setting[1],dimension_setting[2])    
    value = value*j
    return value

def text_to_df(text):
    '''입력된 텍스트를 파이썬형태에 맞게 dataframe으로 변경'''
    count = 0
    x_lst = []
    y_lst = []
    z_lst = []
    temp = []
    vel_mag =[]
    z_velocity = []
    x_velocity = []
    y_velocity = []
    
    # 텍스트파일 한 줄 씩 읽어와서 데이터구성
    for i in text:
        if count == 0:
            count = count + 1
            pass
        else:
            lst = i.split()
            x_lst.append(float(lst[1]))
            y_lst.append(-1*float(lst[3])) # 파이썬의 X/Y/Z 축과 LG에서 사용한 X/Y/Z 축 방향이 다르기때문에 수정
            z_lst.append(float(lst[2])) # 파이썬의 X/Y/Z 축과 LG에서 사용한 X/Y/Z 축 방향이 다르기때문에 수정
            temp.append(float(lst[4]))
            vel_mag.append(float(lst[5]))
            x_velocity.append(float(lst[8]))
            y_velocity.append(-1*float(lst[6])) # 파이썬의 X/Y/Z 축과 LG에서 사용한 X/Y/Z 축 방향이 다르기때문에 수정
            z_velocity.append(float(lst[7])) # 파이썬의 X/Y/Z 축과 LG에서 사용한 X/Y/Z 축 방향이 다르기때문에 수정
    df = pd.DataFrame({"x":x_lst,"y":y_lst,"z":z_lst,"temperature":temp,"velocity":vel_mag,"ux":x_velocity,"uy":y_velocity,"uz":z_velocity}) 
    
    return df

def manipulate_dataframe(df,dimension_setting,option):
    '''입력된 데이터프레임을 정렬'''
    # griddata 매개변수 3가지 지정
    # 1. CFDpoint = 좌표값
    CFDpoint = np.zeros((len(df),3))    
    CFDpoint[:,0] = df["x"]
    CFDpoint[:,1] = df["y"]
    CFDpoint[:,2] = df["z"]

    # 2. values
    values = df[option]
    
    # 3. point 지정
    nx = dimension_setting[0]
    ny = dimension_setting[1]
    nz = dimension_setting[2]
    min_x = np.round(np.min(df["x"]),1)
    max_x = np.round(np.max(df["x"]),1)
    min_y = np.round(np.min(df["y"]),1)
    max_y = np.round(np.max(df["y"]),1)
    min_z = np.round(np.min(df["z"]),1)
    max_z = np.round(np.max(df["z"]),1)
    i = np.linspace(min_x,max_x,nx) 
    j = np.linspace(min_y,max_y,ny) 
    k = np.linspace(min_z,max_z,nz)
    point = [(x,y,z) for y in j for x in i for z in k ] # z => x => y 순
    point = np.array(point)

    # 4. 가공 및 리턴
    values = df[option]
    StructuredValue = griddata(points=CFDpoint,values=values,xi=point,method='nearest') 
    StructuredValue = StructuredValue.reshape(-1,1) 
    alined_df = np.hstack((point,StructuredValue))
    alined_df = pd.DataFrame(alined_df,columns=["x","y","z",option]) 

    return alined_df

def calculate_celsius(output_data,dimension_setting):
    # output_data = 108,120,60,64
    new_output_data_all = []
    for i in tqdm(output_data):
        i = i.reshape(120*60*64)
        new_output_data = []
        for j in i:
            if j == 0 :
                new_output_data.append(j)
            else:
                new_output_data.append(j-273.15)
        new_output_data_all.append(new_output_data)
    new_output_data_all = np.array(new_output_data_all)
    new_output_data_all = new_output_data_all.reshape(-1,dimension_setting[0],dimension_setting[1],dimension_setting[2])
    return new_output_data_all