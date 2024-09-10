import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from collections import Counter

cnn_input_data_path_no = "dataset/processed/input/cnn_input/no_obstacle/"
text_input_data_path_no = "dataset/processed/input/text_input/no_obstacle/"
output_data_path_no = "dataset/processed/output/no_obstacle/"
dataframe_path_no = "dataset/dataframe/no_obstacle/"

cnn_input_data_path_yes = "dataset/processed/input/cnn_input/yes_obstacle/"
text_input_data_path_yes = "dataset/processed/input/text_input/yes_obstacle/"
output_data_path_yes = "dataset/processed/output/yes_obstacle/"
dataframe_path_yes = "dataset/dataframe/yes_obstacle/"

no_obstacle_path = "./dataset/raw/no_obstacle/" 
yes_obstacle_path = "./dataset/raw/yes_obstacle/"

inlist = [0.3798,0.4548,0,0.466,2.7,2.7] # 현재는 입구/출구 위치동일하기에 단순히 리스트로 정의하여사용
outlist = [0.044,0.1898,0,0.466,2.7,2.7] # 현재는 입구/출구 위치동일하기에 단순히 리스트로 정의하여사용

def create_save_directory():
    '''데이터 저장 경로 생성 (없을 경우에)'''
    if not os.path.isdir(cnn_input_data_path_no):
        os.makedirs(cnn_input_data_path_no) 
    if not os.path.isdir(text_input_data_path_no):
        os.makedirs(text_input_data_path_no)    
    if not os.path.isdir(output_data_path_no):
        os.makedirs(output_data_path_no)
    if not os.path.isdir(dataframe_path_no):
        os.makedirs(dataframe_path_no)        
    if not os.path.isdir(cnn_input_data_path_yes):
        os.makedirs(cnn_input_data_path_yes) 
    if not os.path.isdir(text_input_data_path_yes):
        os.makedirs(text_input_data_path_yes)    
    if not os.path.isdir(output_data_path_yes):
        os.makedirs(output_data_path_yes)
    if not os.path.isdir(dataframe_path_yes):
        os.makedirs(dataframe_path_yes)

def load_data():
    '''데이터 로드'''
    no_obstacle_data=[i for i in os.listdir("./dataset/raw/no_obstacle/") if i!=".ipynb_checkpoints"]
    yes_obstacle_data=[i for i in os.listdir("./dataset/raw/yes_obstacle/") if i!=".ipynb_checkpoints"]
    return no_obstacle_data,yes_obstacle_data

def generate_no_obstacle_boundary_data(no_obstacle_data,dimension_setting,option):
    '''장애물 없는 경우 boundary codition 생성'''
    boundary = []
    for num,i in enumerate(tqdm(no_obstacle_data)):
        bdr = make_boundary_no_obstacle(num,i,dimension_setting,option)
        boundary.append(bdr)
    boundary = np.array(boundary)
    np.save("./dataset/processed/input/cnn_input/no_obstacle/boundary_condition.npy",boundary)

def make_boundary_no_obstacle(num,i,dimension_setting,option):
    '''장애물 없는 경우 boundary condition 생성'''
    x_dim,y_dim,z_dim = dimension_setting

    with open(no_obstacle_path+i,'r') as f:
        text = f.readlines()
        df = text_to_df(text)
        df = manipulate_dataframe(num,df,[x_dim,y_dim,z_dim],option)
        df = make_bdr(df,inlist,outlist)
    return df["Boundary"]

def generate_yes_obstacle_boundary_data(yes_obstacle_data,dimension_setting):
    '''장애물 있는 경우 boundary codnition 생성'''
    boundary = []
    obstacle_value_lst = []
    for num,i in enumerate(tqdm(yes_obstacle_data)):
        bdr,obs_val = make_boundary_yes_obstacle(num,i,dimension_setting)
        boundary.append(bdr)
        obstacle_value_lst.append(obs_val)
    boundary = np.array(boundary)
    np.save("./src/data/obstacle_value_lst.npy",obstacle_value_lst)
    np.save("./dataset/processed/input/cnn_input/yes_obstacle/boundary_condition.npy",boundary)    

def make_boundary_yes_obstacle(num,i,dimension_setting):
    '''장애물있는 경우 boundary condition 생성'''
    x_dim,y_dim,z_dim = dimension_setting
    
    with open(yes_obstacle_path+i) as f:
        text = f.readlines()
        df = text_to_df(text)

    # 축별 최대길이
    x_max = np.round(np.max(df["x"]),1)
    y_max = np.round(np.max(df["y"]),1)
    z_max = np.round(np.max(df["z"]),1)

    # 좌표값만 주어진 빈 데이터 프레임 생성 (좌표순서:Z증가/X증가/Y증가)
    df_x_lst = []
    df_y_lst = []
    df_z_lst = []
    for a in np.linspace(0,y_max,y_dim):
        for b in np.linspace(0,x_max,x_dim):
            for c in np.linspace(0,z_max,z_dim):
                df_x_lst.append(b)
                df_y_lst.append(a)
                df_z_lst.append(c)
    new_df=pd.DataFrame({"x":df_x_lst,"y":df_y_lst,"z":df_z_lst})

    # boundary stack 수행 (z 범위를 사용)
    z_range_large = np.linspace(0,z_max,(z_dim+1))[1:]
    z_range_small = z_range_large - 0.06 # 0.06 숫자값은 배열크기, 공간크기에 맞춰 바꿔야할거같음
                                         # 1.422 * 공간최대크기 / 배열크기
    z_range_small[0] = 0

    # 위 데이터프레임에 넣을 컬럼값의 리스트 생성 (순서:Z증가/X증가/Y증가)
    # .transpose 로 축순서변경 (z,y,x) --> (y,x,z)
    arr = np.array(boundary_stack(df,z_range_small,z_range_large,x_dim,y_dim)).transpose(1,2,0)
    val_lst = []
    for i in arr:
        for j in i:
            for val in j:
                val_lst.append(val)
    
    # 0과 1로 구분된 데이터
    new_df["value"] = val_lst 
    
    # 향후 플로팅위해 정렬된 데이터프레임 저장
    new_df.to_csv(f"dataset/dataframe/yes_obstacle/{num}.csv",index=False)

    # 입/출구 반영
    new_df = make_bdr(new_df,inlist,outlist)
    
    # 최종 boundary conditin return
    return new_df["value"] * new_df["Boundary"], val_lst

def boundary_stack(df,z_range_small,z_range_large,x_dim,y_dim):
    '''Z축 단면 채워서 array 형태로 변환'''

    bdr_lst = []
    # boundary 생성
    cnt = 0
    for z_small,z_large in zip(z_range_small,z_range_large):
        x = df[(df["z"]>=z_small) & (df["z"]<=z_large)]["x"]
        y = df[(df["z"]>=z_small) & (df["z"]<=z_large)]["y"]

        # 이미지로 저장
        plt.scatter(x,y)
        plt.margins(0,0)
        plt.axis('off')
        plt.savefig(f'src/data/image/{cnt}image.png', bbox_inches='tight' ,pad_inches = 0)
        plt.clf()

        # 이미지 로드
        img = cv2.imread(f'src/data/image/{cnt}image.png',cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,dsize=(x_dim,y_dim))
        img = cv2.flip(img,0)

        bdr = np.ones([y_dim,x_dim])
        obstacle_pos = np.where(img==255) # 흰색으로 비어있는 부분이 장애물위치

        for i,j in zip(obstacle_pos[0],obstacle_pos[1]):
            bdr[i,j]=0

        bdr_lst.append(bdr)
        
        cnt += 1
    return bdr_lst

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

def manipulate_dataframe(num,df,dimension_setting,option):
    '''입력된 데이터프레임을 정렬(정렬된 격자데이터를 기준으로 보간)'''
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
    
    # 향후 플로팅위해 정렬된 데이터프레임 저장
    # alined_df.to_csv(f"dataset/dataframe/no_obstacle/{num}.csv",index=False)

    return alined_df


def make_bdr(df,inlist,outlist):
    '''Boundary condition 설정'''
    
    # 1. 전체 1로 칠함
    lst_1 = [1 for i in range(df.shape[0])]
    df["Boundary"] = lst_1 

    # 2. 벽면 2로 칠함
    poslist = ["x","y","z"]
    for i in poslist:
        df.loc[df[i]==np.min(df[i]),"Boundary"]=2
        df.loc[df[i]==np.max(df[i]),"Boundary"]=2
    
    # 3. 입/출구 좌표 근사값 찾기
    column_names = ["x","x","y","y","z","z"]
    inlet_pos = []
    outlet_pos = []
    for pos,column_name in zip(inlist,column_names):
        inlet_pos.append(find_nearest(pos,df,column_name)[0])
    for pos,column_name in zip(outlist,column_names):
        outlet_pos.append(find_nearest(pos,df,column_name)[0])  
    
    # 4. 입구 3 출구 4 지정
    condition1 = (df["x"]>=inlet_pos[0]) & (df["x"]<=inlet_pos[1]) & (df["y"]>=inlet_pos[2]) & (df["y"]<=inlet_pos[3]) & (df["z"]>=inlet_pos[4]) & (df["z"]<=inlet_pos[5])
    condition2 = (df["x"]>=outlet_pos[0]) & (df["x"]<=outlet_pos[1]) & (df["y"]>=outlet_pos[2]) & (df["y"]<=outlet_pos[3]) & (df["z"]>=outlet_pos[4]) & (df["z"]<=outlet_pos[5])
    df.loc[condition1,'Boundary'] = 3
    df.loc[condition2,'Boundary'] = 4

    return df

def find_nearest(pos,df,column_name):
    '''근접값 찾는 함수 (입/출구 boundary condition 생성시 사용)'''
    diffvalue = abs(np.unique(df[column_name])-pos)    
    index = np.where(diffvalue==np.min(diffvalue))
    return np.unique(df[column_name])[index] 
