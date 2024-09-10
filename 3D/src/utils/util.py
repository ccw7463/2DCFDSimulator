import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import cv2

def plotting(df,plot_value,setting_lst):
    '''plotting 함수'''
    # setting_lst : [plot크기(가로),plot크기(세로),값 세기,투명도,최소값,최대값,앵글각도(고도),앵글각도(방위)]
    
    x = np.asarray(df['x'])
    y = np.asarray(df['y'])
    z = np.asarray(df['z'])
    
    fig = plt.figure(figsize=(setting_lst[0],setting_lst[1]))
    ax1 = plt.axes(projection="3d")
    im = ax1.scatter3D(x,y,z,zdir='z', c=plot_value, cmap='jet',s=setting_lst[2], alpha=setting_lst[3], vmin=setting_lst[4], vmax=setting_lst[5])
    plt.colorbar(im)

    ax1.view_init(setting_lst[6],setting_lst[7])
    ax1.set_xlabel("x-dim")
    ax1.set_ylabel("y-dim")
    ax1.set_zlabel("z-dim")

    

def pyvista_plot(plot_array,opt_list,array_list):
    '''pyvista plot 하는 함수'''
    # plot : 플롯할 배열 (numpy array)
    # opt_list : [플롯시작값,플롯마지막값,개수,투명도,플롯시작범위,플롯마지막범위]
    # array_list : [x배열,y배열,z배열]
    
    value_1,value_2,value_num,value_opa,clim_1,clim2 = opt_list    
    x,y,z = array_list
    grid = pv.StructuredGrid(x, y, z)    
    grid["value"] = plot_array
    
    values = np.linspace(value_1,value_2 ,num=value_num)
    surfaces = [grid.contour([v]) for v in values]
    
    pt = pv.Plotter(notebook=0,shape=(1,1),border=False)
    pt.subplot(0,0)
    
    for i in range(len(values)):
        pt.add_mesh(surfaces[i], opacity=value_opa, clim=[clim_1,clim2],cmap='jet')
    pt.add_text("Interpolatiing Data",position='upper_edge',font_size=12)
    pt.show_bounds(grid='front') # 좌표값 표시 유무
    pt.add_bounding_box() # 전체 박스 표시유무
    pt.link_views(views=(0,0)) # 카메라 각도 / 근데 숫자바꾸니까 에러나네..(list index out of range)
    pt.show()


def scaling_for_IE_ploation(val_test,config):
    '''내삽 외삽 테스트 위한 스케일링'''
    iv_max = config.iv_max
    it_max = config.it_max
    wt_max = config.wt_max
    angle_max = config.angle_max

    iv = val_test[0]/iv_max
    it = val_test[1]/it_max
    wt = val_test[2]/wt_max
    angle = val_test[3]/angle_max
    
    return np.array([iv, it,wt, angle, 0])[np.newaxis,:]


def check_simulation_info(plot_num, df_plot,simulation_df_no_obs, simulation_df_yes_obs, boundary_data, geometry_plot=True):
    '''시뮬레이션 정보 확인'''
    # 매개변수 : 플롯할 넘버, 테스트 데이터프레임, 시뮬레이션정보담긴 데이터프레임 두개, 공간정보데이터(스플릿 결과로얻음), 공간정보 플로팅 유무
    
    # 플롯할 테스트넘버 및 운전조건
    if df_plot[plot_num].startswith("no"):
        print("장애물없는 데이터입니다.")
        test_num = int(df_plot[plot_num].strip(".csv").strip("no_"))
        df = pd.read_csv(f"dataset/dataframe/no_obstacle/{str(test_num)+'.csv'}")
        print("========시뮬레이션 조건========")
        print(simulation_df_no_obs.loc[test_num])
    else:
        print("장애물있는 데이터입니다.")
        test_num = int(df_plot[plot_num].strip(".csv").strip("yes_"))
        df = pd.read_csv(f"dataset/dataframe/yes_obstacle/{str(test_num)+'.csv'}")
        print("========시뮬레이션 조건========")
        print(simulation_df_yes_obs.loc[test_num])

    if geometry_plot:
        print("======== 공간구조 확인 ========")
        # 공간구조확인
        fig = plt.figure(figsize=(6,6))
        ax1 = plt.axes(projection="3d")
        im = ax1.scatter3D(df["x"],df["y"],df["z"],zdir='z', c=boundary_data[plot_num], cmap='jet', s=0.5, alpha=0.5)
        plt.colorbar(im)

    return df


def manipulate_2D_to_3D(obstacle_value_lst,boundary):

    img = cv2.imread('temp/image1.png')
    img2 = cv2.imread('temp/image2.png')
    img3 = cv2.imread('temp/image3.png')
    plt.figure(figsize=(12,10))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.subplot(1,3,3)
    plt.imshow(img3)