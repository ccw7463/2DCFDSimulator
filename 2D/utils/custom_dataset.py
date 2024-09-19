from torch.utils.data import Dataset
from utils.util import *

class CustomDatasetUNet(Dataset):
    def __init__(self, transform=None, target_transform=None):
        
        # 입/출력 데이터 가져오기 (운전정보 제외)
        PreprocessObj = PreprocessData()
        PreprocessObj.get_solve_array()
        PreprocessObj.scaling_and_resize()
        
        # 운전정보 데이터
        PreprocessObj.get_operating_dataset()
        
        # 데이터 지정
        self.boundary_data = PreprocessObj.boundary_data
        self.operating_data = PreprocessObj.operating_data
        self.Y_data = [PreprocessObj.ux_data,PreprocessObj.uy_data,PreprocessObj.p_data]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.boundary_data)

    def __getitem__(self,idx):

        # 입/출력데이터 (운전정보제외) 지정
        boundary = self.boundary_data[idx]
        ux = self.Y_data[0][idx]
        uy = self.Y_data[1][idx]
        p = self.Y_data[2][idx]
        
        # 운전정보 데이터 지정
        operating = self.operating_data[idx]
        
        # 데이터 전처리
        if self.transform:
            boundary = self.transform(boundary)
        if self.target_transform:
            ux = self.target_transform(ux)
            uy = self.target_transform(uy)
            p = self.target_transform(p)

        return boundary,operating,ux,uy,p