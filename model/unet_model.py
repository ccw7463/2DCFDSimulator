import torch.nn as nn
import torch

class CBR(nn.Module):
    '''( Conv - Batch - ReLU ) * 2 '''

    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(CBR,self).__init__()

        # 중간 채널 사이즈 지정X 일경우 "중간채널크기 = 출력채널크기"
        if mid_channel:
            pass
        else:
            mid_channel = out_channel
        
        # 레이어정의 
        self.CBR_layer = nn.Sequential(
            nn.Conv2d(in_channel,mid_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.CBR_layer(x)


class Down(nn.Module):
    '''Down Sampling'''

    def __init__(self,in_channel,out_channel):
        super(Down,self).__init__()
        self.DownSampling = nn.Sequential(
            nn.MaxPool2d(2), # 숫자값 한개 기입시, kernel_size=2, stride=2 의미를 지님
            CBR(in_channel,out_channel)
        )
    def forward(self,x):
        return self.DownSampling(x)

class Up(nn.Module):
    '''Up Sampling'''

    def __init__(self,in_channel,out_channel,kernel_size_num,stride_num):
        super(Up,self).__init__()
        self.ConvT = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=kernel_size_num, stride=stride_num) # 채널 수 줄임
        self.CBR = CBR(in_channel,out_channel)
        
    def forward(self,older,new):
        new = self.ConvT(new)
        new = torch.cat([older,new],dim=1)    
        result = self.CBR(new)
        return result

class OutConv(nn.Module):
    '''Out Convolution'''

    def __init__(self,in_channel,out_channel):
        super(OutConv,self).__init__()
        self.OutConv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1)
        )
    
    def forward(self,x):
        return self.OutConv(x)


class embedding(nn.Module):
    '''embedding operating condition'''
    def __init__(self):
        super(embedding,self).__init__()

        self.embedding_layer = nn.Sequential(
            nn.Linear(8,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),        
            nn.Linear(1024,512),
            nn.ReLU(),      
            nn.Linear(512,256),  
        )


    def forward(self, operating): 
        # print("operating 진입 :", operating.shape)    
        operating = operating.float() 
        # print("operating 타입변환 :", operating.shape)    
        outputs = self.embedding_layer(operating) # (batch, 5, 256)
        # print("outputs :", outputs.shape)

        # shape 변경
        outputs = torch.reshape(outputs, [-1, 1, 256]) # batch size x 1 x 256 
        # print("outputs :", outputs.shape)
        return outputs

class UNet(nn.Module):
    '''Main Class'''
    def __init__(self):
        super(UNet,self).__init__()
        
        self.first_conv = CBR(1,32)
        self.Down_1 = Down(32,64)
        self.Down_2 = Down(64,128)
        self.Down_3 = Down(128,256)
        # self.Down_4 = Down(512,1024)
        
        self.numeric_model = embedding()
        self.dense_layer = nn.Linear(256,256)

        self.Up_1 = Up(256,128,3,2)
        self.Up_2 = Up(128,64,2,2)
        self.Up_3 = Up(64,32,3,2)

        self.Out = OutConv(32,3)

    def forward(self,inp):
        '''
        나중에추가해야할수도있는것
        1. layer norm
        2. 결합한뒤 dense layer
        '''
        
        # Down Sampling 

        x, operating = inp
        # print("처음 입력 x",x.shape)
        # print("처음 입력 operating",operating.shape)
        x_1 = self.first_conv(x)
        # print("x_1",x_1.shape)
        x_2 = self.Down_1(x_1)
        # print("x_2",x_2.shape)
        x_3 = self.Down_2(x_2)
        # print("x_3",x_3.shape)
        x_4 = self.Down_3(x_3)

        # print("down sampling 끝:",x_4.shape)
        # down sampling 끝: torch.Size([4, 256, 12, 12])

        shape_batch = x_4.shape[0] # batch
        shape_0 = x_4.shape[1] # channel
        shape_1 = x_4.shape[2] # width
        shape_2 = x_4.shape[3] # height

        temp = torch.reshape(x_4,[shape_batch,shape_0,shape_1*shape_2])
        # print("shape 변환 후:",temp.shape)

        text = self.numeric_model(operating)
        # print('text_0:',text.shape)
        text = text.permute(0,2,1)
        # print('text_1:',text.shape)
        text = torch.concat([text]*shape_1*shape_2,axis=2)     
        # print('text_2:',text.shape)
        temp = temp+text
        # print("결합완료 :",temp.shape)
        # torch.Size([4, 256, 144])
        # x = self.dense_layer(x)
        # print("dense layer 통과 후:",x.shape)
        temp = torch.reshape(temp,[shape_batch,shape_0,shape_1,shape_2])
        # print("reshape 후:",temp.shape)

        # Up Sampling
        x_up_1 = self.Up_1(x_3,temp)
        # print("x_up_1",x_up_1.shape)
        x_up_2 = self.Up_2(x_2,x_up_1)
        # print("x_up_2",x_up_2.shape)
        x_up_3 = self.Up_3(x_1,x_up_2)
        # print("x_up_3",x_up_3.shape)

        # x_up_1 = self.Up_4(x_1,x_up_3)
        # print("x_up_1",x_up_1.shape)

        # Output
        x = self.Out(x_up_3)
        # print("최종 출력 :",x.shape)
        return x

        