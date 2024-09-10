import numpy as np

def operating_division(operating_conditions,idx2char):    
    '''운전조건 데이터를 스페셜토큰과 값으로 분할'''
    cons = []
    for operating_condition in operating_conditions:
        condition_names = []
        values = []
        for i in range(len(operating_condition)):
            # 스페셜 토큰 
            if i%2 == 0:
                condition_names.append(operating_condition[i])            
            # 값  
            else:
                values.append(float(idx2char[operating_condition[i]]))
        cons.append([condition_names, values])
    cons = np.array(cons)
    
    return cons

def operating_scaling(operating_data,config):
    
    iv_max = config.iv_max
    it_max = config.it_max
    wt_max = config.wt_max
    angle_max = config.angle_max

    # 운전조건 데이터 scaling 수행
    operating_data[:,0] = operating_data[:,0]
    operating_data[:,1,0] = operating_data[:,1,0]/iv_max
    operating_data[:,1,1] = operating_data[:,1,1]/it_max
    operating_data[:,1,2] = operating_data[:,1,2]/wt_max
    operating_data[:,1,3] = operating_data[:,1,3]/angle_max
    
    return operating_data

