import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import openpyxl

PAD = '<PAD>'
WT = '<wall_temperature>'
AN = '<Angle>'
IV = '<inlet_velocity>'
IT = '<inlet_temperature>'
NONE = '<NONE>'
MARKER = [PAD, WT, AN, IV, IT, NONE]
MAX_SEQUENCE = 10

def load_data():
    excel_path_no_obs='src/data/simulation/raw/no_obstacle_simulation.xlsx'
    excel_path_yes_obs='src/data/simulation/raw/yes_obstacle_simulation.xlsx'
    df_no_obs = pd.read_excel(excel_path_no_obs, engine='openpyxl')
    df_no_obs = df_no_obs.iloc[1:]
    df_yes_obs = pd.read_excel(excel_path_yes_obs,engine='openpyxl',header=3)
    df_yes_obs = df_yes_obs.iloc[1:,1:]
    # df_yes_obs = df_yes_obs[['File name','inlet_velocity','inlet_temperature','wall_temperature','Angle']]
    df_yes_obs['inlet_temperature'] = list(map(int,df_yes_obs['inlet_temperature']-273.15))
    df_yes_obs['wall_temperature'] = list(map(int,df_yes_obs['wall_temperature']-273.15))

    # csv로 새로 저장
    df_no_obs.to_csv("src/data/simulation/processed/no_obstacle.csv",index=False)
    df_yes_obs.to_csv("src/data/simulation/processed/yes_obstacle.csv",index=False)

    return df_no_obs, df_yes_obs

def generate_no_obstacle_operating_data():
    '''장애물 없는 경우 운전 데이터 생성'''
    vocab_path = 'src/data/vocab/vocab_no_obs.txt'
    df_no_obs,_ = load_data()
    tokens = process_data(df_no_obs)
    char2idx, idx2char = load_vocabulary(tokens,vocab_path)
    indexes = transform_number(tokens,char2idx) 
    np.save('dataset/processed/input/text_input/no_obstacle/operating_condition.npy',indexes)
    return char2idx, idx2char 

def generate_yes_obstacle_operating_data():
    '''장애물 있는 경우 운전 데이터 생성'''
    vocab_path = 'src/data/vocab/vocab_yes_obs.txt'    
    _,df_yes_obs = load_data()
    tokens = process_data(df_yes_obs)
    char2idx, idx2char = load_vocabulary(tokens,vocab_path)
    indexes = transform_number(tokens,char2idx)  
    np.save('dataset/processed/input/text_input/yes_obstacle/operating_condition.npy',indexes)
    return char2idx, idx2char 

def generate_total_operating_data():
    '''장애물 있는 경우 운전 데이터 생성'''
    vocab_path = 'src/data/vocab/vocab_total.txt'    
    df_no_obs,df_yes_obs = load_data()
    df_all = pd.concat([df_no_obs,df_yes_obs],ignore_index=True)
    df_all.drop(['Room_X','Sofa_ON/OFF','Table_Location'],axis=1,inplace=True)
    tokens = process_data(df_all)
    char2idx, idx2char = load_vocabulary(tokens,vocab_path)
    indexes = transform_number(tokens,char2idx)  
    np.save('dataset/processed/input/text_input/total/operating_condition.npy',indexes)
    return char2idx, idx2char 

def process_data(df):
    '''특정 컬럼데이터 로드 및 토큰 생성'''
    tokens = []
    for value in df['File name']:
        v = df[df['File name']==value]['inlet_velocity'].to_numpy()[0]
        it = df[df['File name']==value]['inlet_temperature'].to_numpy()[0]
        wt = df[df['File name']==value]['wall_temperature'].to_numpy()[0]
        angle = df[df['File name']==value]['Angle'].to_numpy()[0]
        token = generate_tokens(v, it, wt, angle)
        tokens.append(token)
    return tokens

def generate_tokens(v, it, wt, angle):
    '''텍스트로 운전조건 저장'''
    text_input = []
    text_input.append('<inlet_velocity>')
    text_input.append(str(v))
    text_input.append('<inlet_temperature>')
    text_input.append(str(it))
    text_input.append('<wall_temperature>')
    text_input.append(str(wt))
    text_input.append('<Angle>')
    text_input.append(str(angle))
    text_input.append('<NONE>')
    text_input.append('0')
    text_input = ' '.join(text_input) 
    return [text_input]

def load_vocabulary(tokens,vocab_path):
    ''' 입력된 tokens와 vocab파일(.txt) 을 사용하여 (단어: 인덱스 , 인덱스: 단어) 형태의 딕셔너리 생성'''
    if not os.path.exists(vocab_path):
        words = extract_property_value(tokens)
        words = list(set(words))
        words[:0] = MARKER
        register_vocab(words,vocab_path)
    vocab_list = load_vocab(vocab_path)
    char2idx, idx2char = make_vocabulary(vocab_list)
    return char2idx, idx2char

def extract_property_value(tokens):
    '''속성값 추출 (속성값 예 : 속도 2.38 2.15...)'''
    words = []
    for token in tokens:
        # sentence 예 : ['<inlet_velocity> 2.38 <inlet_temperature> 10 <wall_temperature> 10 <Angle> 50 <NONE> 0']
        for word in token[0].split(): 
            if word not in MARKER: 
                words.append(word)
    return [word for word in words if word]

def register_vocab(words,vocab_path):
    '''vocab 등록'''
    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        for word in words:
            vocab_file.write(word + '\n')

def load_vocab(vocab_path):
    '''vocab load'''
    vocab_list = []
    with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
        for line in vocab_file:
            vocab_list.append(line.strip())
    return vocab_list

def make_vocabulary(vocab_list):
    '''(단어: 인덱스 , 인덱스: 단어) 형태의 딕셔너리 생성'''
    char2idx = {char: idx for idx, char in enumerate(vocab_list)}
    idx2char = {idx: char for idx, char in enumerate(vocab_list)}
    return char2idx, idx2char

def transform_number(tokens,char2idx):
    '''토큰 전부 index(숫자)로 변환'''
    indexes = []
    for token in tokens:
        index = []
        for tok in token[0].split(' '):
            index.append(char2idx[tok])
        indexes.append(index)
    indexes = np.array(indexes)
    return indexes


