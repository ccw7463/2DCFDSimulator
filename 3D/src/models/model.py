import tensorflow as tf
from src.models.encoder import *
from src.models.decoder import *
from src.models.embed import *
from src.models.transformer import * 

class CFDUnet(tf.keras.Model):
    '''유동장 예측 모델'''
    def __init__(self, config):
        super(CFDUnet, self).__init__()       

        # 모델정의
        self.downsampling_model = CFDUnet_Encoder(config) 
        self.numeric_model = NumericalDense(config) 
        self.transformer_layer = Transformer(config)        
        self.upsampling_model = CFDUnet_Decoder(config)     

        # 레이어정의        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
        self.dense_layer = tf.keras.layers.Dense(config.hidden_size) 
        self.conv_layer_1 = tf.keras.layers.Conv3D(config.hidden_size*2, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_1')
        self.lrelu_1=tf.keras.layers.LeakyReLU()        
        self.conv_layer_2 = tf.keras.layers.Conv3D(config.hidden_size*2, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_2')
        self.lrelu_2=tf.keras.layers.LeakyReLU()
        
    def call(self, inp):
         
        x,text1,text2=inp
        print("x :",x.shape)
        print("text1 :",text2.shape)
        print("text2 :",text2.shape)

        # 1. downsampling
        x,shape_0,shape_1,shape_2,shape_3,skip_connect_1,skip_connect_2,skip_connect_3 = self.downsampling_model(x)
        print("x :",x.shape)

        # 2. numeric
        text = self.numeric_model(text1,text2)
        print("text :",text.shape)

        text = tf.concat([text]*shape_0*shape_1*shape_2,axis=1)              
        print("text :",text.shape)

        # 3. downsampling 결과와 numeric 결과 concat
        x = x+text
        print("결합한뒤 x:",x.shape)
        x = self.layer_norm(x)
        print("레이어 정규화후 x:",x.shape)
        x = self.dense_layer(x)
        print("dense layer 통과 후 x:",x.shape)
        
        # 4. transformer lyaer 통과
        x = self.transformer_layer(x)
        
        # 5. 추가 레이어 통과
        x = tf.reshape(x, [-1, shape_0, shape_1, shape_2, shape_3])
        x = self.conv_layer_1(x)
        x = self.lrelu_1(x)
        
        down_output = self.conv_layer_2(x)
        down_output = self.lrelu_2(down_output)
        
        # 6. upsampling
        x = self.upsampling_model(down_output,skip_connect_1,skip_connect_2,skip_connect_3)
    
        return x


