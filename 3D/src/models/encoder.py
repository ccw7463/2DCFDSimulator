import tensorflow as tf
import numpy as np
class CFDUnet_Encoder(tf.keras.Model):
    '''Downsampling'''
    def __init__(self, config):
        super(CFDUnet_Encoder, self).__init__()
        self.dim = config.dimension_setting
        self.dim.append(1)

        # 레이어정의
        self.input_layer = tf.keras.layers.Conv3D(config.hidden_size/4, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='input_layer', input_shape=self.dim, activation='relu')
        self.conv_layer_1 = tf.keras.layers.Conv3D(config.hidden_size/4, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_1', activation='relu')
        self.max_pool_1 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides = 2)
        self.avg_pool_1 = tf.keras.layers.AveragePooling3D(pool_size=(2,2,2), strides = 2)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.conv_layer_2 = tf.keras.layers.Conv3D(config.hidden_size/2, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_2', activation='relu')
        self.conv_layer_3 = tf.keras.layers.Conv3D(config.hidden_size/2, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_3', activation='relu')
        self.max_pool_2 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides = 2)
        self.avg_pool_2 = tf.keras.layers.AveragePooling3D(pool_size=(2,2,2), strides = 2)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.conv_layer_4 = tf.keras.layers.Conv3D(config.hidden_size, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_4', activation='relu')
        self.conv_layer_5 = tf.keras.layers.Conv3D(config.hidden_size, (3,3,3), strides=(1,1,1), padding='same', use_bias=False, name='conv_layer_5', activation='relu')
        self.max_pool_3 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides=2)
        self.avg_pool_3 = tf.keras.layers.AveragePooling3D(pool_size=(2,2,2), strides=2)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_4 = tf.keras.layers.LayerNormalization(epsilon=1e-6) # 6*6*6*256

        self.dense_layer = tf.keras.layers.Dense(config.hidden_size) #1*216*256 1x256

    def call(self, inp):  
        
        x = inp
        
        # down sampling 1
        x = self.input_layer(x)
        skip_connect_1 = self.conv_layer_1(x) # skip-connection 1       
        x_1 = self.max_pool_1(skip_connect_1)
        x_2 = self.avg_pool_1(skip_connect_1)
        x = x_1+x_2
        x = self.layer_norm_1(x)

        # down sampling 2
        x = self.conv_layer_2(x)
        skip_connect_2 = self.conv_layer_3(x)  # skip-connection 2
        x_1 = self.max_pool_2(skip_connect_2)
        x_2 = self.avg_pool_2(skip_connect_2)
        x = x_1+x_2
        x = self.layer_norm_2(x)
        
        # down sampling 3
        x = self.conv_layer_4(x)
        skip_connect_3 = self.conv_layer_5(x)  # skip-connection 3
        x_1 = self.max_pool_3(skip_connect_3)
        x_2 = self.avg_pool_3(skip_connect_3)
        x = x_1+x_2
        x = self.layer_norm_3(x)
        print("pre dense layer x :",x.shape)
        shape_0 = x.shape[1]
        shape_1 = x.shape[2]
        shape_2 = x.shape[3]
        shape_3 = x.shape[4]
        
        # dense layer
        x = self.dense_layer(x)
        print("after dense layer x :",x.shape)
        x = tf.reshape(x, [-1,shape_0*shape_1*shape_2,shape_3])
        print("after down sampling x :",x.shape)
        # x : 압축된 벡터
        return x,shape_0,shape_1,shape_2,shape_3,skip_connect_1,skip_connect_2,skip_connect_3


        