import tensorflow as tf
import numpy as np

class CFDUnet_Decoder(tf.keras.Model):
    '''Upsampling'''
    def __init__(self, config):
        super(CFDUnet_Decoder, self).__init__()

        # 레이어정의
        # up sampling 1
        self.upsample_layer_1 = tf.keras.layers.Conv3DTranspose(config.hidden_size, (2,3,2), strides=(2,2,2), use_bias=False, name = 'upsample_layer_1')
        self.lrelu_1=tf.keras.layers.LeakyReLU()
        self.upsample_conv_1 = tf.keras.layers.Conv3D(config.hidden_size, (3,3,3), padding='same', use_bias=False, name='upsample_conv_1')
        self.lrelu_2=tf.keras.layers.LeakyReLU()
        self.upsample_conv_2 = tf.keras.layers.Conv3D(config.hidden_size, (3,3,3), padding='same', use_bias=False, name='upsample_conv_2')
        self.lrelu_3=tf.keras.layers.LeakyReLU()
        
        # up sampling 2
        self.upsample_layer_2 =  tf.keras.layers.Conv3DTranspose(config.hidden_size/2, (2,2,2), strides=(2,2,2), use_bias=False, name = 'upsample_layer_2')
        self.lrelu_4=tf.keras.layers.LeakyReLU()
        self.upsample_conv_3 = tf.keras.layers.Conv3D(config.hidden_size/2, (3,3,3), padding='same', use_bias=False, name='upsample_conv_3')
        self.lrelu_5=tf.keras.layers.LeakyReLU()
        self.upsample_conv_4 = tf.keras.layers.Conv3D(config.hidden_size/2, (3,3,3), padding='same', use_bias=False, name='upsample_conv_4')
        self.lrelu_6=tf.keras.layers.LeakyReLU()

        # up sampling 3
        self.upsample_layer_3 =  tf.keras.layers.Conv3DTranspose(config.hidden_size/4, (2,2,2), strides=(2,2,2), use_bias=False, name = 'upsample_layer_3')
        self.lrelu_7=tf.keras.layers.LeakyReLU()
        self.upsample_conv_5 = tf.keras.layers.Conv3D(config.hidden_size/4, (3,3,3), padding='same', use_bias=False, name='upsample_conv_5')
        self.lrelu_8=tf.keras.layers.LeakyReLU()
        self.upsample_conv_6 = tf.keras.layers.Conv3D(config.hidden_size/4, (3,3,3), padding='same', use_bias=False, name='upsample_conv_6')
        self.lrelu_9=tf.keras.layers.LeakyReLU()

        # output layer 
        self.output_layer_1 = tf.keras.layers.Conv3D(config.hidden_size/8, (1,1,1), padding='same', use_bias=False, name='output_layer_1')
        self.lrelu_10=tf.keras.layers.LeakyReLU()
        self.output_layer_2 = tf.keras.layers.Conv3D(config.hidden_size/16, (1,1,1), padding='same', use_bias=False, name='output_layer_2')
        self.lrelu_11=tf.keras.layers.LeakyReLU()
        self.output_layer_3 = tf.keras.layers.Conv3D(1, (1,1,1), padding = 'same', use_bias=False, name='output_layer_3')
                                                 
    def call(self, *inp):
        
        (down_output,skip_connect_1,skip_connect_2,skip_connect_3) = inp
        
        # up_sampling 1
        up_sample_1 = self.upsample_layer_1(down_output)
        up_sample_1 = self.lrelu_1(up_sample_1)
        concat_1 = tf.keras.layers.Concatenate()([up_sample_1, skip_connect_3])  # skip-connection 1
        x = self.upsample_conv_1(concat_1)
        x = self.lrelu_2(x)
        x = self.upsample_conv_2(x)
        x = self.lrelu_3(x)
        
        # up sampling 2
        up_sample_2 = self.upsample_layer_2(x)
        up_sample_2 = self.lrelu_4(up_sample_2)
        concat_2 = tf.keras.layers.Concatenate()([up_sample_2, skip_connect_2])  # skip-connection 2
        x = self.upsample_conv_3(concat_2)
        x = self.lrelu_5(x)
        x = self.upsample_conv_4(x)    
        x = self.lrelu_6(x)

        # up sampling 3
        up_sample_3 = self.upsample_layer_3(x)
        up_sample_3 = self.lrelu_7(up_sample_3)
        concat_3 = tf.keras.layers.Concatenate()([up_sample_3, skip_connect_1])  # skip-connection 2
        x = self.upsample_conv_5(concat_3)
        x = self.lrelu_8(x)
        x = self.upsample_conv_6(x)
        x = self.lrelu_9(x)
        
        # output layer
        x = self.output_layer_1(x)
        x = self.lrelu_10(x)
        x = self.output_layer_2(x)
        x = self.lrelu_11(x)
        x = self.output_layer_3(x)
        
        return x