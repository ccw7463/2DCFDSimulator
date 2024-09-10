import tensorflow as tf
import numpy as np

class NumericalDense(tf.keras.Model):
    '''운전조건의 수치파악을 위한 임베딩'''
    def __init__(self, config):
        super(NumericalDense, self).__init__()
        
        self.hidden_size=config.hidden_size

        # 스페셜 토큰을 임베딩 layers     
        self.special_token_embedding_layer = tf.keras.layers.Embedding(6, config.hidden_size, input_length=5)
        
        # 운전정보 숫자파악 Dense layers
        self.dense_layer = tf.keras.layers.Dense(config.hidden_size, activation = 'relu')
        
        # 그 외 레이어 정의
        self.flatten = tf.keras.layers.Flatten()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)        
        self.dense_layer_1 = tf.keras.layers.Dense(config.hidden_size*4, activation = 'relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.2)        
        self.dense_layer_2 = tf.keras.layers.Dense(config.hidden_size*2, activation = 'relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.2)        
        self.dense_layer_3 = tf.keras.layers.Dense(config.hidden_size, activation = 'relu')
                                                 
    def call(self, inp1, inp2):     

        tokens,values = inp1,inp2
        print("tokens :",tokens)
        print("values :",values)
        
        # 스페셜 토큰 임베딩
        toks = self.special_token_embedding_layer(tokens)
        print("toks :",toks)

        # 운전정보 숫자 임베딩
        vals = self.dense_layer(values[:,:,np.newaxis])
        print("vals :",vals)

        #운전정보 값(ux,uy..) + 스페셜토큰 
        x = vals+toks
        print("vals+toks :",x.shape)
        
        # layer 통과
        x = self.flatten(x)
        x = self.layer_norm_1(x)
        x = self.dense_layer_1(x)
        x = self.dropout_1(x)
        x = self.dense_layer_2(x)
        x = self.dropout_2(x)
        
        # 출력 Embedding vector
        outputs = self.dense_layer_3(x) # batch size x 256 
        print("outputs :",outputs.shape)
        
        outputs = tf.reshape(outputs, [-1, 1, self.hidden_size]) # batch size x 1 x 256 
        print("outputs :",outputs.shape)
        
        return outputs