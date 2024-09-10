import tensorflow as tf
import numpy as np


### 포지셔널 인코딩
def get_angles(pos, i, hidden_size):
    angle_rates = 1 / np.power(10000, (2 * i//2) / np.float32(hidden_size))
    return pos * angle_rates

def positional_encoding(position, hidden_size):
    '''
    위치정보반영위한 positional encoding 수행
    '''    
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(hidden_size)[np.newaxis, :],
                          hidden_size)

    # 짝수에는 sin 적용
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 홀수에는 cos 적용
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)



### 어텐션
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    # query key 곱
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # query key 곱을 스케일링
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 마스킹
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # attention weight 구함
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # attention value 구함
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


### 멀티헤드 어텐션
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size

        assert self.hidden_size % self.num_heads == 0

        self.depth = self.hidden_size // self.num_heads

        self.wq = tf.keras.layers.Dense(self.hidden_size)
        self.wk = tf.keras.layers.Dense(self.hidden_size)
        self.wv = tf.keras.layers.Dense(self.hidden_size)

        self.dense = tf.keras.layers.Dense(self.hidden_size)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

### 포인트 와이즈 피드포워드 네트워크
def point_wise_feed_forward_network(config):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(config.mlp_dim, activation='gelu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(config.hidden_size)  # (batch_size, seq_len, d_model)
    ])

### 인코더 레이어
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(config)
        self.ffn = point_wise_feed_forward_network(config)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(config.rate)
        self.dropout2 = tf.keras.layers.Dropout(config.rate)

    def call(self, x, mask):
        x_copy = x
        x = self.layernorm1(x)
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = x_copy + attn_output 
        # out1 = self.layernorm1(x + attn_output)

        x = self.layernorm2(out1) 
        ffn_output = self.ffn(x) 
        ffn_output = self.dropout2(ffn_output)
        out2 = x + ffn_output 

        return out2


    
### 인코더
class Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.hidden_size=config.hidden_size
        self.num_layers=config.num_layers
        self.maximum_position_encoding=config.maximum_position_encoding
        self.rate=config.rate

#        self.embedding = tf.keras.layers.Embedding(kargs['vocab_size'], self.d_model)
        self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.hidden_size)

        self.enc_layers = [EncoderLayer(config) 
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, mask):

        seq_len = tf.shape(x)[1]

#        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x


### 트랜스포머 모델

class Transformer(tf.keras.Model):
    def __init__(self, config):
        super(Transformer, self).__init__(name=config.model_name)
        self.encoder = Encoder(config)
    
    def call(self, x):
        inp = x
        enc_output = self.encoder(inp, None)
        return enc_output


### 모델 로스 정의
def loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def accuracy(real, pred):
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask    
    acc = train_accuracy(real, pred)

    return tf.reduce_mean(acc)





# ### 패딩 및 포워드 마스킹
# def create_padding_mask(seq):
#     '''attention logits을 위해 패딩을 추가하여 차원늘림'''
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)    
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)

# def create_masks(inp, tar):
#     '''
#     인코더 패딩마스크
    
#     1. 디코더의 두번째 attention block과 인코더의 출력에 마스킹하는데 사용
#     2. 디코더의 첫번째 attention block에서 사용됨
#     '''
#     enc_padding_mask = create_padding_mask(inp)

#     dec_padding_mask = create_padding_mask(inp)

#     dec_target_padding_mask = create_padding_mask(tar)
    
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])    
    
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#     return enc_padding_mask, combined_mask, dec_padding_mask