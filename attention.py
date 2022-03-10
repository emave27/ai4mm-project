import tensorflow as tf
from keras.layers import Dense, Layer

class BahdanauAttention(Layer):
  def __init__(self, units, verbose=0):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)
    self.verbose= verbose

  def call(self, query, values):
    #values shape: (batch_size, max_len, hidden size) --> encoder output shape
    #query shape: (batch_size, hidden size) --> decoder output shape

    #reshape query tensor to compute score value
    query_with_time_axis=tf.expand_dims(query, 1)
    #query_with_time_axis shape: (batch_size, 1, hidden size)
    
    #compute alignment score
    score=self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    #score shape: (batch_size, max_length, 1)
    
    #apply softmax to score to get attention weights
    attention_weights=tf.nn.softmax(score, axis=1)
    
    #compute the context vector
    context_vector=attention_weights * values
    context_vector=tf.reduce_sum(context_vector, axis=1)
    #context_vector shape after sum: (batch_size, hidden_size)

    return context_vector, attention_weights
