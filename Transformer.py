import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Model, Sequential
import numpy as np

class Transformer(Model):
  def __init__(self, name, n_vocab, n_ctx = 128, n_embd = 768, n_layer = 12, n_head = 12, embd_pdrop = 0.1, attn_pdrop = 0.1, resid_pdrop = 0.1):
    """
      This is the transformer model in 
      'https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf'
      fine-tuned for language-model
      
      Args:
        name: The name of the model
        n_vocab: Size of the vocabulary
        n_ctx: Size of the context
        n_embd: Embeddings dimension
        n_layer: Number of the transformer blocks
        n_head: Number of attention heads
        embd_pdrop: The dropout probability for embedding layers
        attn_pdrop: The dropout probability for attention layer
        resid_pdrop: ?
    """
    super().__init__(name = name)
    self.name = name
    self.n_vocab = n_vocab
    self.n_ctx = n_ctx
    self.n_embd = n_embd
    self.n_head = n_head
    self.embd_pdrop = embd_pdrop
    self.attn_pdrop = attn_pdrop
    self.resid_pdrop = resid_pdrop
    self.embed = EmbeddingLayer(name + "/we", n_vocab, n_ctx, n_embd)
    
    self.transformer_stack = Sequential()
    for layer in range(n_layer):
      self.transformer_stack.add(Block(name + "/h%d" % layer, n_vocab, n_ctx, n_embd, n_head, embd_pdrop, attn_pdrop, resid_pdrop))
  
  def call(self, inputs, train = False):
    self.embed.we = dropout(self.embed.we, self.embd_pdrop, train)
    tokens = tf.reshape(inputs[0], [-1, self.n_ctx, 2])
    masks = tf.reshape(inputs[1], [-1, self.n_ctx])
    hidden = self.embed(tokens)
    hidden = self.transformer_stack(hidden, train)
    hidden = tf.reshape(hidden, [-1, self.n_embd])
    logits = tf.reshape(tf.matmul(hidden, we[:self.n_vocab, :], transpose_b = True), [-1, self.n_ctx, self.n_vocab])
    logits_truncated = tf.reshape(logits[:, :-1], [-1, self.n_vocab])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = lm_logits_truncated, labels = tf.reshape(tokens[:, 1:, 0], [-1]))
    losses = tf.reshape(losses, [shape_list(tokens)[0], shape_list(tokens)[1] - 1])
    losses = tf.reduce_sum(losses * mask[:, 1:], 1) / tf.reduce_sum(mask[:, 1:], 1)
    return logits, losses

class EmbeddingLayer(keras.layers.Layer):
  def __init__(self, name, n_vocab, n_ctx = 128, n_embd = 768, stddev = 0.02, trainable = True):
    super().__init__(name = name, trainable = trainable)
    self.we = self.add_weight(name = name, shape = (n_vocab + n_ctx, n_embd), initializer = tf.random_normal_initializer(stddev = stddev))
  
  def call(self, inputs):
    return tf.reduce_sum(tf.gather(self.we, inputs), 2)
  
def dropout(input_tensor, dropout_prob, train):
    """
      Perform dropout.
      Args:
        input_tensor: inpout tensor.
        dropout_prob: the probability of dropping out a value
        
      Returns:
        A version of `input_tensor` with dropout applied.
    """
    if not train or dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output
