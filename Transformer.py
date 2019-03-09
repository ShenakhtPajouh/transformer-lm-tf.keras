import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.layers.base import Layer
from Utils import shape_list

def gelu(x):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    
    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.
    """
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
  
def swish(x):
    """
    Swish tends to work better than ReLU on deeper models across a number of challenging data sets.
    For further information:
    medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820
    
    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the swish activation applied.
    """
    return x * tf.nn.sigmoid(x)

act_fns = {
    'relu': tf.nn.relu,
    'swish': swish,
    'gelu': gelu
}

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

class Norm(Layer):
    """
    n_state = shape_list(x)[-1]
    """

    def __init__(self, name, n_state, **kwargs):
        super().__init__(name, **kwargs)
        self.n_state = n_state
        self.g = self.add_weight(name = 'g', shape=[self.n_state], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(1))
        self.b = self.add_weight(name = "b", shape=[self.n_state], initializer=tf.keras.initializers.constant(0))

    def call(self, inputs, **kwargs):
        return self._norm(inputs, self.g, self.b, axis=[-1])

    def _norm(self, x, g = None, b = None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x * g + b
        return x

    def compute_output_shape(self, input_shape):
        return super(Norm, self).compute_output_shape(input_shape)

class Conv1D(Model):

    def __init__(self, name, nx, nf, rf, **kwargs):
        super().__init__(name, **kwargs)
        self.nx = nx
        self.nf = nf
        self.rf = rf

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=[self.rf, self.nx, self.nf], dtype=tf.float32,
                                 initializer=tf.keras.initializers.random_normal(stddev=0.02))
        self.b = self.add_weight(name="b", shape=[self.nf], initializer=tf.keras.initializers.constant(0))
        super(Conv1D, self).build(input_shape = input_shape)

    def call(self, inputs, **kwargs):
        if self.rf == 1:
            c = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.nx]), tf.reshape(self.w, [-1, self.nf])) + self.b,
                           shape_list(inputs)[:-1] + [self.nf])
        else:
            c = tf.nn.conv1d(value=inputs, filters=self.w, stride=1, padding='VALID') + self.b
        return c

    def compute_output_shape(self, input_shape):
        return super(Conv1D, self).compute_output_shape(input_shape)

class Attention(Model):
    """
    nx = shape_list(x)[-1]
    where x in inputs argm of call
    """

    def __init__(self, name, nx, n_state, n_head, attn_pdrop, resid_pdrop, train, scale=False, **kwargs):
        super().__init__(name = name, **kwargs)
        self.nx = nx
        self.n_state = n_state
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.train = train
        self.scale = scale
        self.conv1d_c = Conv1D(name = 'c_attn', nx=self.nx,nf=self.n_state*3,rf=1)
        self.conv1d_a = Conv1D(name = 'c_proj', nx=self.nx,nf=self.n_state,rf=1)

    def call(self, inputs):
        c = self.conv1d_c(inputs)
        q, k, v = tf.split(c, 3, 2)
        q = self.split_heads(q, self.n_head)
        k = self.split_heads(k, self.n_head, k=True)
        v = self.split_heads(v, self.n_head)
        a = self._attn(q, k, v)
        a = self.merge_heads(a)
        a = self.conv1d_a(a)
        a = dropout(a, self.resid_pdrop, self.train)
        return a

    def split_states(self, x, n):
        x_shape = shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1] + [n, m // n]
        return tf.reshape(x, new_x_shape)

    def merge_states(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x, n, k=False):
        if k:
            return tf.transpose(self.split_states(x, n), [0, 2, 3, 1])
        else:
            return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])

    def merge_heads(self, x):
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(self, w):
        n = shape_list(w)[-1]
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
        b = tf.reshape(b, [1, 1, n, n])
        w = w * b + -1e9 * (1 - b)
        return w

    def _attn(self, q, k, v):
        w = tf.matmul(q, k)
        if self.scale:
            n_state = shape_list(v)[-1]
            w = w * tf.rsqrt(tf.cast(n_state, tf.float32))
        w = self.mask_attn_weights(w)
        w = tf.nn.softmax(w)
        w = dropout(w, self.attn_pdrop, self.train)
        a = tf.matmul(w, v)
        return a

    def compute_output_shape(self, input_shape):
        return super(Attention, self).compute_output_shape(input_shape)

class MLP(Model):
    def __init__(self, name, n_embd, n_state, afn, resid_pdrop, train):
        super().__init__(name = name)
        self.n_embd = n_embd
        self.n_state = n_state
        self.act = act_fns[afn]
        self.resid_pdrop = resid_pdrop
        self.train = train
        self.conv_fc = Conv1D("c_fc", self.n_embd, self.n_state, 1)
        self.conv_proj = Conv1D("c_proj", self.n_state, self.n_embd, 1)

    def call(self, inputs):
        hidden1 = self.act(self.conv_fc(inputs))
        hidden2 = self.conv_proj(hidden1)
        hidden2 = dropout(hidden2, self.resid_pdrop, self.train)
        return hidden2

class Block(Model):
    def __init__(self, name, n_vocab, n_ctx, n_embd, n_head, attn_pdrop, resid_pdrop, afn, train, scale):
        """
          Args:
            name: The name of the model
            n_vocab: Size of the vocabulary
            n_ctx: Size of the context
            n_embd: Embeddings dimension
            n_layer: Number of the transformer blocks
            n_head: Number of attention heads
            attn_pdrop: The dropout probability for attention layer
            resid_pdrop: The dropout probability for ?
            afn: The non-linear activation function in MLP
            train: It is a boolean which is true for training model, false for eval model (to control dropout)
            scale: ?
        """
        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.train = train

        self.attn = Attention("/attn", n_embd, n_embd, n_head, attn_pdrop, resid_pdrop, train, scale)
        self.norm1 = Norm("/ln_1", n_embd)
        self.mlp = MLP("/mlp", n_embd, 4 * n_embd, afn, resid_pdrop, train)
        self.norm2 = Norm("/ln_2", n_embd)

    def call(self, inputs):
        a = self.attn(inputs)
        n = self.norm1(inputs + a)
        m = self.mlp(n)
        h = self.norm2(n + m)
        return h

class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, name, n_vocab, n_ctx=128, n_embd=768, stddev=0.02, trainable=True):
        super().__init__(trainable = trainable)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.stddev = stddev
    
    def build(self, input_shape):
        self.we = self.add_weight(name = "we", shape = (self.n_ctx + self.n_vocab, self.n_embd),
                                  initializer = tf.random_normal_initializer(stddev=self.stddev))
        super(EmbeddingLayer, self).build(input_shape = input_shape)
        
    def call(self, inputs):
        return tf.reduce_sum(tf.gather(self.we, inputs), 2)

class Transformer(Model):
    def __init__(self, name, n_vocab, n_ctx=128, n_embd=768, n_layer=12, n_head=12, embd_pdrop=0.1, attn_pdrop=0.1,
                 resid_pdrop=0.1, afn="gelu", train = False, scale = False):
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
            resid_pdrop: The dropout probability for ?
            afn: The non-linear activation function in MLP
            train: It is a boolean which is true for training model, false for eval model (to control dropout)
            scale: ?
        """
        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.afn = afn
        self.train = train
        self.scale = scale
        self.embed = EmbeddingLayer("embedding", n_vocab, n_ctx, n_embd)

        self.transformer_stack = Sequential()
        for layer in range(n_layer):
            self.transformer_stack.add(
                Block("h", n_vocab, n_ctx, n_embd, n_head, attn_pdrop, resid_pdrop, afn, train, scale))

    def call(self, inputs):
        tokens = tf.reshape(inputs[0], [-1, self.n_ctx, 2])
        masks = tf.reshape(inputs[1], [-1, self.n_ctx])
        hidden1 = self.embed(tokens)
        self.embed.we = dropout(self.embed.we, self.embd_pdrop, self.train)
        hidden2 = self.transformer_stack(hidden1)
        hidden3 = tf.reshape(hidden2, [-1, self.n_embd])
        logits = tf.reshape(tf.matmul(hidden3, self.embed.we[:self.n_vocab, :], transpose_b=True),
                            [-1, self.n_ctx, self.n_vocab])
        logits_truncated = tf.reshape(logits[:, :-1], [-1, self.n_vocab])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_truncated,
                                                                labels=tf.reshape(tokens[:, 1:, 0], [-1]))
        losses = tf.reshape(losses, [shape_list(tokens)[0], shape_list(tokens)[1] - 1])
        losses = tf.reduce_sum(losses * masks[:, 1:], 1) / tf.reduce_sum(masks[:, 1:], 1)
        return logits, losses
