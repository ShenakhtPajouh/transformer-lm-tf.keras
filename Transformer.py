import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential,
from tensorflow.python.layers.base import Layer

from Utils import shape_list

act_fns = {
    'relu': tf.nn.relu,
    'swish': swish,
    'gelu': gelu
}


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    return x * tf.nn.sigmoid(x)


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


class Transformer(Model):
    def __init__(self, name, n_vocab, n_ctx=128, n_embd=768, n_layer=12, n_head=12, embd_pdrop=0.1, attn_pdrop=0.1,
                 resid_pdrop=0.1, afn="gelu"):
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
            afn: non linearity function name
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
        self.embed = EmbeddingLayer("/we", n_vocab, n_ctx, n_embd)

        self.transformer_stack = Sequential()
        for layer in range(n_layer):
            self.transformer_stack.add(
                Block("/h%d" % layer, n_vocab, n_ctx, n_embd, n_head, embd_pdrop, attn_pdrop, resid_pdrop, afn))

    def call(self, inputs, train=False):
        self.embed.we = dropout(self.embed.we, self.embd_pdrop, train)
        tokens = tf.reshape(inputs[0], [-1, self.n_ctx, 2])
        masks = tf.reshape(inputs[1], [-1, self.n_ctx])
        hidden1 = self.embed(tokens)
        hidden2 = self.transformer_stack(hidden1, train)
        hidden3 = tf.reshape(hidden2, [-1, self.n_embd])
        logits = tf.reshape(tf.matmul(hidden3, self.embed.we[:self.n_vocab, :], transpose_b=True),
                            [-1, self.n_ctx, self.n_vocab])
        logits_truncated = tf.reshape(logits[:, :-1], [-1, self.n_vocab])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits_truncated,
                                                                labels=tf.reshape(tokens[:, 1:, 0], [-1]))
        losses = tf.reshape(losses, [shape_list(tokens)[0], shape_list(tokens)[1] - 1])
        losses = tf.reduce_sum(losses * mask[:, 1:], 1) / tf.reduce_sum(mask[:, 1:], 1)
        return logits, losses


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, name, n_vocab, n_ctx=128, n_embd=768, stddev=0.02, trainable=True):
        super().__init__(name=name, trainable=trainable)
        self.we = self.add_weight(name=name, shape=(n_vocab + n_ctx, n_embd),
                                  initializer=tf.random_normal_initializer(stddev=stddev))

    def call(self, inputs):
        return tf.reduce_sum(tf.gather(self.we, inputs), 2)


class Block(Model):
    def __init__(self, name, n_vocab, n_ctx, n_embd, n_head, embd_pdrop, attn_pdrop, resid_pdrop, afn):
        """

          Args:
            name: The name of the model
            n_vocab: Size of the vocabulary
            n_ctx: Size of the context
            n_embd: Embeddings dimension
            n_head: Number of attention heads
            embd_pdrop: The dropout probability for embedding layers
            attn_pdrop: The dropout probability for attention layer
            resid_pdrop: ?
        """
        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        self.attn = Attention("/attn")
        self.norm1 = Norm("/ln_1")
        self.mlp = MLP("/mlp", afn, resid_pdrop)
        self.norm2 = Norm("/ln_2")

    def call(self, inputs, train, scale=False):
        nx = shape_list(inputs)[-1]
        a = self.attn(inputs, nx, train, scale)
        n = self.norm1(inputs + a)
        m = self.mlp(n, 4 * nx, train)
        return self.norm2(n + m)


class MLP(Model):
    def __init__(name, afn, resid_pdrop):
        super().__init__(name=name)
        self.act = act_fns[afn]
        self.resid_pdrop = resid_pdrop
        self.conv_fc = None
        self.conv_proj = None

    def call(name, inputs, n_state, train=False):
        nx = shape_list(inputs)[-1]
        hidden1 = act(self.conv_fc(inputs))
        hidden2 = self.conv_proj(hidden1)
        hidden2 = dropout(hidden2, self.resid_pdrop, train)
        return hidden2


########################################################################################################################
# Temporary layer definitions. Will be adjusted as needed.
########################################################################################################################

class Conv1D(Layer):

    def __init__(self, nx, nf, rf, **kwargs):
        self.nx = nx
        self.nf = nf
        self.rf = rf
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=[self.rf, self.nx, self.nf], dtype=tf.float32,
                                 initializer=tf.keras.initializers.random_normal(stddev=0.02))
        self.b = self.add_weight(name="b", shape=[self.nf], initializer=tf.keras.initializers.constant(0))
        super(Conv1D, self).build(input_shape=input_shape)

    def call(self, inputs,**kwargs):
        if self.rf == 1:
            c = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.nx]), tf.reshape(self.w, [-1, self.nf])) + self.b,
                           shape_list(inputs)[:-1] + [self.nf])
        else:
            c = tf.nn.conv1d(value=inputs, filters=self.w, stride=1, padding='VALID') + self.b
        return c

    def compute_output_shape(self, input_shape):
        return super(Conv1D, self).compute_output_shape(input_shape)
