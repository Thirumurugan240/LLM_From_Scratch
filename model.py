import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Layer,
    Embedding,
    Dense,
    LayerNormalization,
    Dropout
)


VOCAB_SIZE = 50_257
MAX_LENGTH = 1_024

EMBED_DIM = 768
NUM_HEADS = 12

DFF = 3_072
NUM_LAYERS = 12
DROPOUT_RATE = 0.1

class MultiHeadSelfAttention(Layer):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads

        self.wq = Dense(embed_dim, use_bias=True)
        self.wk = Dense(embed_dim, use_bias=True)
        self.wv = Dense(embed_dim, use_bias=True)

        self.output_dense = Dense(embed_dim, use_bias=True)

        self.dropout = Dropout(dropout_rate)

    def split_heads(self, x, batch_size):

        x = tf.reshape(
            x,
            [batch_size, -1, self.num_heads, self.depth]
        )

        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    def call(self, x, mask=None, training=None):

        batch_size = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)


        attention_scores = tf.matmul(
            q,
            k,
            transpose_b=True
        )

        scale = tf.math.sqrt(
            tf.cast(self.depth, tf.float32)
        )

        attention_scores = attention_scores / scale


        if mask is not None:
            attention_scores += mask * -1e9

        attention_weights = tf.nn.softmax(
            attention_scores,
            axis=-1
        )

        attention_weights = self.dropout(
            attention_weights,
            training=training
        )

        attention_output = tf.matmul(
            attention_weights,
            v
        )

        attention_output = tf.transpose(
            attention_output,
            perm=[0, 2, 1, 3]
        )

        attention_output = tf.reshape(
            attention_output,
            [batch_size, -1, self.embed_dim]
        )

        return self.output_dense(attention_output)



class FeedForwardNetwork(Layer):

    def __init__(self, embed_dim, dff):
        super().__init__()

        self.dense1 = Dense(
            dff,
            activation=tf.nn.gelu
        )

        self.dense2 = Dense(
            embed_dim
        )

    def call(self, x):

        x = self.dense1(x)

        x = self.dense2(x)

        return x




















class TransformerBlock(Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dff,
        dropout_rate=0.1
    ):

        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            dff=dff
        )

        self.norm1 = LayerNormalization(
            epsilon=1e-5
        )

        self.norm2 = LayerNormalization(
            epsilon=1e-5
        )

    
        self.dropout1 = Dropout(
            dropout_rate
        )

        self.dropout2 = Dropout(
            dropout_rate
        )

    def call(
        self,
        x,
        mask=None,
        training=None
    ):

        # ----------------------------------------------------
        # Self Attention
        # ----------------------------------------------------

        attention_output = self.attention(
            x,
            mask=mask,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        # Residual connection + normalization
        x = self.norm1(
            x + attention_output
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        ffn_output = self.ffn(x)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        # Residual connection + normalization
        x = self.norm2(
            x + ffn_output
        )

        return x











class AIWithThiru(Model):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ):

        super().__init__(
            name="AI_with_Thiru"
        )

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.token_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="token_embedding"
        )

        # ----------------------------------------------------
        # Positional Embedding
        # ----------------------------------------------------

        self.position_embedding = Embedding(
            input_dim=max_length,
            output_dim=embed_dim,
            name="position_embedding"
        )

        # Embedding dropout
        self.embedding_dropout = Dropout(
            dropout_rate
        )

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        self.transformer_blocks = [

            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )

            for _ in range(num_layers)
        ]

        # ----------------------------------------------------
        # Final Layer Normalization
        # ----------------------------------------------------

        self.final_norm = LayerNormalization(
            epsilon=1e-5,
            name="final_layer_norm"
        )


    # ========================================================
    # CAUSAL MASK
    # ========================================================

    def create_causal_mask(self, seq_len):

        # Lower triangular matrix
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        mask = tf.linalg.band_part(
            tf.ones(
                [seq_len, seq_len]
            ),
            -1,
            0
        )

        # Convert:
        #
        # allowed -> 0
        # blocked -> 1

        return 1.0 - mask


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def call(
        self,
        input_ids,
        training=None
    ):

        # Sequence length
        seq_len = tf.shape(input_ids)[1]

        # Check sequence length
        tf.debugging.assert_less_equal(
            seq_len,
            self.max_length,
            message="Sequence length exceeds MAX_LENGTH"
        )

        # ----------------------------------------------------
        # Token Embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        # ----------------------------------------------------
        # Position Embeddings
        # ----------------------------------------------------

        positions = tf.range(
            start=0,
            limit=seq_len,
            delta=1
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Add Token + Position
        # ----------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.embedding_dropout(
            x,
            training=training
        )

        # ----------------------------------------------------
        # Causal Mask
        # ----------------------------------------------------

        mask = self.create_causal_mask(
            seq_len
        )

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        for block in self.transformer_blocks:

            x = block(
                x,
                mask=mask,
                training=training
            )

        # ----------------------------------------------------
        # Final Normalization
        # ----------------------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------------------
        # Weight Tying
        # ----------------------------------------------------
        #
        # Instead of creating:
        #
        # Dense(VOCAB_SIZE)
        #
        # we reuse the token embedding matrix.
        #
        # This saves ~38.6M parameters.
        #

        logits = tf.matmul(
            x,
            self.token_embedding.embeddings,
            transpose_b=True
        )

        return logits


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = AIWithThiru()


# ============================================================
# 7. BUILD MODEL
# ============================================================

dummy_input = tf.zeros(
    [1, MAX_LENGTH],
    dtype=tf.int32
)

dummy_output = model(
    dummy_input,
    training=False
)


# ============================================================
# 8. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 9. PARAMETER COUNT
# ============================================================

trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

print("\n" + "=" * 60)
print("MODEL NAME       : AI with Thiru")
print("VOCAB SIZE       :", VOCAB_SIZE)
print("CONTEXT LENGTH   :", MAX_LENGTH)
print("EMBED DIM        :", EMBED_DIM)
print("ATTENTION HEADS  :", NUM_HEADS)
print("TRANSFORMER      :", NUM_LAYERS)
print("FFN DIMENSION    :", DFF)
print("=" * 60)
print(
    f"TRAINABLE PARAMS : {trainable_params:,}"
)
print(
    f"PARAMETERS (M)   : {trainable_params / 1e6:.2f}M"
)
print("=" * 60)