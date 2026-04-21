# Transformer básico comentado
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import torch
import torch.nn as nn

from attention import Attention


class FeedForward(nn.Module):
    """Red feed-forward ala perceptron con capa intermedia *más amplia*, y
    activación GELU.

    El factor de expansión permite a la red encontrar y procesar patrones en un
    espacio menos denso que d_model."""

    def __init__(self, d_model, expansion, dropout):
        super().__init__()
        hidden = expansion * d_model
        # Red fully-connected con una capa oculta
        self.up = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden, d_model)
        # Añadimos dropout para regularizar
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        model = nn.Sequential(self.up, self.act, 
                              self.down, self.dropout)
        ff = model(z)
        return ff


class Block(nn.Module):
    """Bloque principal del transformer a repetir.

    Incluye las dos cosas principales:
    1. Mecanismo de atención para atender al contexto, aprender matices y ambiguedades.
    2. Red feed-forward, para aprender a abstraer y generar las entradas de la siguiente capa.

    Se incluyen capas de normalización para regularizar el aprendizaje."""

    def __init__(self, max_seq_len, d_model, n_heads, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, expansion, dropout)

    def forward(self, x, causal=True):
        z = x + self.attn(self.norm1(x), causal)
        f2 = z + self.ff(self.norm2(x))
        return f2


class Transformer(nn.Module):
    """Backbone del transformer: embeddings, bloques de atención y normalización final.

    Produce representaciones contextuales (hidden states) para cada token de
    entrada. No realiza ninguna tarea concreta: sirve de base para distintos
    modelos añadiéndole una cabeza específica (generación, clasificación…).

    Parámetros:
      vocab_size   Tamaño del vocabulario
      max_seq_len  Longitud máxima de secuencia
      d_model      Dimensión interna de las representaciones
      n_heads      Número de cabezas de atención
      n_layers     Número de bloques transformer apilados
      expansion    Factor de expansión de la capa feed-forward (típico: 4)
      dropout      Tasa de dropout para regularización
    """

    def __init__(
        self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # El inicio del transformer son los embeddings. Tenemos dos, los de
        # vocabulario y los posicionales (se calculan en forward)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # Dropout para regularizar el aprendizaje de los embeddings
        self.drop = nn.Dropout(dropout)

        # El corazón del transformer es el bloque principal, con atención y
        # feedforward, que repetimos en secuencia varias veces
        self.blocks = nn.ModuleList([Block(max_seq_len, 
            d_model, n_heads, expansion, dropout) for _ in range(n_layers)])


        # Una última normalización final
        self.norm = nn.LayerNorm(d_model)

    def forward(self, idx, causal=True):
        """Devuelve los hidden states para cada token de idx.

        idx     Tensor (batch, n_tokens) con ids de tokens
        causal  Si True, la atención es causal (solo mira tokens anteriores)
        """
        _, n_tokens = idx.shape

        # Los tokens de vocabulario se entrenan, los posicionales se calculan
        # directamente (en GPU si estamos usando GPU)
        pos = self.pos_emb(torch.arange(n_tokens, device=idx.device))
        emb = self.tok_emb(idx)
        x = self.drop(emb + pos)
        for block in self.blocks:
            x = block(x, causal)
        return self.norm(x)

