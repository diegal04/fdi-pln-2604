"""Mecanismo de auto-atención multi-cabezal con RoPE.

El módulo contiene las piezas de bajo nivel del Transformer: proyección QKV,
máscara causal opcional, Rotary Position Embeddings y recombinación de cabezas.
"""

import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax


def _rotate_half(x):
    """Rota la mitad de las dimensiones de x para aplicar RoPE.

    Dada x de forma (..., d), parte la última dimensión en dos mitades
    (x1, x2) y devuelve (-x2, x1) concatenadas. Junto con cos/sin esto
    implementa una rotación 2D por pares de dimensiones.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _precompute_rope(head_dim, max_seq_len):
    """Precomputa las matrices cos y sin para RoPE.

    Usa las frecuencias θ_i = 1 / 10000^(2i / head_dim), que hacen que
    cada par de dimensiones gire a una frecuencia diferente según la
    posición. Las matrices devueltas tienen forma (max_seq_len, head_dim).
    """
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float)
    # freqs[i, j] = posicion_i * frecuencia_j  ->  (max_seq_len, head_dim/2)
    freqs = torch.outer(positions, inv_freq)
    # Duplicamos para cubrir las head_dim dimensiones completas
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


class Attention(nn.Module):
    """Auto-atención multi-cabezal con escala (scaled multi-head self-attention)

    Si `causal=True` en el forward, cada posicion solo atiende a las
    anteriores (util para generacion). Si `causal=False`, cada posicion
    atiende a toda la secuencia, que es lo que necesita la tarea NER.

    dropout es el porcentaje de dropout a usar.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible entre n_heads.")

        self.n_heads = n_heads
        # Distribuimos la dimensión del modelo entre el numero de cabezas
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requiere que d_model / n_heads sea par.")

        # Una única matriz para QKV, luego separaremos
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Capa lineal para permitir al modelo reproyectar los vectores contexto
        self.out = nn.Linear(d_model, d_model)
        # El dropout se activa en train y desactiva en test gracias a pytorch
        self.dropout = nn.Dropout(dropout)
        # La máscara causal pone a -inf las posiciones correspondientes a tokens
        # "futuros" (triangular superior)
        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
        )
        # Registramos la máscara causal como tensor (no entrenable)
        self.register_buffer("mask", mask)

        # RoPE: precomputamos las matrices cos y sin para todas las posiciones
        # posibles y las registramos como buffers (se mueven a GPU con .to())
        cos_cached, sin_cached = _precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x, causal=True):
        # Los tensores de pytorch tienen primero una dimensión batch
        # (entrenamiento más eficiente si hacemos varios a la vez)
        # luego tokens y luego ya la dimensión de los embeddings
        _batch_size, n_tokens, _d_model = x.shape
        if n_tokens > self.mask.size(0):
            raise ValueError(
                f"La secuencia tiene {n_tokens} tokens, pero el modelo se "
                f"creó con max_seq_len={self.mask.size(0)}."
            )

        # multiplicamos x por QKV (todo junto), pero separamos a lo largo de la
        # última dimensión para tener las matrices de queries, keys y values
        q, k, v = self.qkv(x).tensor_split(3, dim=-1)
        # separamos en cabezales (ver función más abajo)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # RoPE: rotamos Q y K con las frecuencias posicionales.
        # cos/sin tienen forma (max_seq_len, head_dim); recortamos a n_tokens
        # y añadimos dimensiones batch y heads para que el broadcasting funcione:
        # (1, 1, n_tokens, head_dim)
        cos = self.cos_cached[:n_tokens].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:n_tokens].unsqueeze(0).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        a = q @ k.transpose(-2, -1)
        a /= math.sqrt(self.head_dim)
        if causal:
            a += self.mask[:n_tokens, :n_tokens]
        a = softmax(a, dim=-1)

        a = self.dropout(a)
        z = a @ v

        # "deshacemos" la partición en cabezales
        # (batch_size, n_heads, n_tokens, head_dim) -> (batch_size, n_tokens, d_model)
        z = z.transpose(1, 2).flatten(-2)

        # re-proyectamos con la última transformación
        return self.out(z)

    def split_heads(self, x):
        # (batch_size, n_tokens, d_model) -> (batch_size, n_tokens, n_heads, head_dim)
        # "partimos" la última dimensión (-1, d_model) en n_heads x head_dim
        x = x.unflatten(-1, (self.n_heads, self.head_dim))
        # (batch_size, n_tokens, n_heads, head_dim) -> (batch_size, n_heads, n_tokens, head_dim)
        #  transponemos n_tokens y n_heads para que cada cabezal de atención se
        # "multiplique por separado", haciéndolos independientes
        return x.transpose(1, 2)
