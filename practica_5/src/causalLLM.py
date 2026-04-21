# LLM causal para generación de texto
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax

from transformer import Transformer


class CausalLLM(Transformer):
    """Modelo de lenguaje causal para generación de texto.

    Extiende Transformer añadiendo una cabeza lineal que proyecta los hidden
    states al vocabulario para predecir el siguiente token.

    Usa weight tying: los pesos del embedding de entrada y de la cabeza de
    salida son los mismos (la proyección inversa), lo que mejora la
    generalización y reduce el número de parámetros.
    """

    def __init__(
        self, vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
    ):
        super().__init__(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        # Proyectamos el espacio d_model al vocabulario para predecir el
        # siguiente token
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Por eficiencia, hacemos que los pesos de convertir vocab a d_model
        # sean los mismos que la inversa (linear multiplica por la traspuesta)
        # (weight tying)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        """Devuelve (logits, loss).

        idx      Tensor (batch, n_tokens) con ids de tokens de entrada
        targets  Tensor (batch, n_tokens) con ids objetivo; si se pasa,
                 calcula el loss de language modeling
        """
        x = super().forward(idx, causal=True)

        # Calculamos los logits para cada elemento del vocabulario
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        # Si tenemos secuencia objetivo, calculamos el error para poder entrenar
        # "Aplanamos" el batch, calculando el loss eficientemente en todo el batch de una vez
        predicted = logits.flatten(0, 1)
        loss = cross_entropy(predicted, targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_tokens=200, temperature=0.8):
        """Genera tokens a partir de un prompt (lista de ids).

        Usa sampling probabilístico para aumentar la "creatividad".

        prompt       Lista de token ids de entrada.
        max_tokens   Número máximo de tokens a generar.
        temperature  Modula lo "puntiaguda" (determinista) que es la
                     distribución de sampling.

        Devuelve la lista de token ids generados (sin el prompt).

        TAREA: ¿Cómo implementarías top-k?
        TAREA: ¿Cómo implementarías restringir la generación a una estructura,
        por ejemplo una gramática de json?
        """

        # Ponemos el modelo en modo eval
        self.eval()

        # Preparamos una ventana deslizante (lo que se suele llamar contexto)
        # a partir del prompt, y preparamos el tensor con la dimensión de batch
        # (corchete exterior, batch=1) y el máximo de tokens que caben
        ventana = torch.tensor(
            [prompt[-self.max_seq_len :]],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        generados = []
        for _ in range(max_tokens):
            # Calculamos los logits del posible próximo token
            logits, _ = self(ventana)
            next_token_logits = logits[:, -1, :]
            # Convertimos en una distribución de probabilidad sobre el vocab.
            # Al dividir por la temperatura en el exponente de la exponencial,
            # modulamos lo "puntiaguda" (determinista) que es la distribución
            next_token_probs = softmax(next_token_logits / temperature, dim=-1)
            # sampleamos un único token
            next_token_id = torch.multinomial(next_token_probs, 1)

            # Guardamos el token y deslizamos la ventana
            generados.append(next_token_id.item())
            ventana = torch.cat([ventana, next_token_id], dim=1)[:, -self.max_seq_len :]

        return generados
