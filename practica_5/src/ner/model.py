"""Modelo NER, dataset y alineamiento palabra -> BPE."""

import re

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset

from models.transformer import Transformer

# Etiquetas NER del corpus anotado
LABEL2ID = {"o": 0, "pi": 1, "pc": 2, "li": 3, "lc": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def split_text_tokens(text):
    """Tokeniza texto como el corpus anotado: palabras, espacios y signos."""
    return re.findall(r"\w+|\s+|[^\w\s]", text, flags=re.UNICODE)


def align_to_bpe(words, word_labels, tokenizer, add_spaces=True):
    """Alinea etiquetas de palabras/tokens a los sub-tokens de BPE.

    Como el tokenizador BPE puede partir una palabra en varios sub-tokens,
    hay que decidir que etiqueta dar a cada trozo. Regla: pi/li se queda
    en el primer sub-token y los siguientes son pc/lc.

      palabra 'alice' con etiqueta pi, BPE la parte en ['al', 'ice']
         -> al: pi, ice: pc
      palabra 'wonderland' con li, BPE en ['won', 'der', 'land']
         -> won: li, der: lc, land: lc
      palabras O -> todos sus sub-tokens O
      espacios entre palabras -> o, si add_spaces=True

    Devuelve (token_ids, token_labels) con etiquetas como strings. Para
    merged.json hay que usar add_spaces=False porque ya contiene espacios y
    saltos de linea como tokens anotados.
    """
    token_ids = []
    token_labels = []
    space_ids = tokenizer.encode(" ")
    for i, (word, label) in enumerate(zip(words, word_labels)):
        label = label.lower()
        if add_spaces and i > 0:
            token_ids.extend(space_ids)
            token_labels.extend(["o"] * len(space_ids))
        word_ids = tokenizer.encode(word)
        token_ids.extend(word_ids)
        if label == "pi":
            token_labels.append(label)
            token_labels.extend(["pc"] * (len(word_ids) - 1))
        elif label == "li":
            token_labels.append(label)
            token_labels.extend(["lc"] * (len(word_ids) - 1))
        else:
            token_labels.extend([label] * len(word_ids))
    return token_ids, token_labels


def explain_alignment(words, word_labels, tokenizer):
    """Imprime el alineamiento palabra -> sub-tokens BPE para una frase.

    Util para ver como el tokenizador parte cada palabra y donde aterriza
    cada etiqueta BIO: la B- se queda en el primer sub-token, el resto son I-.
    """
    print(f"  frase: {' '.join(words)}")
    for word, label in zip(words, word_labels):
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        if label == "pi":
            labs = [label] + ["pc"] * (len(ids) - 1)
        elif label == "li":
            labs = [label] + ["lc"] * (len(ids) - 1)
        else:
            labs = [label] * len(ids)
        pairs = "  ".join(f"{p}/{l}" for p, l in zip(pieces, labs))
        print(f"    {word:<15} {label:<6} -> {pairs}")


class NERLLM(Transformer):
    """Transformer con cabeza de clasificación por token para NER.

    Extiende Transformer añadiendo una cabeza lineal que asigna una etiqueta
    BIO a cada token. Usa atención bidireccional (causal=False): para etiquetar
    un token podemos mirar el contexto a derecha e izquierda.

    Los pesos del backbone se deben inicializar desde un CausalLLM pre-entrenado
    con load_state_dict(strict=False), que ignora las diferencias en las cabezas
    (lm_head vs ner_head) y transfiere solo el backbone compartido.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        expansion,
        dropout,
        num_labels,
    ):
        super().__init__(
            vocab_size, max_seq_len, d_model, n_heads, n_layers, expansion, dropout
        )
        # El transformer ya tiene una representación suficientemente rica,
        # no tenemos más que proyectarla al espacio de etiquetas
        self.ner_head = nn.Linear(d_model, num_labels)
        self.register_buffer("loss_weights", torch.ones(num_labels), persistent=False)

    def set_loss_weights(self, weights):
        """Configura pesos por clase para cross_entropy."""
        self.loss_weights = weights.to(next(self.parameters()).device)

    def forward(self, input_ids, labels=None):
        hidden = super().forward(input_ids, causal=False)
        logits = self.ner_head(hidden)
        loss = None
        if labels is not None:
            # cross_entropy espera logits 2D: para cada elemento, una
            # probabilidad por etiqueta.
            # Aplanamos batch y secuencia y tratamos cada token como una muestra
            # independiente:
            #   logits  (n_batches, n_tokens, num_labels) -> (n_batches*n_tokens, num_labels)
            #   labels  (n_batches, n_tokens)             -> (n_batches*n_tokens,)
            # Las posiciones de padding llevan -100 e ignore_index las descarta.
            flat_logits = logits.flatten(0, 1)
            flat_labels = labels.flatten()
            loss = cross_entropy(
                flat_logits,
                flat_labels,
                weight=self.loss_weights,
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def predict_entities(self, words, tokenizer, add_spaces=True):
        """Predice etiquetas BIO sobre una lista de **palabras**.

        Codifica la frase con `align_to_bpe` (etiquetas ficticias O), corre el
        modelo y agrupa sub-tokens B-X / I-X consecutivos en entidades.

        Devuelve las entidades nombradas ya compuestas [(texto, tipo), ...].
        """
        self.eval()
        ids, _ = align_to_bpe(
            words, ["o"] * len(words), tokenizer, add_spaces=add_spaces
        )
        device = next(self.parameters()).device
        pred_labels = []
        for start in range(0, len(ids), self.max_seq_len):
            chunk = ids[start : start + self.max_seq_len]
            logits, _ = self(torch.tensor([chunk], device=device))
            pred_labels.extend(ID2LABEL[p] for p in logits.argmax(-1)[0].tolist())

        entities = []
        i = 0
        while i < len(ids):
            if pred_labels[i] in ("pi", "li"):
                kind = "PER" if pred_labels[i].startswith("p") else "LOC"
                j = i + 1
                cont = "pc" if kind == "PER" else "lc"
                while j < len(ids):
                    piece = "".join(tokenizer.decode([ids[j]]))
                    if pred_labels[j] == cont:
                        j += 1
                    elif (
                        piece.isspace()
                        and j + 1 < len(ids)
                        and pred_labels[j + 1] == cont
                    ):
                        j += 1
                    else:
                        break
                text = "".join(tokenizer.decode(ids[i:j])).strip()
                if text:
                    entities.append((text, kind))
                i = j
            else:
                i += 1
        return entities


class NERDataset(Dataset):
    """Dataset de NER: aplica `align_to_bpe` a cada frase y convierte a tensores.

    `ner_data` es una lista de pares (words, labels), donde words es la lista
    de palabras de una frase y labels las etiquetas BIO alineadas. Es el
    formato que se carga desde merged.json en la CLI.
    """

    def __init__(self, ner_data, tokenizer, max_len=128, add_spaces=True):
        self.samples = []
        for words, labels in ner_data:
            ids, labs = align_to_bpe(words, labels, tokenizer, add_spaces=add_spaces)
            for start in range(0, len(ids), max_len):
                chunk_ids = ids[start : start + max_len]
                chunk_labs = labs[start : start + max_len]
                self.samples.append(
                    (
                        torch.tensor(chunk_ids, dtype=torch.long),
                        torch.tensor(
                            [LABEL2ID[l] for l in chunk_labs], dtype=torch.long
                        ),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_ner(batch):
    """Padding al largo maximo del batch. Las posiciones de padding usan -100
    en las etiquetas para que cross_entropy las ignore (no son tokens reales)."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : len(x)] = x
        padded_y[i, : len(y)] = y
    return padded_x, padded_y
