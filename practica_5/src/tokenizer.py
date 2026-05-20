# Tokenizador BPE (Byte Pair Encoding) mínimo, entrenado sobre el texto."""
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>


import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """Byte Pair Encoding entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes mas frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.

    NOTA: para ser BPE de verdad, tendríamos que hacerlo sobre bytes, no sobre
    caracteres, pero para la práctica funciona bien.
    """

    def __init__(self, text, vocab_size=300):
        self.vocab_size = vocab_size
        # Inicializamos con caracteres encontrados en el texto
        self.vocab = sorted(set(text))  # vocab[id] -> token string.
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}

        tokens = [self.tok2id[c] for c in text]
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode()

        for new_id in range(len(self.vocab), vocab_size):
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            new_tok = self.vocab[best[0]] + self.vocab[best[1]]
            self.tok2id[new_tok] = new_id
            self.vocab.append(new_tok)
            self.merges.append((best, new_id))

            tokens = self._apply_merge(tokens, best[0], best[1], new_id)
        self.vocab_size = len(self.vocab)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id):
        """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
        merged = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        caracteres = [self.vocab[id_] for id_ in ids]
        return caracteres

    def save(self, path):
        """Guarda vocabulario y merges aprendidos."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": [
                [[a, b], new_id]
                for (a, b), new_id in self.merges
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Carga un tokenizador previamente guardado."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls.__new__(cls)
        tokenizer.vocab = list(data["vocab"])
        tokenizer.vocab_size = len(tokenizer.vocab)
        tokenizer.tok2id = {tok: i for i, tok in enumerate(tokenizer.vocab)}
        tokenizer.merges = [
            ((int(pair[0]), int(pair[1])), int(new_id))
            for pair, new_id in data["merges"]
        ]
        return tokenizer

    def __repr__(self):
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        joined = "', '".join(pretty)
        return f"{len(self.vocab)} tokens: ['{joined}']"


# Si ejecutamos este módulo directamente, probamos el tokenizador
if __name__ == "__main__":
    import sys
    from pathlib import Path

    files_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resources")
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    textos = "\n\n".join(open(p).read() for p in files_path.glob("*.txt"))
    tokenizer = BPETokenizer(textos, vocab_size=vocab_size)
    print(tokenizer)
