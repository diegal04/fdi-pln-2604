# Entrenamiento de LLM causal en base a un corpus
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import time

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Ventana deslizante sobre un tensor de tokens para language modeling.

    Cada sample es un par (x, y) de longitud `seq_len`, donde y es x
    desplazado una posicion a la derecha (predecir el siguiente token).
    """

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _make_dataloaders(tokens, context_size, batch_size, train_ratio=0.9):
    """Los dataloaders se encargan de ir aportando pares para el entrenamiento,
    incluyendo batching, mezcla aleatoria, etc."""
    data = torch.tensor(tokens, dtype=torch.long)

    # Separamos datos en entrenamiento y validación
    split = int(train_ratio * len(data))
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)
    logger.info(f"Train: {len(train_ds):,} muestras, Val: {len(val_ds):,}")

    # Los dataloaders implementan utilidades para el entrenamiento de
    # modelos. Devolvemos uno para train y otro para val
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _run_epoch(model, dataloader, optimizer=None):
    """Ejecuta una epoch completa de entrenamiento o evaluación.

    Si se pasa optimizer, entrena el modelo (forward + backward + step).
    Si no, evalúa sin calcular gradientes.
    Devuelve la media de loss sobre todos los batches.
    """
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    if optimizer:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        if optimizer:
            optimizer.zero_grad()

        # Pase forward, creando el grafo computacional y calculando loss
        _, loss = model(x, y)

        if optimizer:
            # Propaga la pérdida hacia atrás siguiendo el grafo
            loss.backward()
            # Reducimos "gradientes explosivos" para evitar anomalías de train
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Hacemos un paso del optimizador (eg un pequeño paso de descenso
            # siguiendo el gradiente, o lo que determine el optimizador)
            optimizer.step()

        total_loss += loss.item()
        n += 1

    # Devolvemos la media de loss en este epoch
    return total_loss / n


def train(
    model,
    tokens,
    epochs=5,
    context_size=128,
    batch_size=64,
    lr=3e-4,
    train_ratio=0.9,
):
    """Entrena el modelo de lenguaje causal sobre los tokens dados.

    Realiza `epochs` épocas de entrenamiento con AdamW, registrando train/val
    loss en cada época.
    """

    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)

    # El optimizador ajusta los parámetros que le pasamos en función del
    # gradiente (calculado con forward y backward) y la tasa de aprendizaje
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        elapsed = time.time() - t0
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")


if __name__ == "__main__":
    import sys

    from causalLLM import CausalLLM
    from corpus import load_corpus
    from tokenizer import BPETokenizer

    corpus = sys.argv[1] if len(sys.argv) > 1 else "resources"
    text = load_corpus(corpus)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = 300
    CONTEXT_SIZE = 128

    tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)
    tokens = tokenizer.encode(text)

    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=CONTEXT_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=4,
        expansion=4,
        dropout=0.1,
    ).to(device)

    train(model, tokens, epochs=5, context_size=CONTEXT_SIZE)

    prompt = "alice and the cat were studying for the exam. what "
    pred = model.generate(tokenizer.encode(prompt), max_tokens=200)
    logger.opt(colors=True).info(f"<cyan>{prompt}</cyan>{tokenizer.decode(pred)[:500]}")
