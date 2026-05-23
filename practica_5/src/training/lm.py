"""Utilidades de entrenamiento para el modelo de lenguaje causal."""

import math
import time

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from loguru import logger
except ImportError:

    class _FallbackLogger:
        def info(self, message):
            print(message)

        def opt(self, **_kwargs):
            return self

    logger = _FallbackLogger()


class TextDataset(Dataset):
    """Ventana deslizante sobre un tensor de tokens para language modeling.

    Cada sample es un par (x, y) de longitud `seq_len`, donde y es x
    desplazado una posicion a la derecha (predecir el siguiente token).
    """

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _make_dataloaders(tokens, context_size, batch_size, train_ratio=0.9):
    """Los dataloaders se encargan de ir aportando pares para el entrenamiento,
    incluyendo batching, mezcla aleatoria, etc."""
    data = torch.tensor(tokens, dtype=torch.long)
    if len(data) <= context_size:
        raise ValueError(
            "El corpus tokenizado es demasiado corto para el context_size elegido."
        )

    # Separamos datos en entrenamiento y validación
    split = int(train_ratio * len(data))
    split = min(max(split, context_size + 1), len(data))
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)
    logger.info(f"Train: {len(train_ds):,} muestras, Val: {len(val_ds):,}")
    if len(train_ds) == 0:
        raise ValueError("No hay muestras de entrenamiento. Reduce context_size.")

    # Los dataloaders implementan utilidades para el entrenamiento de
    # modelos. Devolvemos uno para train y otro para val
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _make_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Crea un scheduler LambdaLR con warmup lineal y cosine decay.

    Durante los primeros warmup_steps pasos la lr sube linealmente de 0 a lr.
    Después decae siguiendo una curva coseno hasta min_lr_ratio * lr al llegar
    a total_steps (por defecto 10% de la lr máxima, en lugar de 0).
    El scheduler debe llamarse una vez por batch (no por época).
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Escalar para que el mínimo sea min_lr_ratio en lugar de 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _run_epoch(model, dataloader, optimizer=None, scheduler=None):
    """Ejecuta una epoch completa de entrenamiento o evaluación.

    Si se pasa optimizer, entrena el modelo (forward + backward + step).
    Si no, evalúa sin calcular gradientes.
    Si se pasa scheduler, lo avanza un paso tras cada actualización del optimizador.
    Devuelve la media de loss sobre todos los batches.
    """
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    is_training = optimizer is not None
    model.train(is_training)

    with torch.set_grad_enabled(is_training):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            if optimizer:
                optimizer.zero_grad()

            _, loss = model(x, y)

            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n += 1

    # Devolvemos la media de loss en este epoch
    if n == 0:
        return None
    return total_loss / n


def train(
    model,
    tokens,
    epochs=5,
    context_size=128,
    batch_size=64,
    lr=3e-4,
    train_ratio=0.9,
    warmup_steps=100,
    weight_decay=0.1,
):
    """Entrena el modelo de lenguaje causal sobre los tokens dados.

    Realiza `epochs` épocas de entrenamiento con AdamW y un scheduler
    warmup lineal + cosine decay, registrando train/val loss en cada época.
    Guarda internamente el mejor checkpoint según val_loss y lo restaura
    al finalizar (early stopping por mejor validación).
    """

    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)

    # El optimizador ajusta los parámetros que le pasamos en función del
    # gradiente (calculado con forward y backward) y la tasa de aprendizaje.
    # weight_decay penaliza pesos grandes para regularizar el modelo.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # El scheduler sube la lr linealmente durante warmup_steps pasos
    # y luego la baja siguiendo una curva coseno hasta 0.
    total_steps = len(train_dl) * epochs
    scheduler = _make_scheduler(optimizer, warmup_steps, total_steps)

    t0 = time.time()
    history = []
    best_val_loss = None
    best_state = None
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer, scheduler)
        val_loss = _run_epoch(model, val_dl, None, None)
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Early stopping: guardamos el estado del modelo cuando val_loss mejora
        if val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )
        val_msg = (
            f"val={val_loss:.4f}" if val_loss is not None else "val=sin_validacion"
        )
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"{val_msg} | lr={current_lr:.2e} | tiempo={elapsed:.1f}s"
        )

    # Restauramos el mejor checkpoint encontrado durante el entrenamiento
    if best_state is not None:
        model.load_state_dict(best_state)
        best_epoch = next(
            (row["epoch"] for row in history if row.get("val_loss") == best_val_loss),
            epochs,
        )
        logger.info(
            "Restaurado mejor checkpoint LM: "
            f"epoch={best_epoch} | val_loss={best_val_loss:.4f}"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")
    return history
