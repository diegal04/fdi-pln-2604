"""Modelo NER, dataset y alineamiento palabra -> BPE.

El flujo de datos:

  palabras anotadas (word-level, BIO)
        |
        |  align_to_bpe()       <- asigna B-/I- a cada sub-token
        v
  sub-tokens + etiquetas BIO
        |
        |  NERDataset + collate_ner
        v
  batches (ids, labels) listos para cross_entropy
        |
        v
  NERLLM (Transformer + cabeza lineal por token)
"""

import html
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset, Subset

from transformer import Transformer

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
        ids, _ = align_to_bpe(words, ["o"] * len(words), tokenizer, add_spaces=add_spaces)
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
                        torch.tensor([LABEL2ID[l] for l in chunk_labs], dtype=torch.long),
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


def _label_counts(dataset):
    """Cuenta etiquetas reales, sin padding, en un Dataset o Subset."""
    counts = torch.zeros(NUM_LABELS, dtype=torch.long)
    for _, labels in dataset:
        valid = labels[labels != -100]
        counts += torch.bincount(valid, minlength=NUM_LABELS)
    return counts


def _make_loss_weights(entity_loss_weight=10.0):
    """Pesos de loss fijos: O vale 1 y cada etiqueta de entidad vale N."""
    weights = torch.ones(NUM_LABELS, dtype=torch.float)
    weights[1:] = entity_loss_weight
    return weights


def _metrics_from_confusion(confusion):
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    metrics = {
        "accuracy": correct / total if total else None,
        "entity_accuracy": None,
        "non_o_accuracy": None,
        "macro_entity_f1": None,
        "per_label": {},
    }

    non_o_total = confusion[1:, :].sum().item()
    non_o_correct = confusion[1:, 1:].diag().sum().item()
    if non_o_total:
        metrics["non_o_accuracy"] = non_o_correct / non_o_total
        # Alias mantenido para compatibilidad con historiales anteriores.
        metrics["entity_accuracy"] = metrics["non_o_accuracy"]

    entity_f1 = []
    for label_id, label in ID2LABEL.items():
        tp = confusion[label_id, label_id].item()
        predicted = confusion[:, label_id].sum().item()
        expected = confusion[label_id, :].sum().item()
        precision = tp / predicted if predicted else None
        recall = tp / expected if expected else None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        metrics["per_label"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": expected,
        }
        if label_id != LABEL2ID["o"] and f1 is not None:
            entity_f1.append(f1)

    if entity_f1:
        metrics["macro_entity_f1"] = sum(entity_f1) / len(entity_f1)
    return metrics


@torch.no_grad()
def _eval_ner(model, dataloader):
    """Loss y metricas de validacion NER en una sola pasada."""
    device = next(model.parameters()).device
    model.eval()
    total_loss, n = 0, 0
    confusion = torch.zeros(NUM_LABELS, NUM_LABELS, dtype=torch.long)
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        pred = logits.argmax(dim=-1)
        mask = y != -100
        true = y[mask].detach().cpu()
        guessed = pred[mask].detach().cpu()
        if len(true) > 0:
            pairs = true * NUM_LABELS + guessed
            confusion += torch.bincount(
                pairs,
                minlength=NUM_LABELS * NUM_LABELS,
            ).reshape(NUM_LABELS, NUM_LABELS)
        if loss is not None:
            total_loss += loss.item()
            n += 1

    metrics = _metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / n if n else None
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


@torch.no_grad()
def _eval_token_accuracy(model, dataloader):
    """Accuracy por token ignorando padding (-100)."""
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)
        mask = y != -100
        correct += ((pred == y) & mask).sum().item()
        total += mask.sum().item()
    if total == 0:
        return None
    return correct / total


def _save_loss_svg(history, path):
    width, height = 760, 420
    margin_left, margin_top, margin_bottom = 70, 35, 60
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom
    values = [
        value
        for row in history
        for value in (row.get("train_loss"), row.get("val_loss"))
        if value is not None
    ]
    max_loss = max(values) if values else 1.0
    max_loss = max(max_loss, 1e-8)

    def points(key):
        rows = [row for row in history if row.get(key) is not None]
        if len(rows) == 1:
            x = margin_left + plot_w
            y = margin_top + plot_h * (1 - rows[0][key] / max_loss)
            return f"{x:.1f},{y:.1f}"
        coords = []
        denom = max(len(rows) - 1, 1)
        for i, row in enumerate(rows):
            x = margin_left + plot_w * i / denom
            y = margin_top + plot_h * (1 - row[key] / max_loss)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">NER loss</text>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>
<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>
<text x="18" y="{margin_top + 15}" font-family="sans-serif" font-size="12">{max_loss:.3f}</text>
<text x="24" y="{margin_top + plot_h}" font-family="sans-serif" font-size="12">0</text>
<polyline points="{points("train_loss")}" fill="none" stroke="#1f77b4" stroke-width="3"/>
<polyline points="{points("val_loss")}" fill="none" stroke="#d62728" stroke-width="3"/>
<text x="{margin_left}" y="{height - 22}" font-family="sans-serif" font-size="13" fill="#1f77b4">train_loss</text>
<text x="{margin_left + 120}" y="{height - 22}" font-family="sans-serif" font-size="13" fill="#d62728">val_loss</text>
<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="13">epoch</text>
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


def _save_non_o_accuracy_svg(history, path):
    width, height = 760, 420
    margin_left, margin_top, margin_bottom = 70, 35, 60
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom

    def points(key):
        rows = [row for row in history if row.get(key) is not None]
        if len(rows) == 1:
            x = margin_left + plot_w
            y = margin_top + plot_h * (1 - rows[0][key])
            return f"{x:.1f},{y:.1f}"
        coords = []
        denom = max(len(rows) - 1, 1)
        for i, row in enumerate(rows):
            x = margin_left + plot_w * i / denom
            y = margin_top + plot_h * (1 - row[key])
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">NER non-O accuracy</text>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>
<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>
<text x="18" y="{margin_top + 15}" font-family="sans-serif" font-size="12">1.0</text>
<text x="24" y="{margin_top + plot_h}" font-family="sans-serif" font-size="12">0</text>
<polyline points="{points("train_non_o_accuracy")}" fill="none" stroke="#1f77b4" stroke-width="3"/>
<polyline points="{points("val_non_o_accuracy")}" fill="none" stroke="#d62728" stroke-width="3"/>
<text x="{margin_left}" y="{height - 22}" font-family="sans-serif" font-size="13" fill="#1f77b4">train_non_o_acc</text>
<text x="{margin_left + 155}" y="{height - 22}" font-family="sans-serif" font-size="13" fill="#d62728">val_non_o_acc</text>
<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="13">epoch</text>
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


def _save_confusion_csv(confusion, path):
    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    lines = ["," + ",".join(f"pred_{label}" for label in labels)]
    for label_id, label in enumerate(labels):
        row = [str(v) for v in confusion[label_id]]
        lines.append(f"true_{label}," + ",".join(row))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_confusion_svg(confusion, path):
    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    cell, left, top = 72, 95, 80
    width = left + cell * NUM_LABELS + 40
    height = top + cell * NUM_LABELS + 70
    max_value = max(max(row) for row in confusion) if confusion else 1
    max_value = max(max_value, 1)
    cells = []
    for i, row in enumerate(confusion):
        for j, value in enumerate(row):
            intensity = int(245 - 190 * (value / max_value))
            fill = f"rgb({intensity},{intensity},255)"
            x = left + j * cell
            y = top + i * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#444"/>'
            )
            cells.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" '
                f'text-anchor="middle" font-family="sans-serif" '
                f'font-size="15">{value}</text>'
            )

    headers = []
    for i, label in enumerate(labels):
        safe = html.escape(label)
        headers.append(
            f'<text x="{left + i * cell + cell / 2}" y="{top - 18}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13">{safe}</text>'
        )
        headers.append(
            f'<text x="{left - 18}" y="{top + i * cell + cell / 2 + 5}" '
            f'text-anchor="end" font-family="sans-serif" font-size="13">{safe}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">NER confusion matrix</text>
<text x="{left + cell * NUM_LABELS / 2}" y="54" text-anchor="middle" font-family="sans-serif" font-size="13">predicho</text>
<text x="18" y="{top + cell * NUM_LABELS / 2}" transform="rotate(-90 18 {top + cell * NUM_LABELS / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">real</text>
{''.join(headers)}
{''.join(cells)}
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


def _save_ner_artifacts(history, confusion, class_weights, metrics_dir):
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    _save_loss_svg(history, metrics_dir / "loss.svg")
    _save_non_o_accuracy_svg(history, metrics_dir / "non_o_accuracy.svg")
    _save_confusion_csv(confusion, metrics_dir / "confusion_matrix.csv")
    _save_confusion_svg(confusion, metrics_dir / "confusion_matrix.svg")
    (metrics_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metrics_dir / "class_weights.json").write_text(
        json.dumps(
            {label: float(class_weights[i]) for i, label in enumerate(labels)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _fmt_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def _selection_score(val_accuracy, val_non_o_accuracy, accuracy_floor, non_o_weight):
    if val_accuracy is None or val_non_o_accuracy is None:
        return None
    if val_accuracy < accuracy_floor:
        return -1.0
    return val_accuracy + non_o_weight * val_non_o_accuracy


def _copy_state_dict_to_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_ner(
    model,
    ner_data,
    tokenizer,
    epochs=5,
    batch_size=32,
    lr=3e-4,
    max_len=128,
    train_ratio=0.9,
    add_spaces=False,
    entity_loss_weight=10.0,
    selection_accuracy_floor=0.8,
    selection_non_o_weight=1.5,
    metrics_dir=None,
):
    """Fine-tuning NER sobre datos (tokens, labels)."""
    from train import _run_epoch, logger

    dataset = NERDataset(ner_data, tokenizer, max_len=max_len, add_spaces=add_spaces)
    if len(dataset) == 0:
        raise ValueError("No hay muestras NER tras alinear merged.json con el BPE.")

    split = int(train_ratio * len(dataset))
    split = min(max(split, 1), len(dataset))
    train_ds = Subset(dataset, range(split))
    val_ds = Subset(dataset, range(split, len(dataset)))
    label_counts = _label_counts(train_ds)
    class_weights = _make_loss_weights(entity_loss_weight=entity_loss_weight)
    model.set_loss_weights(class_weights)
    logger.info(
        "Conteo etiquetas NER: "
        + ", ".join(
            f"{ID2LABEL[i]}={int(count)}" for i, count in enumerate(label_counts)
        )
    )
    logger.info(
        "Pesos de loss NER: "
        + ", ".join(
            f"{ID2LABEL[i]}={float(weight):.2f}"
            for i, weight in enumerate(class_weights)
        )
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ner,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_ner)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    history = []
    last_confusion = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    best_score = None
    best_state = None
    best_confusion = last_confusion
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        train_metrics = _eval_ner(model, train_dl)
        train_accuracy = train_metrics["accuracy"]
        train_non_o_accuracy = train_metrics["non_o_accuracy"]
        train_macro_entity_f1 = train_metrics["macro_entity_f1"]
        train_per_label = train_metrics["per_label"]
        if len(val_ds) > 0:
            val_metrics = _eval_ner(model, val_dl)
            val_loss = val_metrics["loss"]
            val_accuracy = val_metrics["accuracy"]
            val_non_o_accuracy = val_metrics["non_o_accuracy"]
            val_macro_entity_f1 = val_metrics["macro_entity_f1"]
            val_per_label = val_metrics["per_label"]
            last_confusion = val_metrics["confusion_matrix"]
            val_msg = (
                f" | val={_fmt_metric(val_loss)} | acc={_fmt_metric(val_accuracy)} "
                f"| val_non_o_acc={_fmt_metric(val_non_o_accuracy)} "
                f"| val_ent_f1={_fmt_metric(val_macro_entity_f1)}"
            )
        else:
            val_loss = None
            val_accuracy = None
            val_non_o_accuracy = None
            val_macro_entity_f1 = None
            val_per_label = {}
            val_msg = ""
        elapsed = time.time() - t0
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_non_o_accuracy": train_non_o_accuracy,
                "train_entity_accuracy": train_non_o_accuracy,
                "train_macro_entity_f1": train_macro_entity_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_non_o_accuracy": val_non_o_accuracy,
                "val_entity_accuracy": val_non_o_accuracy,
                "val_macro_entity_f1": val_macro_entity_f1,
                "train_per_label": train_per_label,
                "val_per_label": val_per_label,
                "per_label": val_per_label,
            }
        )
        row = history[-1]
        row["selection_score"] = _selection_score(
            val_accuracy,
            val_non_o_accuracy,
            accuracy_floor=selection_accuracy_floor,
            non_o_weight=selection_non_o_weight,
        )
        row["selection_formula"] = (
            f"val_accuracy + {selection_non_o_weight} * val_non_o_accuracy"
        )
        row["selection_accuracy_floor"] = selection_accuracy_floor
        row["is_best"] = False
        if row["selection_score"] is not None and row["selection_score"] >= 0:
            if best_score is None or row["selection_score"] > best_score:
                for previous in history:
                    previous["is_best"] = False
                row["is_best"] = True
                best_score = row["selection_score"]
                best_state = _copy_state_dict_to_cpu(model)
                best_confusion = last_confusion
        if metrics_dir is not None:
            _save_ner_artifacts(history, last_confusion, class_weights, metrics_dir)
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f}"
            f" | train_non_o_acc={_fmt_metric(train_non_o_accuracy)}"
            f" | score={_fmt_metric(row['selection_score'])}"
            f"{val_msg} | tiempo={elapsed:.1f}s"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        if metrics_dir is not None:
            _save_ner_artifacts(history, best_confusion, class_weights, metrics_dir)
        best_epoch = next(row["epoch"] for row in history if row.get("is_best"))
        logger.info(
            f"Restaurado mejor checkpoint NER: epoch={best_epoch} "
            f"| score={best_score:.4f}"
        )
    return history
