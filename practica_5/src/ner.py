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
from torch.utils.data import DataLoader, Dataset

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


def _word_label_counts(labels):
    """Cuenta etiquetas word-level de una frase anotada."""
    counts = torch.zeros(NUM_LABELS, dtype=torch.float)
    for label in labels:
        label = label.lower()
        if label in LABEL2ID:
            counts[LABEL2ID[label]] += 1
    return counts


def _stratified_phrase_split(ner_data, train_ratio=0.85):
    """Divide NER por frases, intentando preservar etiquetas en validacion.

    No partimos el dataset tras BPE porque eso puede meter chunks de la misma
    frase en train y val. Elegimos frases completas para validacion con un
    greedy sencillo que aproxima la proporcion global de etiquetas, dando mas
    importancia a las clases no-O.
    """
    n_items = len(ner_data)
    if n_items <= 1:
        return ner_data, []

    n_train = int(train_ratio * n_items)
    n_train = min(max(n_train, 1), n_items - 1)
    n_val = n_items - n_train

    item_counts = [_word_label_counts(labels) for _, labels in ner_data]
    total_counts = sum(item_counts, torch.zeros(NUM_LABELS, dtype=torch.float))
    target_val_counts = total_counts * (n_val / n_items)
    weights = torch.ones(NUM_LABELS, dtype=torch.float)
    weights[LABEL2ID["o"]] = 0.05

    def distance(counts):
        scale = torch.clamp(target_val_counts, min=1.0)
        return (((counts - target_val_counts) / scale) ** 2 * weights).sum().item()

    val_indices = set()
    val_counts = torch.zeros(NUM_LABELS, dtype=torch.float)
    remaining = set(range(n_items))
    while len(val_indices) < n_val:
        deficits = target_val_counts - val_counts
        wanted_labels = [
            label_id
            for label_id in range(1, NUM_LABELS)
            if deficits[label_id].item() > 0
        ]
        candidates = []
        if wanted_labels:
            label_id = max(
                wanted_labels,
                key=lambda i: deficits[i].item()
                / max(target_val_counts[i].item(), 1.0),
            )
            candidates = [
                idx for idx in remaining if item_counts[idx][label_id].item() > 0
            ]
        if not candidates:
            candidates = list(remaining)

        chosen = min(
            candidates,
            key=lambda idx: (
                distance(val_counts + item_counts[idx]),
                idx,
            ),
        )
        val_indices.add(chosen)
        val_counts += item_counts[chosen]
        remaining.remove(chosen)

    train_data = [item for i, item in enumerate(ner_data) if i not in val_indices]
    val_data = [item for i, item in enumerate(ner_data) if i in val_indices]
    return train_data, val_data


def _make_loss_weights(entity_loss_weight=10.0, location_weight_multiplier=1.0, continuation_weight_multiplier=1.0):
    """Pesos de loss por clase.

    - o = 1
    - pi (B-PER) = entity_loss_weight
    - pc (I-PER) = entity_loss_weight * continuation_weight_multiplier
    - li (B-LOC) = entity_loss_weight * location_weight_multiplier
    - lc (I-LOC) = entity_loss_weight * location_weight_multiplier * continuation_weight_multiplier

    Subir continuation_weight_multiplier ayuda cuando el modelo corta las
    entidades demasiado pronto (clasifica tokens I-X como O).
    """
    weights = torch.ones(NUM_LABELS, dtype=torch.float)
    weights[1] = entity_loss_weight                                                   # pi
    weights[2] = entity_loss_weight * continuation_weight_multiplier                  # pc
    weights[3] = entity_loss_weight * location_weight_multiplier                      # li
    weights[4] = entity_loss_weight * location_weight_multiplier * continuation_weight_multiplier  # lc
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


def _format_chart_value(value, as_percent=False):
    if as_percent:
        return f"{100 * value:.0f}%"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _chart_ticks(start, end, n_ticks=5):
    if end <= start:
        return [start]
    return [start + (end - start) * i / n_ticks for i in range(n_ticks + 1)]


def _line_chart_svg(
    history,
    path,
    title,
    y_label,
    series,
    y_min=0.0,
    y_max=None,
    as_percent=False,
):
    width, height = 980, 560
    left, right, top, bottom = 92, 42, 92, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [
        row.get(key)
        for row in history
        for key, _, _ in series
        if row.get(key) is not None
    ]
    if not values:
        y_max = 1.0 if y_max is None else y_max
    elif y_max is None:
        y_max = max(values) * 1.08
    y_max = max(y_max, y_min + 1e-8)
    max_epoch = max((row.get("epoch", i + 1) for i, row in enumerate(history)), default=1)

    def x_pos(epoch):
        if max_epoch <= 1:
            return left + plot_w / 2
        return left + (epoch - 1) * plot_w / (max_epoch - 1)

    def y_pos(value):
        return top + plot_h * (1 - (value - y_min) / (y_max - y_min))

    y_grid = []
    for value in _chart_ticks(y_min, y_max):
        y = y_pos(value)
        label = _format_chart_value(value, as_percent=as_percent)
        y_grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'class="grid"/>'
        )
        y_grid.append(
            f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{label}</text>'
        )

    x_ticks = sorted(
        {
            1,
            max_epoch,
            *(
                round(1 + (max_epoch - 1) * i / 4)
                for i in range(1, 4)
                if max_epoch > 4
            ),
        }
    )
    x_grid = []
    for epoch in x_ticks:
        x = x_pos(epoch)
        x_grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
            f'class="grid soft"/>'
        )
        x_grid.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" '
            f'class="tick">{epoch}</text>'
        )

    lines = []
    legend = []
    for idx, (key, label, color) in enumerate(series):
        points = []
        rows = []
        for i, row in enumerate(history):
            value = row.get(key)
            if value is None:
                continue
            epoch = row.get("epoch", i + 1)
            points.append(f"{x_pos(epoch):.1f},{y_pos(value):.1f}")
            rows.append((epoch, value))
        if not points:
            continue
        safe_label = html.escape(label)
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3.2" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
        if rows:
            epoch, value = rows[-1]
            lines.append(
                f'<circle cx="{x_pos(epoch):.1f}" cy="{y_pos(value):.1f}" r="4.5" '
                f'fill="white" stroke="{color}" stroke-width="2.5"/>'
            )
        lx = left + 18 + idx * 190
        ly = 58
        legend.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
        )
        legend.append(
            f'<text x="{lx + 36}" y="{ly + 5}" class="legend">{safe_label}</text>'
        )

    safe_title = html.escape(title)
    safe_y_label = html.escape(y_label)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 24px Arial, sans-serif; fill: #111827; }}
  .label {{ font: 600 14px Arial, sans-serif; fill: #374151; }}
  .tick {{ font: 12px Arial, sans-serif; fill: #6b7280; }}
  .legend {{ font: 13px Arial, sans-serif; fill: #374151; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .grid.soft {{ stroke: #eef2f7; }}
  .axis {{ stroke: #111827; stroke-width: 1.4; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="34" text-anchor="middle" class="title">{safe_title}</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#d1d5db"/>
{''.join(y_grid)}
{''.join(x_grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
{''.join(lines)}
{''.join(legend)}
<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" class="label">epoca</text>
<text x="28" y="{top + plot_h / 2}" transform="rotate(-90 28 {top + plot_h / 2})" text-anchor="middle" class="label">{safe_y_label}</text>
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


def _save_loss_svg(history, path):
    _line_chart_svg(
        history,
        path,
        title="NER loss por epoca",
        y_label="loss",
        series=[
            ("train_loss", "train loss", "#2563eb"),
            ("val_loss", "validation loss", "#dc2626"),
        ],
    )


def _save_non_o_accuracy_svg(history, path):
    _line_chart_svg(
        history,
        path,
        title="Accuracy sobre clases de entidad",
        y_label="accuracy",
        series=[
            ("train_non_o_accuracy", "train non-O", "#2563eb"),
            ("val_non_o_accuracy", "validation non-O", "#dc2626"),
        ],
        y_max=1.0,
        as_percent=True,
    )


def _save_accuracy_svg(history, path):
    _line_chart_svg(
        history,
        path,
        title="Accuracy total y non-O",
        y_label="accuracy",
        series=[
            ("train_accuracy", "train total", "#2563eb"),
            ("val_accuracy", "validation total", "#dc2626"),
            ("train_non_o_accuracy", "train non-O", "#059669"),
            ("val_non_o_accuracy", "validation non-O", "#f97316"),
        ],
        y_max=1.0,
        as_percent=True,
    )


def _save_confusion_csv(confusion, path):
    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    lines = ["," + ",".join(f"pred_{label}" for label in labels)]
    for label_id, label in enumerate(labels):
        row = [str(v) for v in confusion[label_id]]
        lines.append(f"true_{label}," + ",".join(row))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _mix_hex(start, end, ratio):
    ratio = min(max(ratio, 0.0), 1.0)
    a = _hex_to_rgb(start)
    b = _hex_to_rgb(end)
    mixed = tuple(round(a[i] + (b[i] - a[i]) * ratio) for i in range(3))
    return "#" + "".join(f"{value:02x}" for value in mixed)


def _save_confusion_svg(confusion, path):
    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    cell, left, top = 104, 148, 132
    right_pad, bottom_pad = 150, 118
    width = left + cell * NUM_LABELS + right_pad
    height = top + cell * NUM_LABELS + bottom_pad
    cells = []
    for i, row in enumerate(confusion):
        row_total = sum(row)
        recall = row[i] / row_total if row_total else 0
        for j, value in enumerate(row):
            ratio = value / row_total if row_total else 0
            fill = _mix_hex("#ffffff", "#16a34a" if i == j else "#f97316", ratio)
            text_color = "#ffffff" if ratio >= 0.42 else "#111827"
            pct = f"{100 * ratio:.1f}%"
            x = left + j * cell
            y = top + i * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'rx="6" fill="{fill}" stroke="#d1d5db"/>'
            )
            cells.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 - 4}" '
                f'text-anchor="middle" class="cell-main" fill="{text_color}">{value}</text>'
            )
            cells.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 20}" '
                f'text-anchor="middle" class="cell-sub" fill="{text_color}">{pct}</text>'
            )
        y_mid = top + i * cell + cell / 2
        cells.append(
            f'<text x="{left + NUM_LABELS * cell + 26}" y="{y_mid - 6}" '
            f'class="metric">recall</text>'
        )
        cells.append(
            f'<text x="{left + NUM_LABELS * cell + 26}" y="{y_mid + 18}" '
            f'class="metric strong">{100 * recall:.1f}%</text>'
        )

    headers = []
    for i, label in enumerate(labels):
        safe = html.escape(label)
        headers.append(
            f'<text x="{left + i * cell + cell / 2}" y="{top - 24}" '
            f'text-anchor="middle" class="axis-label">{safe}</text>'
        )
        row_total = sum(confusion[i])
        headers.append(
            f'<text x="{left - 26}" y="{top + i * cell + cell / 2 - 2}" '
            f'text-anchor="end" class="axis-label">{safe}</text>'
        )
        headers.append(
            f'<text x="{left - 26}" y="{top + i * cell + cell / 2 + 20}" '
            f'text-anchor="end" class="support">n={row_total}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 25px Arial, sans-serif; fill: #111827; }}
  .subtitle {{ font: 13px Arial, sans-serif; fill: #6b7280; }}
  .axis-title {{ font: 700 14px Arial, sans-serif; fill: #374151; }}
  .axis-label {{ font: 700 16px Arial, sans-serif; fill: #111827; }}
  .support {{ font: 12px Arial, sans-serif; fill: #6b7280; }}
  .cell-main {{ font: 700 21px Arial, sans-serif; }}
  .cell-sub {{ font: 12px Arial, sans-serif; opacity: 0.9; }}
  .metric {{ font: 12px Arial, sans-serif; fill: #6b7280; }}
  .metric.strong {{ font: 700 14px Arial, sans-serif; fill: #111827; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="34" text-anchor="middle" class="title">Matriz de confusion NER</text>
<text x="{width / 2}" y="58" text-anchor="middle" class="subtitle">Cada celda muestra conteo y porcentaje dentro de la etiqueta real</text>
<text x="{left + cell * NUM_LABELS / 2}" y="92" text-anchor="middle" class="axis-title">etiqueta predicha</text>
<text x="32" y="{top + cell * NUM_LABELS / 2}" transform="rotate(-90 32 {top + cell * NUM_LABELS / 2})" text-anchor="middle" class="axis-title">etiqueta real</text>
{''.join(headers)}
{''.join(cells)}
<text x="{left}" y="{height - 48}" class="subtitle">Verde: acierto en diagonal. Naranja: confusion entre clases.</text>
<text x="{left}" y="{height - 28}" class="subtitle">Escala normalizada por fila para que la clase O no oculte las entidades.</text>
</svg>
"""
    Path(path).write_text(svg, encoding="utf-8")


def _save_ner_artifacts(history, confusion, class_weights, metrics_dir):
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    _save_loss_svg(history, metrics_dir / "loss.svg")
    _save_non_o_accuracy_svg(history, metrics_dir / "non_o_accuracy.svg")
    _save_accuracy_svg(history, metrics_dir / "accuracy.svg")
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


def save_ner_metrics(history, entity_loss_weight, metrics_dir):
    """Guarda las metricas completas de un entrenamiento NER."""
    if not history:
        return
    best_row = next((row for row in history if row.get("is_best")), history[-1])
    confusion = best_row.get("confusion_matrix")
    if confusion is None:
        confusion = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    class_weights = _make_loss_weights(entity_loss_weight=entity_loss_weight)
    _save_ner_artifacts(history, confusion, class_weights, metrics_dir)


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
    train_ratio=0.85,
    add_spaces=False,
    entity_loss_weight=10.0,
    location_weight_multiplier=1.0,
    continuation_weight_multiplier=1.0,
    selection_accuracy_floor=0.8,
    selection_non_o_weight=1.5,
    warmup_steps=50,
    weight_decay=0.1,
    freeze_epochs=0,
    metrics_dir=None,
):
    """Fine-tuning NER sobre datos (tokens, labels)."""
    from train import _make_scheduler, _run_epoch, logger

    train_data, val_data = _stratified_phrase_split(ner_data, train_ratio=train_ratio)
    train_ds = NERDataset(
        train_data,
        tokenizer,
        max_len=max_len,
        add_spaces=add_spaces,
    )
    val_ds = NERDataset(
        val_data,
        tokenizer,
        max_len=max_len,
        add_spaces=add_spaces,
    )
    if len(train_ds) == 0:
        raise ValueError("No hay muestras NER tras alinear merged.json con el BPE.")

    label_counts = _label_counts(train_ds)
    val_label_counts = _label_counts(val_ds)
    class_weights = _make_loss_weights(
        entity_loss_weight=entity_loss_weight,
        location_weight_multiplier=location_weight_multiplier,
        continuation_weight_multiplier=continuation_weight_multiplier,
    )
    model.set_loss_weights(class_weights)
    logger.info(
        f"Split NER por frases: train={len(train_data)} frases/{len(train_ds)} chunks, "
        f"val={len(val_data)} frases/{len(val_ds)} chunks"
    )
    logger.info(
        "Conteo etiquetas NER train: "
        + ", ".join(
            f"{ID2LABEL[i]}={int(count)}" for i, count in enumerate(label_counts)
        )
    )
    logger.info(
        "Conteo etiquetas NER val: "
        + ", ".join(
            f"{ID2LABEL[i]}={int(count)}"
            for i, count in enumerate(val_label_counts)
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

    # Opcionalmente congelamos el backbone (tok_emb, blocks, norm) durante
    # los primeros freeze_epochs para que solo se entrene la cabeza NER.
    # Esto evita destruir las representaciones LM en datasets pequenos.
    if freeze_epochs > 0:
        logger.info(
            f"Congelando backbone los primeros {freeze_epochs} epochs "
            "(solo entrena ner_head)."
        )
        for name, param in model.named_parameters():
            if not name.startswith("ner_head"):
                param.requires_grad = False

    def _make_optimizer(params):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    optimizer = _make_optimizer(
        [p for p in model.parameters() if p.requires_grad]
    )
    total_steps = len(train_dl) * epochs
    scheduler = _make_scheduler(optimizer, warmup_steps, total_steps)
    t0 = time.time()
    history = []
    last_confusion = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    best_score = None
    best_state = None
    best_confusion = last_confusion
    for epoch in range(epochs):
        # Al llegar a freeze_epochs descongelamos todo y recreamos el optimizer
        # con una LR reducida para el backbone (fine-tuning diferenciado).
        if freeze_epochs > 0 and epoch == freeze_epochs:
            logger.info(
                f"Epoch {epoch + 1}: descongelando backbone, "
                f"fine-tuning completo con lr={lr * 0.1:.2e}"
            )
            for param in model.parameters():
                param.requires_grad = True
            optimizer = _make_optimizer([
                {"params": model.ner_head.parameters(), "lr": lr},
                {"params": [
                    p for name, p in model.named_parameters()
                    if not name.startswith("ner_head")
                ], "lr": lr * 0.1},
            ])
            remaining = len(train_dl) * (epochs - epoch)
            scheduler = _make_scheduler(optimizer, min(warmup_steps, 10), remaining)
        train_loss = _run_epoch(model, train_dl, optimizer, scheduler)
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
                    previous.pop("confusion_matrix", None)
                row["is_best"] = True
                row["confusion_matrix"] = last_confusion
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
