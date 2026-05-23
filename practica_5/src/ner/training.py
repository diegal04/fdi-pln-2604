"""Bucle de entrenamiento NER, split estratificado y utilidades de selección."""

import time

import torch
from torch.utils.data import DataLoader

from training.lm import _make_scheduler, _run_epoch, logger
from .model import LABEL2ID, ID2LABEL, NUM_LABELS, NERDataset, collate_ner
from .metrics import _eval_ner
from .visualization import _save_ner_artifacts


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


def _make_loss_weights(
    entity_loss_weight=10.0,
    location_weight_multiplier=1.0,
    continuation_weight_multiplier=1.0,
):
    """Pesos de loss por clase.

    - o = 1
    - pi (B-PER) = entity_loss_weight
    - pc (I-PER) = entity_loss_weight * continuation_weight_multiplier
    - li (B-LOC) = entity_loss_weight * location_weight_multiplier
    - lc (I-LOC) = entity_loss_weight * location_weight_multiplier * continuation_weight_multiplier
    """
    weights = torch.ones(NUM_LABELS, dtype=torch.float)
    weights[LABEL2ID["pi"]] = entity_loss_weight
    weights[LABEL2ID["pc"]] = entity_loss_weight * continuation_weight_multiplier
    weights[LABEL2ID["li"]] = entity_loss_weight * location_weight_multiplier
    weights[LABEL2ID["lc"]] = (
        entity_loss_weight * location_weight_multiplier * continuation_weight_multiplier
    )
    return weights


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
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def save_ner_metrics(
    history,
    entity_loss_weight,
    metrics_dir,
    location_weight_multiplier=1.0,
    continuation_weight_multiplier=1.0,
):
    """Guarda las metricas completas de un entrenamiento NER."""
    if not history:
        return
    best_row = next((row for row in history if row.get("is_best")), history[-1])
    confusion = best_row.get("confusion_matrix")
    if confusion is None:
        confusion = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    class_weights = _make_loss_weights(
        entity_loss_weight=entity_loss_weight,
        location_weight_multiplier=location_weight_multiplier,
        continuation_weight_multiplier=continuation_weight_multiplier,
    )
    _save_ner_artifacts(history, confusion, class_weights, metrics_dir)


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
            f"{ID2LABEL[i]}={int(count)}" for i, count in enumerate(val_label_counts)
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

    optimizer = _make_optimizer([p for p in model.parameters() if p.requires_grad])
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
            optimizer = _make_optimizer(
                [
                    {"params": model.ner_head.parameters(), "lr": lr},
                    {
                        "params": [
                            p
                            for name, p in model.named_parameters()
                            if not name.startswith("ner_head")
                        ],
                        "lr": lr * 0.1,
                    },
                ]
            )
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
