"""Evaluación y métricas NER."""

import torch

from .model import ID2LABEL, LABEL2ID, NUM_LABELS


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
