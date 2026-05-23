"""Generación de gráficas SVG y artefactos de métricas NER."""

import html
import json
from pathlib import Path

from .model import ID2LABEL, NUM_LABELS


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
    max_epoch = max(
        (row.get("epoch", i + 1) for i, row in enumerate(history)), default=1
    )

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
            *(round(1 + (max_epoch - 1) * i / 4) for i in range(1, 4) if max_epoch > 4),
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
{"".join(y_grid)}
{"".join(x_grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
{"".join(lines)}
{"".join(legend)}
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
{"".join(headers)}
{"".join(cells)}
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
