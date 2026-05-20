"""Generación del HTML del informe de etiquetado."""

from __future__ import annotations

import html
from datetime import datetime, timezone


_LABEL_CHIP_COLORS = {
    "o": ("#94a3b8", "#f1f5f9"),
    "pi": ("#4f46e5", "#eef2ff"),
    "pc": ("#6366f1", "#e0e7ff"),
    "li": ("#059669", "#ecfdf5"),
    "lc": ("#10b981", "#d1fae5"),
}

_LABEL_CHIP_META = {
    "o": ("Fuera", "Tokens sin entidad"),
    "pi": ("Persona · inicio", "PER (BIO)"),
    "pc": ("Persona · cont.", "PER (BIO)"),
    "li": ("Lugar · inicio", "LOC (BIO)"),
    "lc": ("Lugar · cont.", "LOC (BIO)"),
}


def _merged_json_chips_html(dist: dict[str, int], total: int) -> str:
    if total <= 0:
        return '<p class="note">Sin datos en merged.json.</p>'

    entity_total = sum(dist.get(label, 0) for label in ("pi", "pc", "li", "lc"))
    chips: list[str] = []

    for label in ["o", "pi", "pc", "li", "lc"]:
        count = dist.get(label, 0)
        accent, bg = _LABEL_CHIP_COLORS[label]
        title, subtitle = _LABEL_CHIP_META[label]
        pct_total = 100 * count / total
        pct_entity = 100 * count / entity_total if label != "o" and entity_total else 0
        bar_width = pct_total if label == "o" else pct_entity
        entity_line = (
            f'<span class="chip-sub">{pct_entity:.1f}% de entidades</span>'
            if label != "o" and entity_total
            else ""
        )
        chips.append(
            f'<div class="label-chip" style="--chip-accent:{accent};--chip-bg:{bg}">'
            f'<div class="chip-top">'
            f'<span class="chip-code">{html.escape(label)}</span>'
            f'<span class="chip-title">{html.escape(title)}</span>'
            f"</div>"
            f'<span class="chip-caption">{html.escape(subtitle)}</span>'
            f'<div class="chip-stat-row">'
            f'<span class="chip-count">{count:,}</span>'
            f'<span class="chip-pct">{pct_total:.1f}%</span>'
            f"</div>"
            f'{entity_line}'
            f'<div class="chip-bar"><span style="width:{min(bar_width, 100):.1f}%"></span></div>'
            f"</div>"
        )

    summary = (
        f'<p class="chip-summary">'
        f"<strong>{entity_total:,}</strong> tokens de entidad "
        f"({100 * entity_total / total:.1f}% del corpus) · "
        f"<strong>{total:,}</strong> tokens en total"
        f"</p>"
    )
    return summary + f'<div class="label-chip-grid">{"".join(chips)}</div>'


def _kappa_badge(value: float | None) -> str:
    if value is None:
        return '<span class="badge badge-muted">—</span>'
    if value >= 0.8:
        cls = "badge-good"
    elif value >= 0.6:
        cls = "badge-mid"
    else:
        cls = "badge-low"
    return f'<span class="badge {cls}">{value:.3f}</span>'


def build_report_html(data: dict, charts_js: str) -> str:
    report = data["report"]
    kappas = data["kappa_values"]
    confusion_labels = data["confusion_labels"]
    now = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

    heatmap_rows = []
    for row_label in confusion_labels:
        row = data["confusion"].get(row_label, {})
        max_val = max(row.values(), default=1) or 1
        cells = []
        for col_label in confusion_labels:
            val = row.get(col_label, 0)
            intensity = min(val / max_val, 1.0)
            alpha = 0.06 + 0.64 * intensity
            cells.append(
                f'<td style="background:rgba(99,102,241,{alpha:.2f})">{val}</td>'
            )
        heatmap_rows.append(
            f"<tr><th>{html.escape(row_label)}</th>{''.join(cells)}</tr>"
        )

    worst = data["per_frase"][:6]
    best = sorted(data["per_frase"], key=lambda r: r["kappa"] or 0, reverse=True)[:5]

    coverage_rows = "".join(
        f"<tr><td class='mono'>{html.escape(r['file'])}</td>"
        f"<td>{r['tokens']}</td><td>{r['entity_tokens']}</td>"
        f"<td class='pct-cell'>"
        f"<span class='mini-bar-wrap'><span class='mini-bar' style='width:{min(r['entity_pct'], 100):.0f}%'></span></span>"
        f"{r['entity_pct']:.1f}%</td></tr>"
        for r in sorted(data["coverage"], key=lambda x: -x["entity_pct"])[:15]
    )

    frase_rows = "".join(
        f"<tr><td class='mono'>{f['frase_id']}</td>"
        f"<td><span class='lote-tag'>{f['lote']}</span></td>"
        f"<td>{_kappa_badge(f['kappa'])}</td>"
        f"<td>{f['agreement']:.0%}</td><td>{f['disagreements']}</td>"
        f"<td class='preview'>{html.escape(f['text_preview'][:72])}…</td></tr>"
        for f in data["per_frase"]
    )

    def rank_row(f: dict) -> str:
        return (
            f"<tr><td class='mono'>{f['frase_id']}</td>"
            f"<td><span class='lote-tag'>{f['lote']}</span></td>"
            f"<td>{_kappa_badge(f['kappa'])}</td>"
            f"<td>{f['agreement']:.0%}</td></tr>"
        )

    worst_rows = "".join(rank_row(f) for f in worst)
    best_rows = "".join(rank_row(f) for f in best)

    l1, l2 = report["lotes"][0], report["lotes"][1]
    confusion_header = "".join(f"<th>{html.escape(c)}</th>" for c in confusion_labels)
    n_frases = report["n_frases"]
    merged_json_dist = data.get("merged_json_label_dist", {})
    merged_json_total = sum(merged_json_dist.values())
    merged_json_chips = _merged_json_chips_html(merged_json_dist, merged_json_total)
    kappa = report["mean_cohen_kappa"]
    agreement = report["mean_token_agreement"]

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Informe NER · Alice in Wonderland</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --text: #0f172a;
      --text-secondary: #475569;
      --muted: #94a3b8;
      --border: #e2e8f0;
      --border-light: #f1f5f9;
      --accent: #6366f1;
      --accent-soft: #eef2ff;
      --accent-glow: rgba(99, 102, 241, 0.12);
      --success: #10b981;
      --success-soft: #ecfdf5;
      --warn: #f59e0b;
      --warn-soft: #fffbeb;
      --danger: #ef4444;
      --danger-soft: #fef2f2;
      --radius: 12px;
      --radius-sm: 8px;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.03);
      --shadow-hover: 0 4px 16px rgba(15, 23, 42, 0.06);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      font-size: 13px;
      line-height: 1.5;
      background: var(--bg);
      color: var(--text);
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 1.5rem 1.25rem 2rem;
    }}
    .hero {{
      margin-bottom: 1.25rem;
    }}
    .hero-top {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .hero h1 {{
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: var(--text);
    }}
    .hero h1 span {{
      color: var(--accent);
    }}
    .hero-sub {{
      font-size: 0.8rem;
      color: var(--text-secondary);
      margin-top: 0.25rem;
    }}
    .hero-meta {{
      font-size: 0.72rem;
      color: var(--muted);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.35rem 0.85rem;
      white-space: nowrap;
    }}
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
    }}
    @media (max-width: 720px) {{
      .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi {{
      background: var(--surface);
      border: 1px solid var(--border-light);
      border-radius: var(--radius);
      padding: 0.85rem 1rem;
      box-shadow: var(--shadow);
      transition: box-shadow 0.2s, transform 0.2s;
    }}
    .kpi:hover {{
      box-shadow: var(--shadow-hover);
      transform: translateY(-1px);
    }}
    .kpi.featured {{
      background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
      border: none;
      color: #fff;
    }}
    .kpi.featured .kpi-label {{ color: rgba(255,255,255,0.75); }}
    .kpi.featured .kpi-value {{ color: #fff; }}
    .kpi-label {{
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }}
    .kpi-value {{
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
    }}
    .section {{
      margin-top: 1rem;
    }}
    .section-title {{
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 0.5rem;
      padding-left: 0.15rem;
    }}
    .grid {{ display: grid; gap: 0.75rem; }}
    .g3 {{ grid-template-columns: repeat(3, 1fr); }}
    .g2 {{ grid-template-columns: repeat(2, 1fr); }}
    @media (max-width: 720px) {{
      .g3, .g2 {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border-light);
      border-radius: var(--radius);
      padding: 0.9rem 1rem;
      box-shadow: var(--shadow);
    }}
    .card h3 {{
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      margin-bottom: 0.65rem;
    }}
    .lote-grid {{
      display: grid;
      gap: 0.5rem;
    }}
    .lote-item {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      padding: 0.55rem 0.7rem;
      background: var(--bg);
      border-radius: var(--radius-sm);
    }}
    .lote-tag {{
      display: inline-block;
      font-size: 0.65rem;
      font-weight: 600;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .lote-stats {{
      font-size: 0.78rem;
      color: var(--text-secondary);
    }}
    .lote-stats strong {{
      color: var(--text);
      font-weight: 600;
    }}
    .note {{
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 0.5rem;
    }}
    .chart-wrap {{ height: 120px; position: relative; }}
    .chart-wrap.sm {{ height: 100px; }}
    .chip-summary {{
      font-size: .78rem;
      color: var(--text-secondary);
      margin-bottom: .75rem;
      line-height: 1.45;
    }}
    .chip-summary strong {{ color: var(--text); font-weight: 600; }}
    .label-chip-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
      gap: .55rem;
    }}
    .label-chip {{
      background: var(--chip-bg);
      border: 1px solid color-mix(in srgb, var(--chip-accent) 18%, transparent);
      border-radius: var(--radius-sm);
      padding: .65rem .75rem;
      display: flex;
      flex-direction: column;
      gap: .25rem;
    }}
    .chip-top {{
      display: flex;
      align-items: baseline;
      gap: .4rem;
    }}
    .chip-code {{
      font-family: ui-monospace, monospace;
      font-size: .72rem;
      font-weight: 700;
      color: var(--chip-accent);
      letter-spacing: .02em;
    }}
    .chip-title {{
      font-size: .72rem;
      font-weight: 600;
      color: var(--text);
    }}
    .chip-caption {{
      font-size: .65rem;
      color: var(--muted);
    }}
    .chip-stat-row {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: .5rem;
      margin-top: .15rem;
    }}
    .chip-count {{
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.02em;
    }}
    .chip-pct {{
      font-size: .72rem;
      font-weight: 600;
      color: var(--chip-accent);
    }}
    .chip-sub {{
      font-size: .62rem;
      color: var(--text-secondary);
    }}
    .chip-bar {{
      height: 4px;
      background: rgba(15, 23, 42, 0.06);
      border-radius: 99px;
      overflow: hidden;
      margin-top: .2rem;
    }}
    .chip-bar span {{
      display: block;
      height: 100%;
      background: var(--chip-accent);
      border-radius: 99px;
      min-width: 2px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.75rem;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: var(--surface);
      z-index: 1;
      font-size: 0.65rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      padding: 0.45rem 0.5rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    tbody td {{
      padding: 0.4rem 0.5rem;
      border-bottom: 1px solid var(--border-light);
      color: var(--text-secondary);
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: var(--accent-glow); }}
    td.mono {{
      font-variant-numeric: tabular-nums;
      font-weight: 500;
      color: var(--text);
    }}
    td.preview {{
      max-width: 220px;
      font-size: 0.7rem;
      color: var(--muted);
      line-height: 1.4;
    }}
    .heatmap th, .heatmap td {{
      padding: 0.35rem 0.4rem;
      text-align: center;
      font-size: 0.7rem;
    }}
    .heatmap th:first-child {{ text-align: left; }}
    .heatmap tbody th {{
      font-weight: 600;
      color: var(--text);
      background: var(--bg);
    }}
    .scroll {{
      max-height: 170px;
      overflow: auto;
      margin: 0 -0.25rem;
      padding: 0 0.25rem;
    }}
    .scroll-lg {{ max-height: 240px; overflow: auto; }}
    .scroll::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    .scroll::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 99px;
    }}
    .badge {{
      display: inline-block;
      font-size: 0.68rem;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
      padding: 0.12rem 0.45rem;
      border-radius: 999px;
    }}
    .badge-good {{ background: var(--success-soft); color: #059669; }}
    .badge-mid {{ background: var(--warn-soft); color: #d97706; }}
    .badge-low {{ background: var(--danger-soft); color: #dc2626; }}
    .badge-muted {{ background: var(--bg); color: var(--muted); }}
    .pct-cell {{
      font-variant-numeric: tabular-nums;
      font-weight: 500;
      color: var(--text);
      white-space: nowrap;
    }}
    .mini-bar-wrap {{
      display: inline-block;
      width: 44px;
      height: 4px;
      background: var(--border);
      border-radius: 99px;
      margin-right: 0.45rem;
      vertical-align: middle;
      overflow: hidden;
    }}
    .mini-bar {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #818cf8);
      border-radius: 99px;
    }}
    details.card {{
      cursor: default;
    }}
    details.card summary {{
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      user-select: none;
    }}
    details.card summary::-webkit-details-marker {{ display: none; }}
    details.card summary::after {{
      content: '▼';
      font-size: 0.55rem;
      color: var(--muted);
      transition: transform 0.2s;
    }}
    details.card[open] summary::after {{
      transform: rotate(180deg);
    }}
    details.card[open] summary {{ margin-bottom: 0.65rem; }}
    footer {{
      text-align: center;
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 1.5rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border-light);
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="hero-top">
        <div>
          <h1>Informe de etiquetado <span>NER</span></h1>
          <p class="hero-sub">Alice in Wonderland · acuerdo inter-anotador</p>
        </div>
        <span class="hero-meta">{now}</span>
      </div>
      <div class="kpi-row">
        <div class="kpi featured">
          <div class="kpi-label">κ de Cohen (medio)</div>
          <div class="kpi-value">{kappa:.3f}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Acuerdo por token</div>
          <div class="kpi-value">{agreement:.0%}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Frases fusionadas</div>
          <div class="kpi-value">{n_frases}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Pares anotados</div>
          <div class="kpi-value">{len(kappas)}</div>
        </div>
      </div>
    </header>

    <div class="section">
      <p class="section-title">Distribución y lotes</p>
      <div class="grid g2">
        <div class="card">
          <h3>Lotes de anotación</h3>
          <div class="lote-grid">
            <div class="lote-item">
              <span class="lote-tag">P1</span>
              <span class="lote-stats">
                <strong>{l1["n_frases_fusionadas"]}</strong> frases ·
                κ <strong>{l1["mean_cohen_kappa"]:.2f}</strong> ·
                {l1["mean_token_agreement"]:.0%} acuerdo
              </span>
            </div>
            <div class="lote-item">
              <span class="lote-tag">P2</span>
              <span class="lote-stats">
                <strong>{l2["n_frases_fusionadas"]}</strong> frases ·
                κ <strong>{l2["mean_cohen_kappa"]:.2f}</strong> ·
                {l2["mean_token_agreement"]:.0%} acuerdo
              </span>
            </div>
          </div>
          <p class="note">Frases omitidas sin etiquetar: P1={l1["skipped_unlabeled"]}, P2={l2["skipped_unlabeled"]}</p>
        </div>
        <div class="card">
          <h3>Entidades detectadas</h3>
          <div class="chart-wrap"><canvas id="chartEntities"></canvas></div>
        </div>
      </div>
    </div>

    <div class="section">
      <p class="section-title">Corpus fusionado (merged.json)</p>
      <div class="card">
        <h3>Etiquetas por tipo · merged.json</h3>
        {merged_json_chips}
      </div>
    </div>

    <div class="section">
      <p class="section-title">Análisis de etiquetas y concordancia</p>
      <div class="grid g3">
        <div class="card">
          <h3>Distribución de etiquetas</h3>
          <div class="chart-wrap"><canvas id="chartLabels"></canvas></div>
        </div>
        <div class="card">
          <h3>Histograma κ</h3>
          <div class="chart-wrap sm"><canvas id="chartKappaHist"></canvas></div>
        </div>
        <div class="card">
          <h3>κ por frase</h3>
          <div class="chart-wrap sm"><canvas id="chartKappaScatter"></canvas></div>
        </div>
      </div>
    </div>

    <div class="section">
      <p class="section-title">Matrices y cobertura</p>
      <div class="grid g2">
        <div class="card">
          <h3>Matriz de confusión (tokens)</h3>
          <div class="scroll">
            <table class="heatmap">
              <thead><tr><th></th>{confusion_header}</tr></thead>
              <tbody>{"".join(heatmap_rows)}</tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h3>Cobertura de entidades</h3>
          <div class="scroll">
            <table>
              <thead><tr><th>Fichero</th><th>Tokens</th><th>Ent.</th><th>%</th></tr></thead>
              <tbody>{coverage_rows}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="section">
      <p class="section-title">Ranking por concordancia</p>
      <div class="grid g2">
        <div class="card">
          <h3>Menor κ</h3>
          <table>
            <thead><tr><th>ID</th><th>Lote</th><th>κ</th><th>Acuerdo</th></tr></thead>
            <tbody>{worst_rows}</tbody>
          </table>
        </div>
        <div class="card">
          <h3>Mayor κ</h3>
          <table>
            <thead><tr><th>ID</th><th>Lote</th><th>κ</th><th>Acuerdo</th></tr></thead>
            <tbody>{best_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <details class="card section">
      <summary>Detalle de todas las frases ({n_frases})</summary>
      <div class="scroll-lg">
        <table>
          <thead><tr><th>ID</th><th>Lote</th><th>κ</th><th>Acuerdo</th><th>Desac.</th><th>Texto</th></tr></thead>
          <tbody>{frase_rows}</tbody>
        </table>
      </div>
    </details>

    <footer>Generado con fdi-pln-2611-p5</footer>
  </div>
  <script>{charts_js}</script>
</body>
</html>
"""
