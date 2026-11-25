"""README summary figure (2026-11-25). Values are copied from host evaluation records cited in
docs/STATUS_1125.ko.md and paper/cross-task/manuscript_v2.md; no model is run. Writes PNG/SVG + CSV."""
import csv, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'assets', 'summary')
EDIT, BASE, ORACLE = '#2a78d6', '#b9b8b2', '#86b6ef'
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'

panels = [
    ('A. Fact edit, fact given in context\n4B, sealed set, S (higher = better)',
     [('No edit (best route)', .758, 'b'), ('Logit bias (best)', .756, 'b'), ('Steering (best)', .747, 'b'),
      ('Oracle steering*', .918, 'o'), ('LLM weight edit', .920, 'e')], (0, 1), '{:.3f}'),
    ('B. Fact edit, no context\n4B, sealed set, S',
     [('No edit', .092, 'b'), ('LLM weight edit', .682, 'e'), ('ROME', .806, 'b'), ('MEMIT', .897, 'b'), ('LoRA', .898, 'b')],
     (0, 1), '{:.3f}'),
    ('C. IFEval, thinking mode\n0.8B, sealed 221 items, accuracy %',
     [('Original', 20.8, 'b'), ('LLM weight edit', 65.6, 'e'), ('Budget forcing 4k', 70.1, 'b'),
      ('Edit + budget forcing 4k', 65.6, 'e')], (0, 100), '{:.1f}'),
    ('D. 4B defect audit, decoding only (no edit)\nheld-out 550 items, accuracy %',
     [('Greedy (default)', 77.5, 'b'), ('Greedy + repetition\nsuppression (Qwen rec.)', 86.2, 'b')], (0, 100), '{:.1f}'),
]
colors = {'e': EDIT, 'b': BASE, 'o': ORACLE}

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK})
fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), facecolor=SURF)
rows = []
for ax, (title, bars, xlim, fmt) in zip(axs.flat, panels):
    ax.set_facecolor(SURF)
    labels = [b[0] for b in bars][::-1]; vals = [b[1] for b in bars][::-1]; kinds = [b[2] for b in bars][::-1]
    y = range(len(bars))
    ax.barh(y, vals, height=0.6, color=[colors[k] for k in kinds], hatch=None)
    for yi, v, k in zip(y, vals, kinds):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01, yi, fmt.format(v), va='center', fontsize=9,
                color=INK, fontweight='bold' if k == 'e' else 'normal')
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlim(xlim[0], xlim[1] * 1.12); ax.set_title(title, loc='left', fontsize=10.5, color=INK)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e4e3df', linewidth=0.6); ax.set_axisbelow(True)
    for l, v, k in bars: rows.append({'panel': title.split('\n')[0], 'bar': l.replace('\n', ' '), 'value': v,
                                       'kind': {'e': 'edit', 'b': 'baseline', 'o': 'oracle'}[k]})
fig.text(0.01, 0.005, '* Oracle steering is told where each subject appears (not deployable).  Blue = LLM-written weight edit, '
         'gray = comparison.  Sources: docs/STATUS_1125.ko.md.', fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ('png', 'svg'):
    fig.savefig(os.path.join(OUT, f'results_summary.{ext}'), dpi=150, facecolor=SURF)
with open(os.path.join(OUT, 'results_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['panel', 'bar', 'value', 'kind']); w.writeheader(); w.writerows(rows)
print('ok')
