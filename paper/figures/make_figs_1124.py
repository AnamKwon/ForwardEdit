"""Manuscript-revision figures (2026-11-24). CPU only, reads recorded JSON; loads no model.

Run:  CUDA_VISIBLE_DEVICES="" WSL_DATA_ROOT=<host evidence root> python3 paper/figures/make_figs_1124.py
(WSL_DATA_ROOT is required and has no default; the script exits if it is unset.)
Writes fig_ladder, fig_steer_dose, fig_replicates, fig_qcf20, fig_loops (.png 200 dpi, .svg) plus a .csv
table view of every plotted value next to this script. Any missing key raises immediately.

Palette: dataviz reference categorical slots in fixed order (blue, orange, aqua, yellow), validated with
validate_palette.js --mode light (all hard checks pass; aqua/yellow < 3:1 contrast -> relief via legend,
selected direct labels and the CSV table view). Neutral gray marks "no edit" as a reference, not a series.
Status green/red mark PASS/FAIL and always carry a text label.
"""
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.environ.get('WSL_DATA_ROOT', '').rstrip('/')
if not ROOT:
    raise SystemExit('WSL_DATA_ROOT is not set: export it to the host evidence root '
                     '(the directory that holds qcf15_4b/, qcap_08b/, bench/, defect_audit/ ...)')
DATA = {
    'ladder_qcf15': f'{ROOT}/qcf15_4b/baseline_ladder.json',
    'ladder_qcf16': f'{ROOT}/qcf16_4b/baseline_ladder.json',
    'qcap_prereg': f'{ROOT}/qcap_08b/PREREG.md',
    'qcap_compare': f'{ROOT}/qcap_08b/compare_reps.json',
    'qcap_sealed': {r: f'{ROOT}/qcap_08b/sealed_eval_qcap_{r}_claude.json' for r in ('on', 'off', 'rep1', 'rep2')},
    'qcf20_sealed': f'{ROOT}/qcf20_9b/sealed_eval.json',
    'qcf20_regress': f'{ROOT}/bench/regress_qcf20_9b.json',
    'scan_off': f'{ROOT}/defect_audit/scan_4b_off.json',
    'scan_off_items': f'{ROOT}/defect_audit/scan_4b_off.partial.jsonl',
    'scan_on_items': f'{ROOT}/defect_audit/scan_4b_on8k.partial.jsonl',
}
OUT = Path(__file__).resolve().parent

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
NEUTRAL = '#9a9890'
GOOD, CRITICAL = '#0ca30c', '#d03b3b'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.unicode_minus': True,
    'axes.edgecolor': '#b5b4ad', 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'svg.fonttype': 'none', 'legend.frameon': False,
})


# ---------------------------------------------------------------- helpers
def load(path):
    if not Path(path).exists():
        raise FileNotFoundError(f'missing data file: {path}')
    return json.loads(Path(path).read_text())


def req(d, *keys):
    cur = d
    for i, k in enumerate(keys):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f'missing key {"/".join(map(str, keys[:i + 1]))}')
        cur = cur[k]
    return cur


def style(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8, ls='-')
    ax.set_axisbelow(True)


def save(fig, name, rows, header):
    fig.savefig(OUT / f'{name}.png', dpi=200, metadata={'Software': None})
    fig.savefig(OUT / f'{name}.svg', metadata={'Date': None, 'Creator': None})
    plt.close(fig)
    with open(OUT / f'{name}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', name)


COMPONENTS = [('rewrite_generation_nll', 'New fact\nadopted'),
              ('preserve_neighborhood_nll', 'Neighbours kept\n(in context)'),
              ('preserve_neighborhood_plain_nll', 'Neighbours kept\n(plain)'),
              ('S', 'S\n(harmonic mean)')]
# (display label, data key); labels are the readable experiment names (docs/EXPERIMENT_NAMES.md)
TASKS = [('4B Fact-Edit A', 'ladder_qcf15'), ('4B Fact-Edit B', 'ladder_qcf16')]


# ---------------------------------------------------------------- fig 1: baseline ladder
def fig_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    rows = []
    handles = None
    for ax, (task, key) in zip(axes, TASKS):
        d = load(DATA[key])
        arms = req(d, 'arms')
        lb_name = req(d, 'best', 'LB')[0]
        st_name = req(d, 'best', 'ST')[0]
        series = [('No edit', req(arms, 'O'), NEUTRAL),
                  (f'Best logit bias, {lb_name}', req(arms, lb_name), ORANGE),
                  (f'Best always-on steering, {st_name}', req(arms, st_name), AQUA),
                  ('Oracle per-fact steering (OR)', req(arms, 'OR'), YELLOW),
                  ('LLM edit (E)', req(arms, 'E'), BLUE)]
        n = len(series)
        w = 0.15
        hs = []
        for j, (lab, vals, col) in enumerate(series):
            xs = [i + (j - (n - 1) / 2) * (w + 0.012) for i in range(len(COMPONENTS))]
            ys = [req(vals, c) for c, _ in COMPONENTS]
            hs.append(ax.bar(xs, ys, w, color=col, label=lab.split(',')[0] if j in (1, 2) else lab))
            # direct label only on S (the headline component)
            ax.text(xs[-1], ys[-1] + 0.015, f'{ys[-1]:.2f}', ha='center', va='bottom', fontsize=7, color=INK,
                    rotation=90)
            for (c, _), y in zip(COMPONENTS, ys):
                rows.append([task, lab, c, y, req(vals, 'mmlu200')])
        handles = hs
        ax.set_xticks(range(len(COMPONENTS)))
        ax.set_xticklabels([l for _, l in COMPONENTS])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, .25, .5, .75, 1])
        style(ax)
        ax.set_title(f'{task} (layer {req(d, "layer")})   best grid: {lb_name}, {st_name}',
                     loc='left', fontsize=9.5, color=INK)
    axes[0].set_ylabel('Validation score (fraction of items)')
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.06, 1.0), ncol=5, fontsize=8.2,
               handlelength=1.2, columnspacing=1.2)
    fig.text(0.06, 0.012, 'Validation split, not sealed. Baseline strengths chosen on the same split '
             '(favours baselines); "best" = highest S with MMLU-200 within 2 points of no edit.',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, 'fig_ladder', rows, ['task', 'arm', 'component', 'score', 'mmlu200'])


# ---------------------------------------------------------------- fig 2: steering dose
def fig_steer_dose():
    measures = COMPONENTS + [('mmlu200', 'MMLU-200\naccuracy')]
    fig, axes = plt.subplots(1, len(measures), figsize=(11.5, 3.3), sharey=True)
    cols = {TASKS[0][0]: BLUE, TASKS[1][0]: ORANGE}
    rows = []
    lines = {}
    for task, key in TASKS:
        d = load(DATA[key])
        arms = req(d, 'arms')
        st = sorted(((float(re.fullmatch(r'ST\(([\d.]+)\)', k).group(1)), v) for k, v in arms.items()
                     if k.startswith('ST(')), key=lambda t: t[0])
        if not st:
            raise KeyError(f'{task}: no ST(x) arms')
        strengths = [s for s, _ in st]
        xs = list(range(len(st) + 1))
        for ax, (m, lab) in zip(axes, measures):
            ys = [req(arms, 'O', m)] + [req(v, m) for _, v in st]
            ln, = ax.plot(xs, ys, color=cols[task], lw=2, marker='o', ms=6, mec=SURFACE, mew=1.5,
                          solid_capstyle='round', label=f'{task}: steering')
            e = req(arms, 'E', m)
            ax.plot([len(xs) + 0.4 + (-0.18 if task == TASKS[0][0] else 0.18)], [e], ls='none', marker='D', ms=6, color=cols[task], mec=SURFACE, mew=1.5)
            lines[task] = ln
            for s, y in zip([0] + strengths, ys):
                rows.append([task, m, s, y])
            rows.append([task, m, 'edit', e])
    xt = ['none'] + [f'{s:g}' for s in strengths] + ['edit']
    for ax, (m, lab) in zip(axes, measures):
        ax.set_xticks(list(range(len(strengths) + 1)) + [len(strengths) + 1.4])
        ax.set_xticklabels(xt, fontsize=8)
        ax.axvline(len(strengths) + 0.7, color=GRID, lw=0.8)
        ax.set_title(lab, fontsize=9, color=INK, loc='left')
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('steering strength s')
        style(ax)
    axes[0].set_ylabel('Validation score / accuracy')
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=cols[t], lw=2, marker='o', ms=6, mec=SURFACE, label=f'{t}: always-on steering')
          for t, _ in TASKS] + \
         [Line2D([], [], color=INK2, ls='none', marker='D', ms=6, label='LLM edit (same task), for reference')]
    fig.legend(handles=hl, loc='upper left', bbox_to_anchor=(0.05, 1.0), ncol=3, fontsize=8.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, 'fig_steer_dose', rows, ['task', 'measure', 'strength', 'value'])


# ---------------------------------------------------------------- fig 3: qcap replicates
def fig_replicates():
    prereg = Path(DATA['qcap_prereg']).read_text()
    cmp_ = load(DATA['qcap_compare'])
    behaviour = req(cmp_, 'behaviour', 'arms')
    runs = []
    for r, label in [('on', '0.8B Fact-Edit (2% cap)'), ('off', '0.8B Fact-Edit (no cap; never bound)'),
                     ('rep1', '0.8B Fact-Edit (2% cap) replicate 1'),
                     ('rep2', '0.8B Fact-Edit (2% cap) replicate 2')]:
        e = load(DATA['qcap_sealed'][r])
        p = req(e, 'primary')
        layers = sorted({int(m.group(1)) for t in req(e, 'footprint', 'tensors')
                         for m in [re.match(r'model\.layers\.(\d+)\.', t)] if m}, reverse=True)
        s, bar = req(p, 'edit_allctx_S'), req(p, 'best_noedit_S')
        ps, ns = req(p, 'PS_allctx'), req(p, 'NS_allctx')
        c1 = s > bar
        c2 = req(ps, 'diff_pts') > 2 * req(ps, 'se_pts')
        c3 = req(ns, 'diff_pts') >= -2 * req(ns, 'se_pts')
        arm = f'qcap_{r}_claude'
        if c1 and c2 and c3:
            gen = req(behaviour, arm, 'general_bf16')  # needed only when 1-3 pass
            c4 = all(req(v, 'diff_pts') >= -2 * req(v, 'se_pts') for v in gen.values())
        else:
            c4 = None  # not needed: PRIMARY already failed
        passed = bool(c1 and c2 and c3 and c4)
        runs.append(dict(label=label, layer='+'.join(f'L{x}' for x in layers), S=s, bar=bar,
                         ps=req(ps, 'diff_pts'), se=req(ps, 'se_pts'), passed=passed, frozen=True))
    m = re.search(r'^\| rep3 \| (L\d+)[^|]*\|[^|]*\| ✗ \(([^)]*)\)', prereg, re.M)
    if not m:
        raise KeyError('PREREG.md replicates table: rep3 row not found')
    if req(cmp_, 'structure', 'qcap_rep3_claude', 'status') != 'no freeze':
        raise ValueError('rep3 status changed; update figure')
    runs.append(dict(label='0.8B Fact-Edit (2% cap) replicate 3', layer=m.group(1), frozen=False, note='no freeze'))
    bar = runs[0]['bar']
    assert all(r['bar'] == bar for r in runs if r['frozen'])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True, gridspec_kw={'width_ratios': [1, 1.15]})
    ys = list(range(len(runs)))[::-1]
    rows = []
    for y, r in zip(ys, runs):
        if not r['frozen']:
            a1.text(0.02, y, 'no freeze', transform=a1.get_yaxis_transform(),
                    ha='left', va='center', fontsize=8, color=INK2, style='italic')
            a2.text(0.5, y, 'no freeze (confirmation gate failed)', transform=a2.get_yaxis_transform(),
                    ha='center', va='center', fontsize=8, color=INK2, style='italic')
            rows.append([r['label'], r['layer'], 'no freeze', '', '', '', ''])
            continue
        col = GOOD if r['passed'] else CRITICAL
        fill = col if r['passed'] else SURFACE
        a1.plot([r['S']], [y], 'o', ms=8, color=col, mfc=fill, mew=2)
        a1.text(r['S'] + 0.004, y + 0.22, f"{r['S']:.3f}", fontsize=7.5, color=INK, ha='left')
        a2.errorbar([r['ps']], [y], xerr=[[2 * r['se']], [2 * r['se']]], fmt='o', ms=8, color=col, mfc=fill,
                    mew=2, elinewidth=1.5, capsize=0)
        a2.text(r['ps'] + 2 * r['se'] + 0.8, y, ('PASS' if r['passed'] else 'FAIL'), va='center', fontsize=8,
                color=INK, fontweight='bold')
        rows.append([r['label'], r['layer'], 'frozen', r['S'], r['ps'], r['se'], 'PASS' if r['passed'] else 'FAIL'])
    a1.axvline(bar, color=INK2, lw=1)
    a1.text(bar - 0.001, len(runs) - 0.35, f'strongest no-edit route {bar:.3f}', fontsize=7.5, color=INK2,
            ha='right', va='center')
    a1.set_xlim(0.76, 0.885)
    a1.set_xlabel('Sealed all-context S')
    a2.axvline(0, color=INK2, lw=1)
    a2.set_xlim(-8, 34)
    a2.set_xlabel('Sealed all-context PS gain, points (± 2 SE)')
    a1.set_yticks(ys)
    a1.set_yticklabels([f"{r['label']}  ·  {r['layer']}" for r in runs])
    a1.set_ylim(-0.6, len(runs) - 0.05)
    for ax in (a1, a2):
        style(ax, grid='x')
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], ls='none', marker='o', ms=8, color=GOOD, mew=2, label='PASS (all 4 PRIMARY criteria)'),
          Line2D([], [], ls='none', marker='o', ms=8, color=CRITICAL, mfc=SURFACE, mew=2, label='FAIL')]
    n_pass = sum(r.get('passed', False) for r in runs)
    n_frz = sum(r['frozen'] for r in runs)
    fig.suptitle(f'Qwen3.5-0.8B, {len(runs)} editor runs of one condition: {n_pass}/{len(runs)} PASS, '
                 f'{n_frz}/{len(runs)} freeze', x=0.01, ha='left', y=0.985, fontsize=9.5, color=INK)
    fig.legend(handles=hl, loc='upper left', bbox_to_anchor=(0.01, 0.93), ncol=2, fontsize=8.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, 'fig_replicates', rows, ['run', 'layers', 'status', 'sealed_allctx_S', 'PS_gain_pts', 'PS_se_pts',
                                       'primary'])


# ---------------------------------------------------------------- fig 4: qcf20 9B
ROUTES = ['none', 'all', 'all_x2', 'oracle', 'override']


def fig_qcf20():
    e = load(DATA['qcf20_sealed'])
    rg = load(DATA['qcf20_regress'])
    best = req(e, 'primary', 'best_noedit_S')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'width_ratios': [1.5, 1]})
    rows = []
    w = 0.34
    for i, r in enumerate(ROUTES):
        n0, n1 = req(e, 'routes', r, 'noedit', 'S'), req(e, 'routes', r, 'edit', 'S')
        a1.bar(i - w / 2 - 0.01, n0, w, color=NEUTRAL, label='No edit' if i == 0 else None)
        a1.bar(i + w / 2 + 0.01, n1, w, color=BLUE, label='LLM edit' if i == 0 else None)
        a1.text(i - w / 2 - 0.01, n0 + 0.015, f'{n0:.2f}', ha='center', fontsize=7.5, color=INK)
        a1.text(i + w / 2 + 0.01, n1 + 0.015, f'{n1:.2f}', ha='center', fontsize=7.5, color=INK)
        rows.append(['sealed_S', r, n0, n1, '', ''])
    a1.axhline(best, color=INK2, lw=1, label=f'Strongest no-edit route (all_x2), S = {best:.3f}')
    if max(ROUTES, key=lambda r: req(e, 'routes', r, 'noedit', 'S')) != 'all_x2':
        raise ValueError('strongest no-edit route is not all_x2; update legend text')
    a1.set_xticks(range(len(ROUTES)))
    a1.set_xticklabels(ROUTES)
    a1.set_xlabel('Prompt route (sealed set)')
    a1.set_ylabel('Sealed S')
    a1.set_ylim(0, 1.05)
    a1.legend(loc='upper left', bbox_to_anchor=(0, 1.13), ncol=3, fontsize=8.2, handlelength=1.4)
    style(a1)

    arm = 'qcf20_9b_claude'
    names = [('mmlu', 'MMLU (n=2000)'), ('arc', 'ARC-C (n=1172)'), ('gsm8k', 'GSM8K (n=200)')]
    for j, (k, lab) in enumerate(names):
        v = req(rg, arm, k)
        d, se = req(v, 'diff_pts'), req(v, 'se_pts')
        sig = d + 2 * se < 0
        y = len(names) - 1 - j
        a2.errorbar([d], [y], xerr=[[2 * se], [2 * se]], fmt='o', ms=8, color=CRITICAL if sig else INK2,
                    mfc=CRITICAL if sig else INK2, mec=SURFACE, mew=1.5, elinewidth=1.5, capsize=0)
        txt = f'{d:+.2f}'.replace('-', '\u2212') + f' ± {2 * se:.2f}' + ('  (interval excludes 0)' if sig else '')
        a2.text(d, y + 0.25, txt, ha='center', fontsize=7.5, color=INK)
        rows.append(['regression', k, req(v, 'orig'), req(v, 'edit'), d, se])
    a2.axvline(0, color=INK2, lw=1)
    a2.set_yticks(range(len(names)))
    a2.set_yticklabels([l for _, l in names][::-1])
    a2.set_ylim(-0.6, len(names) - 0.3)
    a2.set_xlim(-9, 6)
    a2.set_xlabel('Edit − original, points (± 2 SE, paired)')
    style(a2, grid='x')
    fig.suptitle('9B Blind Fact-Edit: sealed S by route, and general-capability regression', x=0.01, ha='left',
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, 'fig_qcf20', rows, ['panel', 'item', 'noedit_or_orig', 'edit', 'diff_pts', 'se_pts'])


# ---------------------------------------------------------------- fig 5: loops
def fig_loops():
    off_tab = req(load(DATA['scan_off']), 'table')

    def items(path):
        out = {}
        for ln in Path(path).read_text().splitlines():
            r = json.loads(ln)
            for k in ('task', 'correct', 'finish', 'labels'):
                req(r, k)
            out.setdefault(r['task'], []).append(r)
        return out

    off, on = items(DATA['scan_off_items']), items(DATA['scan_on_items'])
    if set(off) != set(on):
        raise ValueError(f'task mismatch: {set(off) ^ set(on)}')

    def stats(rs):
        tr = [r for r in rs if r['finish'] != 'eos']
        lp = [r for r in tr if 'loop' in r['labels']]
        return dict(n=len(rs), acc=sum(bool(r['correct']) for r in rs) / len(rs), trunc=len(tr) / len(rs),
                    n_tr=len(tr), n_lp=len(lp), loop=(len(lp) / len(tr)) if tr else None)

    S = {t: (stats(off[t]), stats(on[t])) for t in off}
    for t, (so, _) in S.items():  # cross-check against the off-mode summary table
        if abs(so['acc'] - req(off_tab, t, 'acc')) > 1e-3:
            raise ValueError(f'{t}: item accuracy {so["acc"]} != table {off_tab[t]["acc"]}')
    order = sorted(S, key=lambda t: S[t][1]['acc'] - S[t][0]['acc'])  # biggest drop on top
    pretty = {t: (t.replace('bbh_', 'BBH ').replace('_five_objects', ' (5)').replace('_', ' ')
                  .replace('gsm8k', 'GSM8K').replace('mmlu', 'MMLU').replace('arc c', 'ARC-C')
                  .replace('ifeval', 'IFEval').replace('math', 'MATH')) + f" (n={S[t][0]['n']})" for t in S}

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
    ys = list(range(len(order)))[::-1]
    rows = []
    for y, t in zip(ys, order):
        so, sn = S[t]
        for ax, k in zip(axes, ('acc', 'trunc', 'loop')):
            a, b = so[k], sn[k]
            if a is not None and b is not None:
                ax.plot([a, b], [y, y], color=GRID, lw=2, zorder=1)
            if a is not None:
                ax.plot([a], [y], 'o', ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
            if b is not None:
                ax.plot([b], [y], 'o', ms=7, color=ORANGE, mec=SURFACE, mew=1.5, zorder=3)
        def frac(s):
            return f"{s['n_lp']}/{s['n_tr']}" if s['n_tr'] else '\u2013'
        axes[2].text(1.15, y, f"{frac(so)}", va='center', ha='right', fontsize=7.5, color=INK2,
                     transform=axes[2].get_yaxis_transform())
        axes[2].text(1.18, y, f"{frac(sn)}", va='center', ha='left', fontsize=7.5, color=INK2,
                     transform=axes[2].get_yaxis_transform())
        rows.append([t, so['n'], so['acc'], sn['acc'], so['trunc'], sn['trunc'], so['n_tr'], so['n_lp'],
                     sn['n_tr'], sn['n_lp']])
    titles = ['Accuracy', 'Share of items truncated\n(hit the token limit)',
              'Loops among truncated items\n(right: loops/truncated, off | on)']
    for ax, ti in zip(axes, titles):
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(ti, loc='left', fontsize=9, color=INK)
        style(ax, grid='x')
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([pretty[t] for t in order])
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], ls='none', marker='o', ms=7, color=BLUE, label='thinking off, 1,024 new tokens'),
          Line2D([], [], ls='none', marker='o', ms=7, color=ORANGE, label='thinking on, 8,192 new tokens')]
    fig.legend(handles=hl, loc='upper left', bbox_to_anchor=(0.18, 1.0), ncol=2, fontsize=8.2)
    fig.text(0.01, 0.012, '4B Defect Scan (thinking off and on): greedy decoding, same 1,100 items in both modes. Off mode: '
             f"{sum(S[t][0]['n_lp'] for t in S)} of {sum(S[t][0]['n_tr'] for t in S)} truncations "
             'carry the loop label.', fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 0.93, 0.92))
    save(fig, 'fig_loops', rows, ['task', 'n', 'acc_off', 'acc_on', 'trunc_off', 'trunc_on', 'n_trunc_off',
                                  'n_loop_trunc_off', 'n_trunc_on', 'n_loop_trunc_on'])


if __name__ == '__main__':
    import matplotlib.ticker  # noqa: F401
    fig_ladder()
    fig_steer_dose()
    fig_replicates()
    fig_qcf20()
    fig_loops()
