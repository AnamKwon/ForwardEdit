import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

OUT = Path(__file__).resolve().parent; OUT.mkdir(parents=True, exist_ok=True)
fam = 'NanumBarunGothic' if any(f.name == 'NanumBarunGothic' for f in fm.fontManager.ttflist) else 'NanumGothic'
plt.rcParams.update({'font.family': fam, 'axes.unicode_minus': False, 'font.size': 11,
                     'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9890'
INK, INK2 = '#0b0b0b', '#52514e'

def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='x' if ax.get_xlim() else 'y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)

def hbars(ax, labels, a, b, la, lb, unit='', fmt='{:.1f}'):
    y = range(len(labels)); h = 0.36
    ax.barh([i - h / 2 - 0.02 for i in y], a, h, color=BLUE, label=la)
    ax.barh([i + h / 2 + 0.02 for i in y], b, h, color=ORANGE, label=lb)
    for i, (x1, x2) in enumerate(zip(a, b)):
        ax.text(x1, i - h / 2 - 0.02, ' ' + fmt.format(x1) + unit, va='center', fontsize=9.5, color=INK)
        ax.text(x2, i + h / 2 + 0.02, ' ' + fmt.format(x2) + unit, va='center', fontsize=9.5, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, frameon=False)

# 1. skill tasks: edit vs strongest no-edit prompt (sealed accuracy, %)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
labels = ['2–3자리 덧셈/뺄셈 (0.8B)', 'thinking 루프 수리, IFEval (0.8B)', '세 번째 단어 첫 글자 (0.8B)', '세 번째 단어 첫 글자 (2B)']
hbars(ax, labels, [67.75, 65.6, 24.0, 42.5], [95.5, 70.1, 32.25, 96.5], 'LLM 편집', '가장 강한 무편집 프롬프트 (CoT/thinking)', '%')
ax.set_xlim(0, 112); ax.set_xlabel('sealed 정답률 (%)')
ax.set_title('기술형 과제: sealed 통과 편집도 프롬프트에 진다', loc='left', color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'fig1_skill_vs_prompt.png', dpi=160); plt.close(fig)

# 2. CounterFact 4B: edit vs strongest no-edit route (S)
fig, ax = plt.subplots(figsize=(8.2, 3.8))
labels = ['qcf15 (맥락 포함)', 'qcf16 (맥락 포함, 재현)', 'qcf17 (맥락 없음)']
hbars(ax, labels, [0.922, 0.894, 0.682], [0.760, 0.785, 0.763], 'LLM 편집', '가장 강한 무편집 프롬프트', '', '{:.3f}')
ax.set_xlim(0, 1.08); ax.set_xlabel('CounterFact S (ES·PS·NS 조화평균)')
ax.set_title('지식 재작성(4B): 맥락이 있으면 이기고, 맥락 없이는 못 미친다', loc='left', color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'fig2_counterfact.png', dpi=160); plt.close(fig)

# 3. qcf17 same-footprint comparison
fig, ax = plt.subplots(figsize=(8.2, 3.9))
names = ['원본 (편집 없음)', '같은 38개 뉴런, 원래 값에서 학습', 'LLM 편집 (38개 뉴런)', '같은 38개 뉴런, 편집기 값에서 학습', 'LoRA (96개 행렬 전체)']
vals = [0.092, 0.668, 0.682, 0.884, 0.898]
cols = [GRAY, GRAY, BLUE, AQUA, ORANGE]
ax.barh(range(5), vals, 0.6, color=cols)
for i, v in enumerate(vals): ax.text(v, i, f' {v:.3f}', va='center', fontsize=10, color=INK)
ax.axvline(0.763, color=INK2, ls='--', lw=1.2); ax.text(0.775, 0, '← 최강 프롬프트 .763', fontsize=9.5, color=INK2, va='center', ha='left')
ax.set_yticks(range(5)); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.grid(axis='x', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel('맥락 없는 sealed S (qcf17_4b)')
ax.set_title('H5: LLM은 구조를 고르고, 값은 경사 학습이 더 잘 맞춘다', loc='left', color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'fig3_structure_vs_values.png', dpi=160); plt.close(fig)

# 4. qcf18 GSM reserve
fig, ax = plt.subplots(figsize=(8.2, 3.0))
ax.barh([0], [120], 0.45, color=ORANGE, label='탐색 후보(c1) 측정에 사용 (노출됨)')
ax.barh([0], [60], 0.45, left=[120], color=BLUE, label='최종 후보용으로 남은 문항 (미노출)')
ax.text(60, 0, '120', color='white', ha='center', va='center', fontsize=12); ax.text(150, 0, '60', color='white', ha='center', va='center', fontsize=12)
ax.plot([80, 80], [-0.3, 0.3], color=INK, lw=1.8, ls='--'); ax.text(80, 0.36, '탐색에 쓸 수 있는 한도 80개\n(남은 문항 100개 이상 필요)', fontsize=9.5, color=INK, ha='center', va='bottom')
ax.set_xlim(0, 185); ax.set_ylim(-0.5, 1.05); ax.set_yticks([]); ax.set_xlabel('GSM 보존 문항 (전체 180개 = 탐색 120 + 검증 60)')
for s_ in ('top', 'right', 'left'): ax.spines[s_].set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(0, -0.38), ncol=2, frameon=False, fontsize=9.5)
ax.set_title('qcf18 동결 거부 원인: 최종 확인용 GSM 문항을 탐색에 써 버림', loc='left', color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(OUT / 'fig4_qcf18_reserve.png', dpi=160); plt.close(fig)
print('ok', fam)
