import csv
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data/raw/sleep_memory_2x2.csv'
REPORT_PATH = ROOT / 'outputs/reports/apa7_sleep_memory_report.md'
DESC_PATH = ROOT / 'outputs/results/sleep_memory_descriptives.csv'
STATS_PATH = ROOT / 'outputs/results/sleep_memory_anova_results.txt'
FIG_PATH = ROOT / 'outputs/figures/sleep_memory_interaction.svg'
PDF_PATH = ROOT / 'outputs/reports/apa7_sleep_memory_report.pdf'

# ----- Numeric helpers (no third-party dependencies) -----

def betacf(a: float, b: float, x: float) -> float:
    MAXIT = 200
    EPS = 3.0e-14
    FPMIN = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delh = d * c
        h *= delh
        if abs(delh - 1.0) <= EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta)

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def f_sf(f: float, d1: float, d2: float) -> float:
    # Survival function P(F >= f)
    if f <= 0:
        return 1.0
    x = (d1 * f) / (d1 * f + d2)
    cdf = betai(d1 / 2.0, d2 / 2.0, x)
    return max(0.0, min(1.0, 1.0 - cdf))


def t_two_tailed_p(t: float, df: float) -> float:
    # t^2 ~ F(1, df)
    return f_sf(t * t, 1.0, df)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return '< .001'
    return f'= {p:.3f}'.replace('0.', '.')


# ----- Load data -----
rows = []
with DATA_PATH.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'id': int(r['id']),
            'sleep': r['sleep'],
            'cue': r['cue'],
            'recall_score': float(r['recall_score']),
        })

scores = [r['recall_score'] for r in rows]
grand_mean = mean(scores)

# Grouping
cells = {}
by_sleep = {'Sleep': [], 'Wake': []}
by_cue = {'TMR': [], 'Control': []}

for r in rows:
    key = (r['sleep'], r['cue'])
    cells.setdefault(key, []).append(r['recall_score'])
    by_sleep[r['sleep']].append(r['recall_score'])
    by_cue[r['cue']].append(r['recall_score'])

# Descriptives
cell_desc = []
for sleep in ['Sleep', 'Wake']:
    for cue in ['TMR', 'Control']:
        vals = cells[(sleep, cue)]
        n = len(vals)
        m = mean(vals)
        sd = stdev(vals)
        se = sd / math.sqrt(n)
        cell_desc.append((sleep, cue, n, m, sd, se))

# 2x2 ANOVA components
# SS_total
ss_total = sum((x - grand_mean) ** 2 for x in scores)

# SS_sleep (A)
n_per_sleep = len(by_sleep['Sleep'])
ss_sleep = sum(len(by_sleep[a]) * (mean(by_sleep[a]) - grand_mean) ** 2 for a in by_sleep)

# SS_cue (B)
ss_cue = sum(len(by_cue[b]) * (mean(by_cue[b]) - grand_mean) ** 2 for b in by_cue)

# SS_interaction (balanced or unbalanced general)
ss_cells = 0.0
for (a, b), vals in cells.items():
    m_ab = mean(vals)
    m_a = mean(by_sleep[a])
    m_b = mean(by_cue[b])
    ss_cells += len(vals) * (m_ab - m_a - m_b + grand_mean) ** 2
ss_interaction = ss_cells

# SS_error
ss_error = ss_total - ss_sleep - ss_cue - ss_interaction

# dfs
df_sleep = 1
df_cue = 1
df_interaction = 1
df_error = len(rows) - 4

# MS/F
ms_error = ss_error / df_error
ms_sleep = ss_sleep / df_sleep
ms_cue = ss_cue / df_cue
ms_interaction = ss_interaction / df_interaction

f_sleep = ms_sleep / ms_error
f_cue = ms_cue / ms_error
f_interaction = ms_interaction / ms_error

p_sleep = f_sf(f_sleep, df_sleep, df_error)
p_cue = f_sf(f_cue, df_cue, df_error)
p_interaction = f_sf(f_interaction, df_interaction, df_error)

eta_sleep = ss_sleep / (ss_sleep + ss_error)
eta_cue = ss_cue / (ss_cue + ss_error)
eta_interaction = ss_interaction / (ss_interaction + ss_error)

# Simple effects t-tests (equal variances)
def indep_t(g1, g2):
    n1, n2 = len(g1), len(g2)
    m1, m2 = mean(g1), mean(g2)
    s1, s2 = stdev(g1), stdev(g2)
    sp2 = (((n1 - 1) * s1 * s1) + ((n2 - 1) * s2 * s2)) / (n1 + n2 - 2)
    se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    t = (m1 - m2) / se
    df = n1 + n2 - 2
    p = t_two_tailed_p(abs(t), df)
    return t, df, p

sleep_t, sleep_df, sleep_p = indep_t(cells[('Sleep', 'TMR')], cells[('Sleep', 'Control')])
wake_t, wake_df, wake_p = indep_t(cells[('Wake', 'TMR')], cells[('Wake', 'Control')])
tmr_t, tmr_df, tmr_p = indep_t(cells[('Sleep', 'TMR')], cells[('Wake', 'TMR')])
control_t, control_df, control_p = indep_t(cells[('Sleep', 'Control')], cells[('Wake', 'Control')])

# Save descriptives CSV
with DESC_PATH.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sleep', 'cue', 'n', 'mean', 'sd', 'se'])
    for row in cell_desc:
        w.writerow([row[0], row[1], row[2], f"{row[3]:.4f}", f"{row[4]:.4f}", f"{row[5]:.4f}"])

# Save stats text
with STATS_PATH.open('w') as f:
    f.write('Two-way between-subjects ANOVA (2x2)\n')
    f.write(f'SS_sleep={ss_sleep:.6f}, df=1, MS={ms_sleep:.6f}, F={f_sleep:.6f}, p={p_sleep:.8f}, pes={eta_sleep:.6f}\n')
    f.write(f'SS_cue={ss_cue:.6f}, df=1, MS={ms_cue:.6f}, F={f_cue:.6f}, p={p_cue:.8f}, pes={eta_cue:.6f}\n')
    f.write(f'SS_interaction={ss_interaction:.6f}, df=1, MS={ms_interaction:.6f}, F={f_interaction:.6f}, p={p_interaction:.8f}, pes={eta_interaction:.6f}\n')
    f.write(f'SS_error={ss_error:.6f}, df={df_error}, MS={ms_error:.6f}\n')
    f.write('\nSimple effects independent t tests\n')
    f.write(f'Sleep: TMR vs Control: t({sleep_df})={sleep_t:.6f}, p={sleep_p:.8f}\n')
    f.write(f'Wake: TMR vs Control: t({wake_df})={wake_t:.6f}, p={wake_p:.8f}\n')
    f.write(f'TMR: Sleep vs Wake: t({tmr_df})={tmr_t:.6f}, p={tmr_p:.8f}\n')
    f.write(f'Control: Sleep vs Wake: t({control_df})={control_t:.6f}, p={control_p:.8f}\n')

# Build simple SVG interaction plot
# Coordinates
width, height = 700, 460
margin_l, margin_r, margin_t, margin_b = 90, 40, 50, 80
plot_w = width - margin_l - margin_r
plot_h = height - margin_t - margin_b

all_vals = scores
y_min = math.floor(min(all_vals) - 1)
y_max = math.ceil(max(all_vals) + 1)


def x_pos(cue):
    return margin_l + (0 if cue == 'Control' else plot_w)


def y_pos(v):
    return margin_t + (y_max - v) / (y_max - y_min) * plot_h

means = {(s, c): mean(cells[(s, c)]) for s in ['Sleep', 'Wake'] for c in ['Control', 'TMR']}

sleep_pts = [(x_pos('Control'), y_pos(means[('Sleep', 'Control')])), (x_pos('TMR'), y_pos(means[('Sleep', 'TMR')]))]
wake_pts = [(x_pos('Control'), y_pos(means[('Wake', 'Control')])), (x_pos('TMR'), y_pos(means[('Wake', 'TMR')]))]

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
svg.append('<style>text{font-family:Arial,sans-serif} .axis{stroke:#111;stroke-width:2} .grid{stroke:#ddd;stroke-width:1} .sleep{stroke:#1f77b4;fill:#1f77b4} .wake{stroke:#d62728;fill:#d62728}</style>')
svg.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20">Interaction of Sleep and Cue on Recall Score</text>')

# grid + y ticks
for yv in range(y_min, y_max + 1, 2):
    y = y_pos(yv)
    svg.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}"/>')
    svg.append(f'<text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{yv}</text>')

# axes
svg.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}"/>')
svg.append(f'<line class="axis" x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}"/>')

# x labels
for cue in ['Control', 'TMR']:
    x = x_pos(cue)
    svg.append(f'<text x="{x}" y="{height-margin_b+30}" text-anchor="middle" font-size="13">{cue}</text>')

# axis titles
svg.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14">Cue Condition</text>')
svg.append(f'<text x="25" y="{height/2}" transform="rotate(-90,25,{height/2})" text-anchor="middle" font-size="14">Recall Score</text>')

# lines
svg.append(f'<polyline class="sleep" fill="none" stroke-width="3" points="{sleep_pts[0][0]:.1f},{sleep_pts[0][1]:.1f} {sleep_pts[1][0]:.1f},{sleep_pts[1][1]:.1f}"/>')
svg.append(f'<polyline class="wake" fill="none" stroke-width="3" points="{wake_pts[0][0]:.1f},{wake_pts[0][1]:.1f} {wake_pts[1][0]:.1f},{wake_pts[1][1]:.1f}"/>')

# markers
for x, y in sleep_pts:
    svg.append(f'<circle class="sleep" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
for x, y in wake_pts:
    svg.append(f'<rect class="wake" x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10"/>')

# legend
lx, ly = width - 190, margin_t + 10
svg.append(f'<line class="sleep" x1="{lx}" y1="{ly}" x2="{lx+30}" y2="{ly}" stroke-width="3"/>')
svg.append(f'<circle class="sleep" cx="{lx+15}" cy="{ly}" r="5"/>')
svg.append(f'<text x="{lx+40}" y="{ly+4}" font-size="13">Sleep</text>')
svg.append(f'<line class="wake" x1="{lx}" y1="{ly+24}" x2="{lx+30}" y2="{ly+24}" stroke-width="3"/>')
svg.append(f'<rect class="wake" x="{lx+10}" y="{ly+19}" width="10" height="10"/>')
svg.append(f'<text x="{lx+40}" y="{ly+28}" font-size="13">Wake</text>')

svg.append('</svg>')
FIG_PATH.write_text('\n'.join(svg))

def _pdf_escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def write_simple_pdf(path: Path, lines):
    # Minimal one-page PDF writer using Helvetica.
    content = ["BT", "/F1 11 Tf"]
    y = 790
    for line in lines:
        content.append(f"1 0 0 1 50 {y} Tm ({_pdf_escape(line)}) Tj")
        y -= 14
        if y < 60:
            break
    content.append("ET")
    stream = ("\n".join(content) + "\n").encode('latin-1', errors='replace')

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append((f"5 0 obj << /Length {len(stream)} >> stream\n".encode('ascii') + stream + b"endstream endobj\n"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode('ascii'))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode('ascii'))
    pdf.extend((
        f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode('ascii'))
    path.write_bytes(pdf)


# Report markdown APA 7 style structure
lines = []
lines.append('# Effects of Sleep and Targeted Memory Reactivation on Recall Performance')
lines.append('')
lines.append('[Download PDF version](./apa7_sleep_memory_report.pdf)')
lines.append('')
lines.append('## Abstract')
lines.append('This report presents a 2 × 2 between-subjects analysis of variance (ANOVA) examining memory recall as a function of sleep condition (Sleep vs. Wake) and cue condition (TMR vs. Control). Significant main effects of sleep and cue condition were observed, indicating higher recall after sleep and under TMR. The Sleep × Cue interaction was not statistically significant. Descriptive statistics and follow-up independent-samples comparisons are reported to support interpretation.')
lines.append('')
lines.append('## Introduction')
lines.append('Sleep is often associated with improved consolidation of recently learned information. Targeted memory reactivation (TMR) has been proposed as an intervention that can further enhance consolidation by presenting memory-related cues. The current analysis evaluates whether recall differs as a function of sleep state, cue condition, and their interaction in a 2 × 2 factorial design.')
lines.append('')
lines.append('## Methods')
lines.append('### Design and Data')
lines.append('The dataset included 80 participants distributed evenly across four between-subjects cells (Sleep–TMR, Sleep–Control, Wake–TMR, Wake–Control; n = 20 per cell). The outcome variable was recall score.')
lines.append('')
lines.append('### Statistical Analysis')
lines.append('A two-way between-subjects ANOVA tested main effects of sleep and cue condition and their interaction. Follow-up independent-samples *t* tests compared cue effects within sleep levels and sleep effects within cue levels. Partial η² is reported as effect size for ANOVA terms.')
lines.append('')
lines.append('## Results')
lines.append(f'The main effect of sleep was significant, *F*({df_sleep}, {df_error}) = {f_sleep:.2f}, *p* {fmt_p(p_sleep)}, partial η² = {eta_sleep:.3f}. The main effect of cue was significant, *F*({df_cue}, {df_error}) = {f_cue:.2f}, *p* {fmt_p(p_cue)}, partial η² = {eta_cue:.3f}. The Sleep × Cue interaction was not statistically significant, *F*({df_interaction}, {df_error}) = {f_interaction:.2f}, *p* {fmt_p(p_interaction)}, partial η² = {eta_interaction:.3f}.')
lines.append('')
lines.append('Follow-up independent-samples comparisons showed that TMR outperformed control cueing in the Sleep condition, '
             f'*t*({sleep_df}) = {sleep_t:.2f}, *p* {fmt_p(sleep_p)}, and in the Wake condition, *t*({wake_df}) = {wake_t:.2f}, *p* {fmt_p(wake_p)}. Sleep also outperformed Wake both under TMR, *t*({tmr_df}) = {tmr_t:.2f}, *p* {fmt_p(tmr_p)}, and under Control, *t*({control_df}) = {control_t:.2f}, *p* {fmt_p(control_p)}.')
lines.append('')
lines.append('### Descriptive Statistics')
lines.append('')
lines.append('| Sleep | Cue | *n* | *M* | *SD* | *SE* |')
lines.append('|---|---|---:|---:|---:|---:|')
for sleep, cue, n, m, sd, se in cell_desc:
    lines.append(f'| {sleep} | {cue} | {n} | {m:.2f} | {sd:.2f} | {se:.2f} |')
lines.append('')
lines.append('### Interaction Graph')
lines.append('')
lines.append('![Interaction graph for Sleep × Cue effects on recall](../figures/sleep_memory_interaction.svg)')
lines.append('')
lines.append('## Conclusion')
lines.append('Recall performance was higher in Sleep than Wake and higher in TMR than Control, indicating reliable main effects of sleep and cueing. However, there was no evidence that the cueing effect differed by sleep state (non-significant interaction). These findings support additive benefits of sleep and TMR in this sample rather than a multiplicative interaction.')

REPORT_PATH.write_text('\n'.join(lines))

pdf_lines = [
    'Effects of Sleep and Targeted Memory Reactivation on Recall Performance',
    '',
    'Abstract: 2x2 ANOVA on recall with Sleep (Sleep/Wake) and Cue (TMR/Control).',
    'Main effects: Sleep F(1,76) = %.2f, p %s, partial eta^2 = %.3f.' % (f_sleep, fmt_p(p_sleep), eta_sleep),
    'Main effects: Cue F(1,76) = %.2f, p %s, partial eta^2 = %.3f.' % (f_cue, fmt_p(p_cue), eta_cue),
    'Interaction: F(1,76) = %.2f, p %s, partial eta^2 = %.3f (not significant).' % (f_interaction, fmt_p(p_interaction), eta_interaction),
    'Cell means: Sleep-TMR %.2f, Sleep-Control %.2f, Wake-TMR %.2f, Wake-Control %.2f.' % (
        mean(cells[('Sleep', 'TMR')]), mean(cells[('Sleep', 'Control')]),
        mean(cells[('Wake', 'TMR')]), mean(cells[('Wake', 'Control')])
    ),
    'See markdown report for full introduction, methods, results table, and interaction graph.',
]
write_simple_pdf(PDF_PATH, pdf_lines)

print('Analysis complete')
print(f'ANOVA interaction p = {p_interaction:.6f}')
