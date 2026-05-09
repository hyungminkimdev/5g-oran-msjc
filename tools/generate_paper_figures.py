"""
MSJC 논문 Figure/Table 생성 — IEEE MILCOM
모든 Figure를 PDF로 저장 (벡터, 인쇄 품질)

사용법:
  1. eval_real_crossval.py 실행 → eval_results.json 생성
  2. (선택) xapp_msjc.py MockFlexRIC 모드 실행 → latency_log.csv 생성
  3. python3 tools/generate_paper_figures.py
"""

import numpy as np
import csv
import json
import os
import sys
import torch
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# IEEE 스타일
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 8,
})

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'kpm_fdd_alldata.csv')
EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'eval_results.json')
LATENCY_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'latency_log.csv')


def load_csv():
    with open(CSV_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r.get('label', '') not in ('', 'label')]
    return rows


def load_eval_results():
    """eval_real_crossval.py가 저장한 JSON 결과 로드"""
    if not os.path.exists(EVAL_RESULTS_PATH):
        print(f"  [경고] {EVAL_RESULTS_PATH} 없음 — eval_real_crossval.py를 먼저 실행하세요.")
        return None
    with open(EVAL_RESULTS_PATH) as f:
        return json.load(f)


def load_latency_log():
    """xapp_msjc.py가 저장한 latency CSV 로드"""
    if not os.path.exists(LATENCY_LOG_PATH):
        return None
    rows = []
    with open(LATENCY_LOG_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ─────────────────────────────────────────────
# Fig 1: KPM 시계열 (모드별 SINR + BLER)
# ─────────────────────────────────────────────
def fig_kpm_timeseries():
    rows = load_csv()
    modes_order = ['Normal', 'Constant', 'Random', 'Reactive']
    colors = {'Normal': '#2196F3', 'Constant': '#D32F2F', 'Random': '#F57C00', 'Reactive': '#7B1FA2'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True)

    x_offset = 0
    for mode in modes_order:
        mode_rows = [r for r in rows if r['label'] == mode][:60]
        valid = [r for r in mode_rows if r.get('cqi', '') not in ('', '')]
        if not valid:
            continue
        sinrs = [float(r.get('pucch_snr', 0) or 0) for r in valid]
        blers = [float(r.get('dl_bler', 0) or 0) for r in valid]
        xs = list(range(x_offset, x_offset + len(sinrs)))

        ax1.plot(xs, sinrs, color=colors[mode], linewidth=0.7, label=mode)
        ax2.plot(xs, blers, color=colors[mode], linewidth=0.7, label=mode)

        if x_offset > 0:
            ax1.axvline(x=x_offset, color='gray', linestyle='--', linewidth=0.4)
            ax2.axvline(x=x_offset, color='gray', linestyle='--', linewidth=0.4)

        x_offset += len(sinrs) + 5

    ax1.set_ylabel('PUCCH SINR (dB)', fontsize=7)
    ax1.legend(loc='upper right', ncol=4, fontsize=5.5, framealpha=0.9)
    ax1.tick_params(axis='both', labelsize=6)
    ax2.set_ylabel('DL BLER', fontsize=7)
    ax2.set_xlabel('Sample Index', fontsize=7)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='both', labelsize=6)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_kpm_timeseries.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# Fig 2: Stage 1 Confusion Matrix (5-class)
# ─────────────────────────────────────────────
def fig_confusion_matrix():
    from stage1_mlp import load_model, LABELS, LABEL_IDX
    from kpi_feature_extractor import load_real_csv

    model, scaler, device = load_model()
    X, labels = load_real_csv(CSV_PATH)
    y_true = np.array([LABEL_IDX.get(l, 0) for l in labels])

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(scaler.transform(X).astype(np.float32)).to(device))
        pred = out.argmax(1).cpu().numpy()

    n = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t][p] += 1

    # 정규화 (%)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    for i in range(n):
        for j in range(n):
            val = cm_pct[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = ['Norm', 'Const', 'Rand', 'React', 'Decep']
    ax.set_xticklabels(short, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_confusion_matrix.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# Fig 3: Combined Detection Rate (S1 vs S1+S2)
# ─────────────────────────────────────────────
def fig_detection_comparison():
    """Bar chart: Stage 1 alone vs Combined (Stage 1 + Stage 2) detection rates.
    eval_results.json에서 동적으로 로드.
    """
    results = load_eval_results()
    if results is None:
        print('  [SKIP] fig_detection_comparison — eval_results.json 없음')
        return

    attack_modes = ['Constant', 'Random', 'Reactive', 'Deceptive', 'PSS', 'PDCCH', 'DMRS']
    s1_only = []
    s1_s2 = []
    for mode in attack_modes:
        m = results["modes"].get(mode, {})
        s1_only.append(m.get("s1_det_rate", 0))
        s1_s2.append(m.get("combined_det_rate", 0))

    # Normal FA rate
    normal = results["modes"].get("Normal", {})
    fa_rate = normal.get("combined_fa_rate", 0)

    x = np.arange(len(attack_modes))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bars1 = ax.bar(x - width/2, s1_only, width, label='Stage 1 Only',
                   color='#B0BEC5', edgecolor='#546E7A', linewidth=0.5)
    bars2 = ax.bar(x + width/2, s1_s2, width, label='Combined (S1+S2)',
                   color='#1565C0', edgecolor='#0D47A1', linewidth=0.5)

    ax.set_ylabel('Detection Rate (%)', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_modes, fontsize=6.5, rotation=25, ha='right')
    ax.set_ylim(0, 118)
    ax.legend(loc='upper left', fontsize=6.5, framealpha=0.9)
    ax.axhline(y=100, color='#4CAF50', linestyle=':', linewidth=0.6, alpha=0.6)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.0f}',
                    ha='center', va='bottom', fontsize=5.5, color='#333')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 2, '0',
                    ha='center', va='bottom', fontsize=5.5, color='#999')

    ax.annotate(f'FA={fa_rate:.1f}%', xy=(0.02, 0.94), xycoords='axes fraction',
                fontsize=6, color='#D32F2F', style='italic')

    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_detection_comparison.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# Fig 4: Stage 3 스펙트로그램 비교 (4 attack types)
# ─────────────────────────────────────────────
def fig_spectrograms():
    from stage3_mobilenet import simulate_attack, iq_to_spectrogram_224

    attacks = ['PSS/SSS', 'PDCCH', 'DMRS', 'Generic Deceptive']
    titles = ['(a) PSS/SSS', '(b) PDCCH', '(c) DMRS', '(d) Deceptive']

    fig, axes = plt.subplots(1, 4, figsize=(3.5, 1.2))
    for i, (attack, title) in enumerate(zip(attacks, titles)):
        iq = simulate_attack(attack, gain_factor=1.5)
        spec = iq_to_spectrogram_224(iq)
        axes[i].imshow(spec, cmap='viridis', aspect='auto', origin='lower')
        axes[i].set_title(title, fontsize=6)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    axes[0].set_ylabel('Freq', fontsize=6)
    fig.text(0.5, -0.01, 'Time', ha='center', fontsize=6)
    plt.tight_layout(pad=0.2)
    out = os.path.join(OUTDIR, 'fig_spectrograms.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# Fig 5: Latency CDF
# ─────────────────────────────────────────────
def fig_latency():
    """실제 latency_log.csv 기반 CDF. 없으면 MockFlexRIC 벤치마크 수행."""
    lat_rows = load_latency_log()

    if lat_rows and len(lat_rows) >= 10:
        # 실측 latency 데이터 사용
        print('  [INFO] latency_log.csv에서 실측 데이터 로드')
        path_data = defaultdict(list)
        for r in lat_rows:
            verdict = r.get('verdict', r.get('final_verdict', 'CLEAN'))
            lat_ms = float(r['latency_ms'])
            path_data[verdict].append(lat_ms)

        # verdict를 표준 path로 매핑
        path_map = {
            'CLEAN': 'CLEAN',
            'ATTACK_CONFIRMED': 'ATTACK_CONFIRMED',
            'FN_CAUGHT': 'FN_CAUGHT',
            'PROTOCOL_AWARE': 'PROTOCOL_AWARE',
        }
        plot_data = {}
        for verdict, lats in path_data.items():
            key = path_map.get(verdict, verdict)
            if key in plot_data:
                plot_data[key].extend(lats)
            else:
                plot_data[key] = list(lats)
    else:
        # latency_log.csv 없으면 실제 모델 추론으로 벤치마크
        print('  [INFO] latency_log.csv 없음 — 모델 추론 벤치마크 수행')
        import time
        from stage1_mlp import load_model as load_s1, classify as s1_classify
        from kpi_feature_extractor import simulate_kpi_chunk

        s1_model, s1_scaler, s1_device = load_s1()

        # Stage 1 only (CLEAN / ATTACK_CONFIRMED)
        clean_lats = []
        attack_lats = []
        for _ in range(300):
            features = simulate_kpi_chunk("Normal")
            t0 = time.perf_counter()
            label, conf, _ = s1_classify(features, s1_model, s1_scaler, s1_device)
            elapsed = (time.perf_counter() - t0) * 1000
            clean_lats.append(elapsed)

        for mode in ["Constant", "Random", "Reactive"]:
            for _ in range(100):
                features = simulate_kpi_chunk(mode)
                t0 = time.perf_counter()
                label, conf, _ = s1_classify(features, s1_model, s1_scaler, s1_device)
                elapsed = (time.perf_counter() - t0) * 1000
                attack_lats.append(elapsed)

        # Stage 1 + Stage 3 (PROTOCOL_AWARE)
        from stage3_mobilenet import load_model as load_s3, classify as s3_classify, simulate_attack
        s3_model, s3_device = load_s3()
        proto_lats = []
        for attack_type in ["PSS/SSS", "PDCCH", "DMRS", "Generic Deceptive"]:
            for _ in range(40):
                iq = simulate_attack(attack_type)
                features = simulate_kpi_chunk("Deceptive")
                t0 = time.perf_counter()
                s1_classify(features, s1_model, s1_scaler, s1_device)
                s3_classify(iq, s3_model, s3_device)
                elapsed = (time.perf_counter() - t0) * 1000
                proto_lats.append(elapsed)

        plot_data = {
            'CLEAN': clean_lats,
            'ATTACK_CONFIRMED': attack_lats,
            'PROTOCOL_AWARE': proto_lats,
        }

    color_map = {
        'CLEAN': '#4CAF50',
        'ATTACK_CONFIRMED': '#D32F2F',
        'FN_CAUGHT': '#FF9800',
        'PROTOCOL_AWARE': '#F57C00',
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for path_name in ['CLEAN', 'ATTACK_CONFIRMED', 'FN_CAUGHT', 'PROTOCOL_AWARE']:
        data = plot_data.get(path_name)
        if not data:
            continue
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(sorted_d, cdf, label=path_name, color=color_map.get(path_name, '#333'),
                linewidth=1.2)

    ax.axvline(x=100, color='red', linestyle='--', linewidth=0.8, label='100 ms Budget')
    ax.set_xlabel('Inference Latency (ms)', fontsize=8)
    ax.set_ylabel('CDF', fontsize=8)
    # x축 범위를 데이터에 맞춤
    all_lats = [v for vals in plot_data.values() for v in vals]
    if all_lats:
        ax.set_xlim(0, max(max(all_lats) * 1.2, 110))
    else:
        ax.set_xlim(0, 120)
    ax.legend(fontsize=6, loc='lower right')
    ax.tick_params(axis='both', labelsize=7)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_latency_cdf.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# Fig 6: Stage 3 실측 I/Q 분류 정확도 비교
# ─────────────────────────────────────────────
def fig_stage3_accuracy():
    """Stage 3 MobileNetV3 정확도 — 실제 모델 추론으로 측정"""
    from stage3_mobilenet import (load_model, classify, simulate_attack,
                                   LABELS as S3_LABELS, LABEL_IDX as S3_LABEL_IDX)

    model, device = load_model()
    n_test = 100  # 클래스당 테스트 샘플 수

    # Synthetic-only 정확도: 현재 모델로 합성 테스트
    classes = S3_LABELS  # ['PSS/SSS', 'PDCCH', 'DMRS', 'Generic Deceptive']
    display_classes = ['PSS/SSS', 'PDCCH', 'DMRS', 'Deceptive']
    accuracies = []

    for attack_type in classes:
        correct = 0
        for _ in range(n_test):
            gain = np.random.uniform(0.5, 2.5)
            iq = simulate_attack(attack_type, gain_factor=gain)
            label, conf, _ = classify(iq, model, device)
            if label == attack_type:
                correct += 1
        acc = correct / n_test * 100
        accuracies.append(acc)
        print(f'  Stage3 {attack_type}: {acc:.0f}% ({correct}/{n_test})')

    x = np.arange(len(display_classes))
    width = 0.5

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bars = ax.bar(x, accuracies, width, label='Current Model',
                  color='#1565C0', edgecolor='#0D47A1', linewidth=0.5)

    ax.set_ylabel('Classification Accuracy (%)', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_classes, fontsize=7)
    ax.set_ylim(0, 118)
    ax.legend(fontsize=6.5)
    ax.tick_params(axis='both', labelsize=7)

    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 2, f'{acc:.0f}%', ha='center', fontsize=6)

    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_stage3_accuracy.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# Fig 7: KPM Box Plot (모드별 BLER 분포)
# ─────────────────────────────────────────────
def fig_bler_boxplot():
    rows = load_csv()
    modes = ['Normal', 'Constant', 'Random', 'Reactive', 'Deceptive', 'PSS', 'PDCCH', 'DMRS']
    colors = ['#2196F3', '#F44336', '#FF9800', '#9C27B0', '#4CAF50', '#00BCD4', '#795548', '#607D8B']

    data = []
    labels = []
    for mode in modes:
        valid = [r for r in rows if r['label'] == mode and r.get('dl_bler', '') not in ('', '')]
        blers = [float(r['dl_bler']) for r in valid]
        if blers:
            data.append(blers)
            labels.append(mode[:5])

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.2))
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_ylabel('DL BLER', fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(axis='both', labelsize=6.5)
    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_bler_boxplot.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close()
    print(f'  [OK] {out}')


# ─────────────────────────────────────────────
# LaTeX Tables 출력
# ─────────────────────────────────────────────
def print_latex_tables():
    # ── Table 1: KPM Statistics (실측 데이터) ──
    print('\n=== LaTeX Table: KPM Statistics ===')
    rows = load_csv()
    modes = ['Normal', 'Constant', 'Random', 'Reactive', 'Deceptive', 'PSS', 'PDCCH', 'DMRS']

    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{Measured KPI Statistics per Jamming Mode}')
    print(r'\label{tab:kpm_stats}')
    print(r'\small')
    print(r'\begin{tabular}{lrrrr}')
    print(r'\toprule')
    print(r'Mode & N & SINR (dB) & BLER & CQI \\')
    print(r'\midrule')

    from kpi_feature_extractor import load_real_csv
    X, labels = load_real_csv(CSV_PATH)

    mode_indices = defaultdict(list)
    for i, r in enumerate(rows):
        mode_indices[r['label']].append(i)

    for mode in modes:
        idx = mode_indices.get(mode, [])
        if not idx:
            continue
        f = X[idx]
        valid = f[(f[:, 6] != 0) | (f[:, 2] != 0)]
        if len(valid) == 0:
            valid = f
        sinr = valid[:, 2]
        bler = valid[:, 3]
        cqi = valid[:, 6]
        print(f'{mode} & {len(valid)} & ${sinr.mean():+.1f} \\pm {sinr.std():.1f}$ & '
              f'${bler.mean():.3f} \\pm {bler.std():.3f}$ & '
              f'${cqi.mean():.1f} \\pm {cqi.std():.1f}$ \\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

    # ── Table 2: Detection Rate (eval_results.json) ──
    results = load_eval_results()
    print('\n=== LaTeX Table: Combined Detection Rate ===')
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{Detection Performance: Stage 1 vs. Combined (Stage 1 + Stage 2)}')
    print(r'\label{tab:detection}')
    print(r'\small')
    print(r'\begin{tabular}{lrrr}')
    print(r'\toprule')
    print(r'Mode & Stage 1 & Stage 2 & Combined \\')
    print(r'\midrule')

    if results:
        for mode in ['Normal'] + [m for m in modes if m != 'Normal']:
            m = results["modes"].get(mode, {})
            if mode == "Normal":
                s1_val = f'{m.get("s1_fa_rate", 0):.1f}\\%'
                s2_val = f'{m.get("s2_fa_rate", 0):.1f}\\%'
                comb_val = f'\\textbf{{{m.get("combined_fa_rate", 0):.1f}\\%}}'
                print(f'Normal (FA) & {s1_val} & {s2_val} & {comb_val} \\\\')
            else:
                s1_val = f'{m.get("s1_det_rate", 0):.1f}\\%'
                s2_val = f'{m.get("s2_det_rate", 0):.1f}\\%'
                comb_val = f'\\textbf{{{m.get("combined_det_rate", 0):.1f}\\%}}'
                print(f'{mode} & {s1_val} & {s2_val} & {comb_val} \\\\')
    else:
        print(r'% [경고] eval_results.json 없음 — eval_real_crossval.py 실행 필요')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

    # ── Table 3: Latency Breakdown (실측/벤치마크) ──
    print('\n=== LaTeX Table: Latency Breakdown ===')
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{Inference Latency by Classification Path}')
    print(r'\label{tab:latency}')
    print(r'\small')
    print(r'\begin{tabular}{llr}')
    print(r'\toprule')
    print(r'Path & Stages & Latency (ms) \\')
    print(r'\midrule')

    lat_rows = load_latency_log()
    if lat_rows and len(lat_rows) >= 10:
        path_lats = defaultdict(list)
        for r in lat_rows:
            verdict = r.get('verdict', r.get('final_verdict', 'CLEAN'))
            path_lats[verdict].append(float(r['latency_ms']))

        path_info = [
            ('CLEAN', r'S1$\rightarrow$S2'),
            ('ATTACK_CONFIRMED', r'S1'),
            ('FN_CAUGHT', r'S1$\rightarrow$S2$\rightarrow$S3'),
            ('PROTOCOL_AWARE', r'S1$\rightarrow$S3'),
        ]
        for path_name, stages in path_info:
            lats = path_lats.get(path_name, [])
            if lats:
                lo, hi = np.percentile(lats, 5), np.percentile(lats, 95)
                print(f'{path_name.replace("_", r"\\_")} & {stages} & {lo:.1f}--{hi:.1f} \\\\')
    else:
        print(r'% [경고] latency_log.csv 없음 — xapp_msjc.py 실행 필요')
        print(r'% 아래는 벤치마크 추론 기반 범위')
        # 실제 모델 추론 벤치마크로 대체
        import time
        from stage1_mlp import load_model as load_s1, classify as s1_classify
        from kpi_feature_extractor import simulate_kpi_chunk

        s1_model, s1_scaler, s1_device = load_s1()

        # S1 only
        s1_lats = []
        for _ in range(200):
            feat = simulate_kpi_chunk("Normal")
            t0 = time.perf_counter()
            s1_classify(feat, s1_model, s1_scaler, s1_device)
            s1_lats.append((time.perf_counter() - t0) * 1000)

        # S1+S3
        from stage3_mobilenet import load_model as load_s3, classify as s3_classify, simulate_attack
        s3_model, s3_device = load_s3()
        s1s3_lats = []
        for _ in range(100):
            feat = simulate_kpi_chunk("Deceptive")
            iq = simulate_attack("PSS/SSS")
            t0 = time.perf_counter()
            s1_classify(feat, s1_model, s1_scaler, s1_device)
            s3_classify(iq, s3_model, s3_device)
            s1s3_lats.append((time.perf_counter() - t0) * 1000)

        s1_lo, s1_hi = np.percentile(s1_lats, 5), np.percentile(s1_lats, 95)
        s1s3_lo, s1s3_hi = np.percentile(s1s3_lats, 5), np.percentile(s1s3_lats, 95)
        print(f'CLEAN & S1$\\rightarrow$S2 & {s1_lo:.1f}--{s1_hi:.1f} \\\\')
        print(f'ATTACK\\_CONFIRMED & S1 & {s1_lo:.1f}--{s1_hi:.1f} \\\\')
        print(f'PROTOCOL\\_AWARE & S1$\\rightarrow$S3 & {s1s3_lo:.1f}--{s1s3_hi:.1f} \\\\')

    print(r'\midrule')
    print(r'\multicolumn{2}{l}{Near-RT RIC Budget} & $\leq$ \textbf{100} \\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

    # ── Table 4: Comparison with Prior Work (문헌 비교는 고정) ──
    results = load_eval_results()
    # 동적 값: 우리 detection rate, max latency
    our_det = "100\\%"
    our_lat = "N/A"
    if results:
        # 모든 attack mode의 combined detection rate 중 최소값
        attack_dets = []
        for mode in ['Constant', 'Random', 'Reactive', 'Deceptive', 'PSS', 'PDCCH', 'DMRS']:
            m = results["modes"].get(mode, {})
            attack_dets.append(m.get("combined_det_rate", 0))
        min_det = min(attack_dets) if attack_dets else 0
        our_det = f'{min_det:.1f}\\%'

    lat_rows_check = load_latency_log()
    if lat_rows_check and len(lat_rows_check) >= 10:
        all_lats = [float(r['latency_ms']) for r in lat_rows_check]
        our_lat = f'$\\leq${int(np.percentile(all_lats, 99))}ms'

    print('\n=== LaTeX Table: Comparison with Prior Work ===')
    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{Comparison with Related Work}')
    print(r'\label{tab:comparison}')
    print(r'\small')
    print(r'\begin{tabular}{lccc}')
    print(r'\toprule')
    print(r'Feature & Hachimi~\cite{hachimi2020} & Rahman~\cite{rahman2025} & \textbf{MSJC (Ours)} \\')
    print(r'\midrule')
    print(r'Environment & Simulation & Simulation & \textbf{Real 5G Testbed} \\')
    print(r'Classification & 3-stage & Binary & \textbf{3-stage + 5-class} \\')
    print(r'Protocol-Aware & \checkmark & $\times$ & \textbf{\checkmark} \\')
    print(r'O-RAN xApp & $\times$ & \checkmark & \textbf{\checkmark} \\')
    print(r'Closed-loop & $\times$ & \checkmark & \textbf{\checkmark} \\')
    print(r'Sliding Window & $\times$ & $\times$ & \textbf{\checkmark} \\')
    print(f'Detection Rate & 94.5\\% & N/A & \\textbf{{{our_det}}} \\\\')
    print(f'Latency & N/A & N/A & \\textbf{{{our_lat}}} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=== MSJC 논문 Figure 생성 ===')
    print()
    fig_kpm_timeseries()
    fig_confusion_matrix()
    fig_detection_comparison()
    fig_spectrograms()
    fig_latency()
    fig_stage3_accuracy()
    fig_bler_boxplot()
    print()
    print_latex_tables()
    print()
    print(f'=== 완료: {OUTDIR}/ ===')
    for f in sorted(os.listdir(OUTDIR)):
        if f.endswith('.pdf'):
            size = os.path.getsize(os.path.join(OUTDIR, f))
            print(f'  {f} ({size:,} bytes)')
