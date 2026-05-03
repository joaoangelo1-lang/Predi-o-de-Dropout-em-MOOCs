"""
OULAD — Listagem de features por grupo + Heatmap de correlação
com corte temporal individual por aluno.
"""

import warnings
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ── Configurações ──────────────────────────────────────────────────
path        = kagglehub.dataset_download("rocki37/open-university-learning-analytics-dataset")
DATA_DIR    = path
CUTOFF_BASE = 28
CHAVE       = ['id_student', 'code_module', 'code_presentation']



student_info       = pd.read_csv(f'{DATA_DIR}/studentInfo.csv')
student_assessment = pd.read_csv(f'{DATA_DIR}/studentAssessment.csv')
assessments        = pd.read_csv(f'{DATA_DIR}/assessments.csv')
student_vle        = pd.read_csv(f'{DATA_DIR}/studentVle.csv')
vle                = pd.read_csv(f'{DATA_DIR}/vle.csv')
student_reg        = pd.read_csv(f'{DATA_DIR}/studentRegistration.csv')

student_info = student_info.merge(
    student_reg[['id_student', 'code_module', 'code_presentation',
                 'date_unregistration']],
    on=CHAVE, how='left'
)
student_info['date_unregistration'] = pd.to_numeric(
    student_info['date_unregistration'], errors='coerce'
)

# ═════════════════════════════════════════════════════════════════
# 2 — CORTE TEMPORAL INDIVIDUAL
# ═════════════════════════════════════════════════════════════════

# Remove Withdrawn sem data de desmatrícula
mask_withdrawn = student_info['final_result'] == 'Withdrawn'
mask_sem_sinal = (student_info['date_unregistration'].isna() |
                  (student_info['date_unregistration'] <= 1))
chave_excluir  = set(zip(
    student_info[mask_withdrawn & mask_sem_sinal]['id_student'],
    student_info[mask_withdrawn & mask_sem_sinal]['code_module'],
    student_info[mask_withdrawn & mask_sem_sinal]['code_presentation'],
))
mask_manter = ~student_info.apply(
    lambda r: (r['id_student'], r['code_module'], r['code_presentation'])
              in chave_excluir, axis=1
)
si = student_info[mask_manter].copy()

def calcular_cutoff(row):
    if row['final_result'] == 'Withdrawn':
        val = row['date_unregistration']
        return max(1, int(val) - 1) if not pd.isna(val) else CUTOFF_BASE
    return CUTOFF_BASE

si['cutoff_dia'] = si.apply(calcular_cutoff, axis=1)

cutoff_lookup = dict(zip(
    zip(si['id_student'], si['code_module'], si['code_presentation']),
    si['cutoff_dia']
))

# ═════════════════════════════════════════════════════════════════
# 3 — GRUPO PERFIL
# ═════════════════════════════════════════════════════════════════

perfil = si[CHAVE + [
    'gender', 'age_band', 'highest_education', 'imd_band',
    'disability', 'num_of_prev_attempts', 'studied_credits',
    'final_result'
]].copy()

for col in ['gender', 'age_band', 'highest_education', 'disability']:
    perfil[col] = LabelEncoder().fit_transform(perfil[col].astype(str))

imd_map = {f'{i*10}-{(i+1)*10}%': i+1 for i in range(10)}
perfil['imd_band_ord'] = si['imd_band'].map(imd_map).fillna(0)
perfil.drop(columns=['imd_band'], inplace=True)

FEATURES_PERFIL = [
    'gender', 'age_band', 'highest_education', 'imd_band_ord',
    'disability', 'num_of_prev_attempts', 'studied_credits'
]

# ═════════════════════════════════════════════════════════════════
# 4 — GRUPO INTERAÇÃO VLE
# ═════════════════════════════════════════════════════════════════

vle_full = student_vle.merge(vle[['id_site', 'activity_type']], on='id_site', how='left')
vle_full['cutoff_dia'] = vle_full.apply(
    lambda r: cutoff_lookup.get(
        (r['id_student'], r['code_module'], r['code_presentation']), CUTOFF_BASE
    ), axis=1
)
vle_cortado = vle_full[vle_full['date'] <= vle_full['cutoff_dia']].copy()

def agregar_vle(g):
    cutoff      = g['cutoff_dia'].iloc[0]
    meio        = cutoff / 2
    dias_unicos = g['date'].nunique()
    clicks_dia  = g.groupby('date')['sum_click'].sum()
    sum_total   = g['sum_click'].sum()
    c1a         = g[g['date'] <= meio]['sum_click'].sum()
    c2a         = g[g['date'] >  meio]['sum_click'].sum()

    return pd.Series({
        'sum_clicks':            sum_total,
        'avg_clicks_por_dia':    clicks_dia.mean(),
        'active_days':           dias_unicos,
        'unique_activities':     g['id_site'].nunique(),
        'click_variance':        clicks_dia.var(ddof=0) if dias_unicos > 1 else 0,
        'clicks_1a_metade':      c1a,
        'clicks_2a_metade':      c2a,
        'ratio_metade':          c2a / (c1a + 1),
        'clicks_por_dia_ativo':  sum_total / (dias_unicos + 1),
    })

interacao = vle_cortado.groupby(CHAVE).apply(agregar_vle).reset_index()
interacao['click_variance'] = interacao['click_variance'].fillna(0)
interacao['ratio_metade']   = interacao['ratio_metade'].fillna(1.0)

FEATURES_INTERACAO = [
    'avg_clicks_por_dia', 'active_days', 'unique_activities',
    'click_variance', 'clicks_1a_metade', 'clicks_2a_metade',
    'ratio_metade', 'clicks_por_dia_ativo',
]

# ═════════════════════════════════════════════════════════════════
# 5 — GRUPO AVALIAÇÃO
# ═════════════════════════════════════════════════════════════════

aval_full = student_assessment.merge(
    assessments[['id_assessment', 'weight', 'date', 'assessment_type']],
    on='id_assessment', how='left'
)
aval_full = aval_full.merge(
    si[CHAVE].drop_duplicates(), on='id_student', how='left'
)
aval_full['score']            = aval_full['score'].fillna(0)
aval_full['score_ponderada']  = aval_full['score'] * aval_full['weight'] / 100
aval_full['dias_antes_prazo'] = aval_full['date'] - aval_full['date_submitted']
aval_full['cutoff_dia'] = aval_full.apply(
    lambda r: cutoff_lookup.get(
        (r['id_student'], r['code_module'], r['code_presentation']), CUTOFF_BASE
    ), axis=1
)
aval_cortado = aval_full[aval_full['date_submitted'] <= aval_full['cutoff_dia']].copy()

def agregar_aval(g):
    n        = len(g)
    sorted_g = g.sort_values('date')
    mean_s   = g['score'].mean() if n > 0 else 0
    first_s  = sorted_g['score'].iloc[0] if n > 0 else 0
    w_score  = g['score_ponderada'].sum()

    return pd.Series({
        'num_submitted':          n,
        'mean_score':             mean_s,
        'first_score':            first_s,
        'weighted_score':         w_score,
        'num_late':               int((g['dias_antes_prazo'] < 0).sum()),
        'dias_antes_prazo_media': g['dias_antes_prazo'].mean() if n > 0 else 0,
        'score_variance':         g['score'].var(ddof=0) if n > 1 else 0,
        'delta_score':            mean_s - first_s,
        'score_por_submissao':    w_score / (n + 1),
    })

avaliacao = aval_cortado.groupby(CHAVE).apply(agregar_aval).reset_index()

FEATURES_AVALIACAO = [
    'num_submitted', 'mean_score', 'first_score', 'weighted_score',
    'num_late', 'dias_antes_prazo_media', 'score_variance',
    'delta_score', 'score_por_submissao',
]

# ═════════════════════════════════════════════════════════════════
# 6 — MERGE FINAL
# ═════════════════════════════════════════════════════════════════

df = perfil.merge(interacao, on=CHAVE, how='left').merge(avaliacao, on=CHAVE, how='left')
todas = FEATURES_PERFIL + FEATURES_INTERACAO + FEATURES_AVALIACAO
df[todas] = df[todas].fillna(0)

# ═════════════════════════════════════════════════════════════════
# 7 — LISTAGEM DAS FEATURES POR GRUPO
# ═════════════════════════════════════════════════════════════════

SEP = "=" * 60
print(SEP)
print("FEATURES POR GRUPO")
print(SEP)

grupos = {
    'PERFIL':    FEATURES_PERFIL,
    'INTERACAO': FEATURES_INTERACAO,
    'AVALIACAO': FEATURES_AVALIACAO,
}

for nome, cols in grupos.items():
    print(f"\n[{nome}]  — {len(cols)} features")
    for i, c in enumerate(cols, 1):
        print(f"  {i:>2}. {c}")

print(f"\n  Total: {len(todas)} features")
print(SEP)

# ═════════════════════════════════════════════════════════════════
# 8 — HEATMAP DE CORRELAÇÃO POR GRUPO
# ═════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(
    f'Correlação de Pearson por grupo — com corte temporal individual\n'
    f'(Withdrawn: até dia anterior à desmatrícula | Outros: até dia {CUTOFF_BASE})',
    fontsize=13
)

for ax, (nome, cols) in zip(axes, grupos.items()):
    corr      = df[cols].corr()
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 0)
    max_corr  = np.abs(corr_vals).max()

    # pares acima de 0.80
    pares_altos = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_vals[i, j]) > 0.80:
                pares_altos.append(f"{cols[i]} ↔ {cols[j]}  r={corr_vals[i,j]:.2f}")

    sns.heatmap(
        corr, ax=ax, annot=True, fmt='.2f',
        cmap='RdYlGn', vmin=-1, vmax=1, center=0,
        annot_kws={'size': 8}, linewidths=0.5
    )
    cor_titulo = 'red' if max_corr > 0.80 else 'black'
    ax.set_title(
        f'{nome} — {len(cols)} features\n'
        f'max |r| = {max_corr:.2f}  '
        f'{"⚠ pares > 0.80" if pares_altos else "✓ ok"}',
        fontsize=10, color=cor_titulo
    )

    if pares_altos:
        print(f"\n  [{nome}] Pares com |r| > 0.80:")
        for p in pares_altos:
            print(f"    {p}")

plt.tight_layout()
plt.savefig('correlacao_features_brutas.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Salvo: correlacao_features_brutas.png")