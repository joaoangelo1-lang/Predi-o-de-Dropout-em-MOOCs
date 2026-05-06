"""
OULAD Pipeline — Dropout (Withdrawn vs Outros) — v4  [CORRIGIDO]
=================================================================
Correções em relação ao original:
  - SHAP Etapa 10: cada modelo usa seu próprio scaler (extraído do pipeline
    treinado) — elimina dupla escala que causava shape errado no KernelExplainer
  - KernelExplainer: tratamento robusto do retorno de shap_values
    (lista OU array direto, com verificação de shape)
  - scaler_ref global removido — substituído por extração individual por modelo
"""

import os
import warnings
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, classification_report)
from sklearn.pipeline import Pipeline
import shap
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Configurações ──────────────────────────────────────────────────
path           = kagglehub.dataset_download("rocki37/open-university-learning-analytics-dataset")
DATA_DIR       = path
OUTPUT_DIR     = 'output_v4'
K_POR_GRUPO    = 4
N_FOLDS        = 5
RANDOM_STATE   = 42
CUTOFF_BASE    = 28
CORR_THRESHOLD = 0.80
PCA_VAR_MIN    = 0.95   # variância mínima explicada pelos PCs retidos

os.makedirs(OUTPUT_DIR, exist_ok=True)
SEP = "=" * 65


# ═════════════════════════════════════════════════════════════════
# ETAPA 1 — CARREGAMENTO
# ═════════════════════════════════════════════════════════════════

print(SEP)
print("ETAPA 1 — Carregando tabelas do OULAD  [pipeline v4 corrigido]")
print("  Correção: scaler individual por modelo + SHAP shape robusto")
print(SEP)

student_info       = pd.read_csv(f'{DATA_DIR}/studentInfo.csv')
student_assessment = pd.read_csv(f'{DATA_DIR}/studentAssessment.csv')
assessments        = pd.read_csv(f'{DATA_DIR}/assessments.csv')
student_vle        = pd.read_csv(f'{DATA_DIR}/studentVle.csv')
vle                = pd.read_csv(f'{DATA_DIR}/vle.csv')
student_reg        = pd.read_csv(f'{DATA_DIR}/studentRegistration.csv')

student_info = student_info.merge(
    student_reg[['id_student', 'code_module', 'code_presentation',
                 'date_unregistration']],
    on=['id_student', 'code_module', 'code_presentation'],
    how='left'
)
student_info['date_unregistration'] = pd.to_numeric(
    student_info['date_unregistration'], errors='coerce'
)

for nome, tbl in [('studentInfo', student_info),
                   ('studentAssessment', student_assessment),
                   ('assessments', assessments),
                   ('studentVle', student_vle),
                   ('vle', vle),
                   ('studentRegistration', student_reg)]:
    print(f"  {nome:<22}: {tbl.shape}")

print(f"\n  Distribuição final_result (original):")
print(student_info['final_result'].value_counts().to_string())


# ═════════════════════════════════════════════════════════════════
# ETAPA 2 — CORTE TEMPORAL INDIVIDUAL
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 2 — Corte temporal individual por aluno")
print(SEP)

CHAVE = ['id_student', 'code_module', 'code_presentation']

mask_withdrawn = student_info['final_result'] == 'Withdrawn'
mask_sem_sinal = (student_info['date_unregistration'].isna() |
                  (student_info['date_unregistration'] <= 1))

alunos_sem_sinal = student_info[mask_withdrawn & mask_sem_sinal]
print(f"  Withdrawn sem sinal (excluídos): {len(alunos_sem_sinal)}")

chave_excluir = set(zip(
    alunos_sem_sinal['id_student'],
    alunos_sem_sinal['code_module'],
    alunos_sem_sinal['code_presentation']
))

mask_manter = ~student_info.apply(
    lambda r: (r['id_student'], r['code_module'], r['code_presentation'])
              in chave_excluir, axis=1
)
student_info_filtrado = student_info[mask_manter].copy()

def calcular_cutoff(row):
    if row['final_result'] == 'Withdrawn':
        val = row['date_unregistration']
        if pd.isna(val):
            return CUTOFF_BASE
        return max(1, int(val) - 1)
    return CUTOFF_BASE

student_info_filtrado['cutoff_dia'] = student_info_filtrado.apply(
    calcular_cutoff, axis=1
)

print(f"  Registros mantidos: {len(student_info_filtrado)}")
print(f"\n  Distribuição final_result (após filtro):")
print(student_info_filtrado['final_result'].value_counts().to_string())


# ═════════════════════════════════════════════════════════════════
# ETAPA 3 — FEATURE ENGINEERING v4
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 3 — Feature engineering v4")
print(SEP)

cutoff_lookup = dict(zip(
    zip(student_info_filtrado['id_student'],
        student_info_filtrado['code_module'],
        student_info_filtrado['code_presentation']),
    student_info_filtrado['cutoff_dia']
))

# ── Grupo 1: Perfil ───────────────────────────────────────────────
print("\n[Grupo 1] Perfil")

perfil = student_info_filtrado[[
    'id_student', 'code_module', 'code_presentation',
    'gender', 'age_band', 'highest_education',
    'imd_band', 'disability',
    'num_of_prev_attempts', 'studied_credits',
    'final_result', 'cutoff_dia'
]].copy()

for col in ['gender', 'age_band', 'highest_education', 'disability']:
    perfil[col] = LabelEncoder().fit_transform(perfil[col].astype(str))

imd_map = {f'{i*10}-{(i+1)*10}%': i+1 for i in range(10)}
perfil['imd_band_ord'] = student_info_filtrado['imd_band'].map(imd_map).fillna(0)
perfil.drop(columns=['imd_band'], inplace=True)

FEATURES_PERFIL = [
    'gender', 'age_band', 'highest_education',
    'imd_band_ord', 'disability',
    'num_of_prev_attempts', 'studied_credits'
]
print(f"  {len(FEATURES_PERFIL)} features: {FEATURES_PERFIL}")

# ── Grupo 2: Interação VLE ────────────────────────────────────────
print(f"\n[Grupo 2] Interação VLE (avg_clicks_por_dia reintroduzida — PCA resolverá)")

vle_full = student_vle.merge(
    vle[['id_site', 'activity_type']], on='id_site', how='left'
)
vle_full['cutoff_dia'] = vle_full.apply(
    lambda r: cutoff_lookup.get(
        (r['id_student'], r['code_module'], r['code_presentation']), CUTOFF_BASE
    ), axis=1
)
vle_cortado = vle_full[vle_full['date'] <= vle_full['cutoff_dia']].copy()

print(f"\n  Registros VLE originais  : {len(vle_full):,}")
print(f"  Registros VLE após corte : {len(vle_cortado):,}  "
      f"({len(vle_cortado)/len(vle_full):.1%})")

def agregar_vle_v4(g):
    cutoff      = g['cutoff_dia'].iloc[0]
    meio        = cutoff / 2
    dias_unicos = g['date'].nunique()
    clicks_dia  = g.groupby('date')['sum_click'].sum()
    sum_total   = g['sum_click'].sum()
    c1a         = g[g['date'] <= meio]['sum_click'].sum()
    c2a         = g[g['date'] >  meio]['sum_click'].sum()

    return pd.Series({
        'avg_clicks_por_dia':    clicks_dia.mean(),
        'active_days':           dias_unicos,
        'unique_activities':     g['id_site'].nunique(),
        'click_variance':        clicks_dia.var(ddof=0) if dias_unicos > 1 else 0,
        'clicks_2a_metade':      c2a,
        'ratio_metade':          c2a / (c1a + 1),
        'clicks_por_dia_ativo':  sum_total / (dias_unicos + 1),
    })

interacao = (
    vle_cortado
    .groupby(CHAVE)
    .apply(agregar_vle_v4)
    .reset_index()
)
interacao['click_variance'] = interacao['click_variance'].fillna(0)
interacao['ratio_metade']   = interacao['ratio_metade'].fillna(1.0)

FEATURES_INTERACAO = [
    'avg_clicks_por_dia', 'active_days', 'unique_activities',
    'click_variance', 'clicks_2a_metade',
    'ratio_metade', 'clicks_por_dia_ativo',
]
print(f"\n  {len(FEATURES_INTERACAO)} features: {FEATURES_INTERACAO}")

# ── Grupo 3: Avaliações ───────────────────────────────────────────
print(f"\n[Grupo 3] Avaliações")

aval_full = student_assessment.merge(
    assessments[['id_assessment', 'weight', 'date', 'assessment_type']],
    on='id_assessment', how='left'
)
aval_full = aval_full.merge(
    student_info_filtrado[['id_student', 'code_module',
                           'code_presentation']].drop_duplicates(),
    on='id_student', how='left'
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

print(f"\n  Registros aval. originais  : {len(aval_full):,}")
print(f"  Registros aval. após corte : {len(aval_cortado):,}  "
      f"({len(aval_cortado)/len(aval_full):.1%})")

def agregar_aval_v4(g):
    n        = len(g)
    sorted_g = g.sort_values('date')
    mean_s   = g['score'].mean() if n > 0 else 0
    first_s  = sorted_g['score'].iloc[0] if n > 0 else 0
    w_score  = g['score_ponderada'].sum()

    return pd.Series({
        'weighted_score':         w_score,
        'num_late':               int((g['dias_antes_prazo'] < 0).sum()),
        'dias_antes_prazo_media': g['dias_antes_prazo'].mean() if n > 0 else 0,
        'score_variance':         g['score'].var(ddof=0) if n > 1 else 0,
        'delta_score':            mean_s - first_s,
        'score_por_submissao':    w_score / (n + 1),
    })

avaliacao = (
    aval_cortado
    .groupby(CHAVE)
    .apply(agregar_aval_v4)
    .reset_index()
)

FEATURES_AVALIACAO = [
    'weighted_score', 'num_late', 'dias_antes_prazo_media',
    'score_variance', 'delta_score', 'score_por_submissao',
]
print(f"\n  {len(FEATURES_AVALIACAO)} features: {FEATURES_AVALIACAO}")


# ═════════════════════════════════════════════════════════════════
# ETAPA 4 — MERGE FINAL
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 4 — Merge final")
print(SEP)

df = (
    perfil
    .merge(interacao, on=CHAVE, how='left')
    .merge(avaliacao,  on=CHAVE, how='left')
)

todas_features = FEATURES_PERFIL + FEATURES_INTERACAO + FEATURES_AVALIACAO
df[todas_features] = df[todas_features].fillna(0)

df['target'] = df['final_result'].apply(
    lambda x: 1 if str(x).strip() == 'Withdrawn' else 0
)

n_withdrawn = (df.target == 1).sum()
n_outros    = (df.target == 0).sum()
taxa        = df.target.mean()

print(f"  Shape     : {df.shape}")
print(f"  Withdrawn : {n_withdrawn}  ({taxa:.1%})")
print(f"  Outros    : {n_outros}  ({1-taxa:.1%})")

y          = df['target'].values
ids_alunos = df['id_student'].values

df.to_csv(f'{OUTPUT_DIR}/oulad_dropout_v4.csv', index=False)
print(f"\n  Salvo em {OUTPUT_DIR}/oulad_dropout_v4.csv")


# ═════════════════════════════════════════════════════════════════
# ETAPA 5 — RESOLUÇÃO AUTOMÁTICA DE CORRELAÇÃO POR PCA + SELEÇÃO
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 5 — Resolução automática de correlação por PCA + seleção greedy")
print(f"  threshold correlação : |r| > {CORR_THRESHOLD}")
print(f"  variância mínima PCA : {PCA_VAR_MIN:.0%}")
print(f"  K por grupo          : {K_POR_GRUPO}")
print(SEP)

GRUPOS_BRUTOS = {
    'perfil':    FEATURES_PERFIL,
    'interacao': FEATURES_INTERACAO,
    'avaliacao': FEATURES_AVALIACAO,
}


def detectar_clusters(cols, corr_matrix, threshold):
    """
    Agrupa features em clusters pelo critério de correlação mútua.
    Usa busca em grafo simples (bfs): dois nós ficam no mesmo cluster
    se |r| > threshold com qualquer membro já no cluster.
    """
    visitado = set()
    clusters = []
    for c in cols:
        if c in visitado:
            continue
        cluster = [c]
        visitado.add(c)
        fila = [c]
        while fila:
            atual = fila.pop(0)
            for d in cols:
                if d not in visitado and corr_matrix.loc[atual, d] > threshold:
                    cluster.append(d)
                    visitado.add(d)
                    fila.append(d)
        clusters.append(cluster)
    return clusters


def aplicar_pca_cluster(df, cluster, var_minima=PCA_VAR_MIN):
    """
    Aplica PCA em um cluster correlato.
    Retorna array com os PCs retidos e seus nomes descritivos.
    """
    X_cluster = StandardScaler().fit_transform(df[cluster].values)
    pca       = PCA(n_components=None)
    pca.fit(X_cluster)
    var_acum  = np.cumsum(pca.explained_variance_ratio_)
    n_comp    = int(np.searchsorted(var_acum, var_minima)) + 1
    X_pca     = pca.transform(X_cluster)[:, :n_comp]
    base_nome = '+'.join(cluster)
    nomes_pc  = [f"PC[{base_nome}]_{k+1}" for k in range(n_comp)]
    return X_pca, nomes_pc, pca, n_comp, var_acum[n_comp - 1]


def selecionar_greedy(X_candidatas, nomes, y, k, df_orig=None,
                      threshold=CORR_THRESHOLD):
    """
    Seleção greedy: ordena por F-score, aceita candidata apenas se
    |r| <= threshold com todas as já selecionadas.
    Funciona sobre arrays numpy (compatível com PCs).
    """
    fscores, _ = f_classif(X_candidatas, y)
    ordem      = np.argsort(fscores)[::-1]
    sel_idx    = []
    sel_X      = []
    for i in ordem:
        candidata_X = X_candidatas[:, i]
        correlacionada = any(
            abs(np.corrcoef(candidata_X, sel_X[j])[0, 1]) > threshold
            for j in range(len(sel_X))
        )
        if not correlacionada:
            sel_idx.append(i)
            sel_X.append(candidata_X)
        if len(sel_idx) == k:
            break
    return sel_idx, fscores


def resolver_grupo(df, cols, y, k, threshold=CORR_THRESHOLD,
                   var_minima=PCA_VAR_MIN, nome_grupo=''):
    """
    Pipeline completo por grupo:
      1. Detecta clusters de features correlatas
      2. Clusters solitários → feature original direta
         Clusters com 2+ → substituídos por PCs
      3. Seleção greedy sobre o pool de features/PCs resultante
    Retorna X_final (array), nomes das features/PCs selecionadas,
    e dict de log para auditoria.
    """
    corr_abs = df[cols].corr().abs()
    clusters = detectar_clusters(cols, corr_abs, threshold)

    pool_X     = []
    pool_nomes = []
    log        = {'clusters': [], 'pca': [], 'greedy': []}

    print(f"\n  [{nome_grupo}]  {len(cols)} features → {len(clusters)} cluster(s)")

    for cluster in clusters:
        if len(cluster) == 1:
            pool_X.append(df[cluster[0]].values.reshape(-1, 1))
            pool_nomes.append(cluster[0])
            print(f"    Cluster [{cluster[0]}] → direto (sem correlação alta)")
            log['clusters'].append({'cluster': cluster, 'tipo': 'direto'})
        else:
            X_pca, nomes_pc, pca_obj, n_comp, var_ret = \
                aplicar_pca_cluster(df, cluster, var_minima)
            pool_X.append(X_pca)
            pool_nomes.extend(nomes_pc)
            print(f"    Cluster {cluster}")
            print(f"      → {n_comp} PC(s)  |  var. explicada: {var_ret:.1%}")
            for k_pc, (nome_pc, ev) in enumerate(
                    zip(nomes_pc, pca_obj.explained_variance_ratio_[:n_comp])):
                print(f"         {nome_pc}  ({ev:.1%})")
            log['clusters'].append({'cluster': cluster, 'tipo': 'pca',
                                    'n_comp': n_comp, 'var_retida': var_ret})
            log['pca'].append({'cluster': cluster, 'pcs': nomes_pc,
                               'var': var_ret})

    X_pool    = np.hstack(pool_X)
    sel_idx, fscores = selecionar_greedy(X_pool, pool_nomes, y, k, threshold=threshold)
    X_final   = X_pool[:, sel_idx]
    nomes_sel = [pool_nomes[i] for i in sel_idx]
    fmap      = {pool_nomes[i]: fscores[i] for i in range(len(pool_nomes))}

    print(f"\n    Greedy selecionou {len(nomes_sel)} de {len(pool_nomes)} candidatas:")
    for n in nomes_sel:
        print(f"      {n:<55}  F = {fmap[n]:>8.2f}")
    rejeitadas = [pool_nomes[i] for i in range(len(pool_nomes)) if i not in sel_idx]
    if rejeitadas:
        print(f"    Rejeitadas: {rejeitadas}")

    log['greedy'] = {'selecionadas': nomes_sel, 'rejeitadas': rejeitadas}
    return X_final, nomes_sel, log


# Executa resolução por grupo
GRUPOS_LOG      = {}
feature_names   = []
grupos_por_feat = []
mapa_grupos     = {}
arrays_X        = []
offset          = 0

for nome_grupo, cols in GRUPOS_BRUTOS.items():
    X_grupo, nomes_sel, log = resolver_grupo(
        df, cols, y, K_POR_GRUPO,
        threshold=CORR_THRESHOLD,
        var_minima=PCA_VAR_MIN,
        nome_grupo=nome_grupo
    )
    GRUPOS_LOG[nome_grupo]  = log
    mapa_grupos[nome_grupo] = list(range(offset, offset + len(nomes_sel)))
    offset                 += len(nomes_sel)
    arrays_X.append(X_grupo)
    feature_names.extend(nomes_sel)
    grupos_por_feat.extend([nome_grupo] * len(nomes_sel))

X = np.hstack(arrays_X)
n_features_total = X.shape[1]
print(f"\n  Total features no modelo: {n_features_total}")
print(f"  Features: {feature_names}")

# Reconstrói mapa_grupos a partir de grupos_por_feat (garantia de consistência)
mapa_grupos = {}
for i, grupo in enumerate(grupos_por_feat):
    mapa_grupos.setdefault(grupo, []).append(i)

# Heatmap de correlação — features originais por grupo (auditoria)
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Correlação de Pearson — v4 (antes do PCA, por grupo)', fontsize=13)
for col_idx, (nome, cols) in enumerate(GRUPOS_BRUTOS.items()):
    corr = df[cols].corr()
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 0)
    max_corr = np.abs(corr_vals).max()
    sns.heatmap(corr, ax=axes[col_idx], annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=-1, vmax=1, center=0,
                annot_kws={'size': 8}, linewidths=0.5)
    axes[col_idx].set_title(
        f'{nome} — {len(cols)} features\n'
        f'max |r| = {max_corr:.2f}  (PCA resolve automaticamente)',
        fontsize=10
    )
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlacao_v4.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Salvo: correlacao_v4.png")

# Salva log de features
log_rows = []
for grupo, log in GRUPOS_LOG.items():
    for entry in log['clusters']:
        for feat in entry['cluster']:
            log_rows.append({
                'grupo': grupo, 'feature_original': feat,
                'tipo': entry['tipo'],
                'n_pcs': entry.get('n_comp', 1),
                'var_retida': entry.get('var_retida', 1.0),
            })
pd.DataFrame(log_rows).to_csv(
    f'{OUTPUT_DIR}/features_pca_log_v4.csv', index=False)


# ═════════════════════════════════════════════════════════════════
# ETAPA 6 — MODELOS
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 6 — Definindo modelos (SVM removido)")
print(SEP)

spw = n_outros / n_withdrawn

MODELOS = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        ))
    ]),
    'MLP': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', alpha=1e-4, learning_rate='adaptive',
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE
        ))
    ]),
}
for nome in MODELOS:
    print(f"  ✓ {nome}")


# ═════════════════════════════════════════════════════════════════
# ETAPA 7 — VALIDAÇÃO CRUZADA AGRUPADA
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print(f"ETAPA 7 — Validação cruzada agrupada ({N_FOLDS}-fold)")
print(SEP)

cv_agrupada = GroupKFold(n_splits=N_FOLDS)
METRICAS_CV = {}

for nome, pipeline in MODELOS.items():
    print(f"\n  [{nome}]")
    accs, f1s, aucs = [], [], []
    for idx_tr, idx_val in cv_agrupada.split(X, y, groups=ids_alunos):
        X_tr, X_val = X[idx_tr], X[idx_val]
        y_tr, y_val = y[idx_tr], y[idx_val]
        pipeline.fit(X_tr, y_tr)
        y_pred  = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))
        aucs.append(roc_auc_score(y_val, y_proba))
    METRICAS_CV[nome] = {
        'accuracy':     (np.mean(accs), np.std(accs)),
        'f1_withdrawn': (np.mean(f1s),  np.std(f1s)),
        'roc_auc':      (np.mean(aucs), np.std(aucs)),
    }
    for m, (mean, std) in METRICAS_CV[nome].items():
        print(f"    {m:<18}: {mean:.4f} ± {std:.4f}")


# ═════════════════════════════════════════════════════════════════
# ETAPA 8 — TREINO FINAL + AVALIAÇÃO NO TESTE
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 8 — Treino final e avaliação no teste (hold-out 20%)")
print(SEP)

alunos_unicos = df['id_student'].unique()
np.random.seed(RANDOM_STATE)
alunos_teste  = set(np.random.choice(
    alunos_unicos, size=int(len(alunos_unicos) * 0.2), replace=False
))

mask_teste      = df['id_student'].isin(alunos_teste).values
X_train, X_test = X[~mask_teste], X[mask_teste]
y_train, y_test = y[~mask_teste], y[mask_teste]

print(f"  Treino : {X_train.shape[0]} amostras")
print(f"  Teste  : {X_test.shape[0]} amostras")

MODELOS_TREINADOS = {}
METRICAS_TESTE    = {}

for nome, pipeline in MODELOS.items():
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    MODELOS_TREINADOS[nome] = pipeline
    METRICAS_TESTE[nome] = {
        'accuracy':     accuracy_score(y_test, y_pred),
        'f1_withdrawn': f1_score(y_test, y_pred, pos_label=1),
        'roc_auc':      roc_auc_score(y_test, y_proba),
    }
    print(f"\n  [{nome}]")
    print(classification_report(
        y_test, y_pred,
        target_names=['Outros', 'Withdrawn'], digits=4
    ))
    print(f"  ROC-AUC: {METRICAS_TESTE[nome]['roc_auc']:.4f}")


# ═════════════════════════════════════════════════════════════════
# ETAPA 9 — TABELA COMPARATIVA
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 9 — Tabela comparativa de métricas")
print(SEP)

rows = []
for nome in MODELOS:
    cv  = METRICAS_CV[nome]
    tst = METRICAS_TESTE[nome]
    rows.append({
        'Modelo':            nome,
        'CV Accuracy':       f"{cv['accuracy'][0]:.4f} ± {cv['accuracy'][1]:.4f}",
        'CV F1 Withdrawn':   f"{cv['f1_withdrawn'][0]:.4f} ± {cv['f1_withdrawn'][1]:.4f}",
        'CV ROC-AUC':        f"{cv['roc_auc'][0]:.4f} ± {cv['roc_auc'][1]:.4f}",
        'Test Accuracy':     f"{tst['accuracy']:.4f}",
        'Test F1 Withdrawn': f"{tst['f1_withdrawn']:.4f}",
        'Test ROC-AUC':      f"{tst['roc_auc']:.4f}",
    })

comp_df = pd.DataFrame(rows)
print(comp_df.to_string(index=False))
comp_df.to_csv(f'{OUTPUT_DIR}/comparativo_modelos_v4.csv', index=False)

melhor_nome = max(METRICAS_TESTE, key=lambda x: METRICAS_TESTE[x]['roc_auc'])
print(f"\n  → Melhor modelo (ROC-AUC): [{melhor_nome}]")


# ═════════════════════════════════════════════════════════════════
# ETAPA 10 — SHAP PARA TODOS OS MODELOS
# ═════════════════════════════════════════════════════════════════
#
# CORREÇÃO APLICADA:
#   - Cada modelo usa seu próprio scaler (extraído do pipeline treinado)
#     em vez de um scaler_ref global treinado separadamente.
#   - Isso evita dupla escala nos dados e garante shape correto para o SHAP.
#   - KernelExplainer: tratamento robusto do retorno (lista ou array).
#   - Verificação de shape antes de calcular shap_mean.
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 10 — SHAP por modelo")
print("  TreeExplainer   → Random Forest, XGBoost  (exato)")
print("  LinearExplainer → Logistic Regression      (exato)")
print("  KernelExplainer → MLP                      (aproximado, 300 pts)")
print(f"  n_features esperado por modelo: {n_features_total}")
print(SEP)

# Tipo de explainer por modelo
SHAP_CONFIGS = {
    'Random Forest':       'tree',
    'XGBoost':             'tree',
    'Logistic Regression': 'linear',
    'MLP':                 'kernel',
}

RANKING_GRUPOS_POR_MODELO = {}
SHAP_MEAN_POR_MODELO      = {}

for nome, tipo in SHAP_CONFIGS.items():
    print(f"\n  [{nome}] — {tipo} explainer")

    # ── CORREÇÃO: usa o scaler JÁ TREINADO dentro do pipeline ──────
    # Cada pipeline tem seu próprio StandardScaler ajustado em X_train.
    # Extraímos ele aqui para escalar X_train e X_test de forma consistente,
    # passando apenas o clf (sem o pipeline wrapper) para o explainer.
    pipeline_treinado = MODELOS_TREINADOS[nome]
    scaler_modelo     = pipeline_treinado['scaler']   # scaler do pipeline
    clf               = pipeline_treinado['clf']       # classificador puro

    X_train_sc = scaler_modelo.transform(X_train)     # shape: (n_train, n_features_total)
    X_test_sc  = scaler_modelo.transform(X_test)      # shape: (n_test,  n_features_total)

    if tipo == 'tree':
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_sc)
        # RF retorna lista [classe0, classe1]; XGBoost retorna array direto
        if isinstance(shap_values, list):
            sv = shap_values[1]   # classe positiva (Withdrawn)
        else:
            sv = shap_values

    elif tipo == 'linear':
        explainer = shap.LinearExplainer(clf, X_train_sc)
        sv        = explainer.shap_values(X_test_sc)
        # LinearExplainer pode retornar lista ou array; normaliza para array 2-D
        if isinstance(sv, list):
            sv = sv[1]

    elif tipo == 'kernel':
        np.random.seed(RANDOM_STATE)
        bg_idx    = np.random.choice(len(X_train_sc), size=200, replace=False)
        explainer = shap.KernelExplainer(clf.predict_proba, X_train_sc[bg_idx])
        tst_idx   = np.random.choice(len(X_test_sc),
                                     size=min(300, len(X_test_sc)), replace=False)
        sv_raw = explainer.shap_values(X_test_sc[tst_idx])

        # KernelExplainer com predict_proba retorna lista [classe0, classe1]
        # cada elemento com shape (n_samples, n_features)
        if isinstance(sv_raw, list):
            sv = sv_raw[1]    # classe positiva (Withdrawn)
        else:
            sv = sv_raw       # já é array direto

    # ── Verificação de shape: garante (n_samples, n_features_total) ──
    if sv.ndim == 1:
        raise ValueError(
            f"[{nome}] SHAP retornou array 1-D com shape {sv.shape}. "
            f"Esperado 2-D (n_samples, {n_features_total})."
        )
    if sv.shape[1] != n_features_total:
        raise ValueError(
            f"[{nome}] SHAP shape incorreto: {sv.shape}. "
            f"Esperado (n_samples, {n_features_total}). "
            f"Verifique se o scaler e o clf são do mesmo pipeline treinado."
        )

    shap_mean = np.abs(sv).mean(axis=0)              # shape: (n_features_total,)
    SHAP_MEAN_POR_MODELO[nome] = shap_mean

    ranking_grupos = {g: shap_mean[idx].mean() for g, idx in mapa_grupos.items()}
    RANKING_GRUPOS_POR_MODELO[nome] = ranking_grupos

    for g, v in sorted(ranking_grupos.items(), key=lambda x: -x[1]):
        print(f"    {g:<12}: {v:.6f}")

# Consenso: qual grupo venceu em mais modelos
contagem_melhor = {g: 0 for g in GRUPOS_BRUTOS}
for nome, ranking in RANKING_GRUPOS_POR_MODELO.items():
    contagem_melhor[max(ranking, key=ranking.get)] += 1

print(f"\n  Consenso SHAP ({len(SHAP_CONFIGS)} modelos):")
for g, cnt in sorted(contagem_melhor.items(), key=lambda x: -x[1]):
    print(f"    {g:<12}: {cnt}/{len(SHAP_CONFIGS)} modelos")

melhor_grupo_consenso = max(contagem_melhor, key=contagem_melhor.get)


# ═════════════════════════════════════════════════════════════════
# ETAPA 11 — VISUALIZAÇÕES
# ═════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("ETAPA 11 — Gerando visualizações")
print(SEP)

CORES_GRUPO = {
    'perfil':    '#6ee7b7',
    'interacao': '#f87171',
    'avaliacao': '#93c5fd'
}
CORES_MODELO = {
    'Logistic Regression': '#a78bfa',
    'Random Forest':       '#6ee7b7',
    'XGBoost':             '#f87171',
    'MLP':                 '#38bdf8',
}

# Fig 1: Comparativo de métricas
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Dropout (Withdrawn vs Outros) — v4 (PCA automático)', fontsize=13)
for ax, met, titulo in zip(axes,
    ['accuracy', 'f1_withdrawn', 'roc_auc'],
    ['Accuracy', 'F1 Withdrawn', 'ROC-AUC']):
    modelos_nomes = list(MODELOS.keys())
    vals_cv   = [METRICAS_CV[m][met][0]  for m in modelos_nomes]
    errs_cv   = [METRICAS_CV[m][met][1]  for m in modelos_nomes]
    vals_test = [METRICAS_TESTE[m][met]  for m in modelos_nomes]
    cores     = [CORES_MODELO[m]         for m in modelos_nomes]
    x         = np.arange(len(modelos_nomes))
    ax.bar(x - 0.175, vals_cv,   0.35, yerr=errs_cv,
           color=cores, alpha=0.6, capsize=4, label='CV (média±std)')
    ax.bar(x + 0.175, vals_test, 0.35,
           color=cores, alpha=1.0, label='Teste')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos_nomes, rotation=15, ha='right', fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.set_title(titulo, fontsize=11)
    if ax == axes[0]:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparativo_metricas_v4.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Salvo: comparativo_metricas_v4.png")

# Fig 2: SHAP por grupo — todos os modelos (painel por modelo)
fig, axes = plt.subplots(1, len(SHAP_CONFIGS), figsize=(18, 5))
fig.suptitle('Importância SHAP por grupo — v4 (PCA automático)', fontsize=13)
for ax, (nome, ranking) in zip(axes, RANKING_GRUPOS_POR_MODELO.items()):
    grupos_ord = sorted(ranking.items(), key=lambda x: -x[1])
    nomes_g    = [g for g, _ in grupos_ord]
    vals_g     = [v for _, v in grupos_ord]
    bars = ax.barh(nomes_g, vals_g,
                   color=[CORES_GRUPO[g] for g in nomes_g], height=0.5)
    for bar, val in zip(bars, vals_g):
        ax.text(val + max(vals_g) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.5f}', va='center', fontsize=8)
    ax.set_title(nome, fontsize=10)
    ax.set_xlabel('SHAP médio absoluto', fontsize=9)
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_por_modelo_v4.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Salvo: shap_por_modelo_v4.png")

# Fig 3: SHAP por feature (melhor modelo) + consenso
sv_melhor = SHAP_MEAN_POR_MODELO[melhor_nome]
feat_df = pd.DataFrame({
    'feature': feature_names,
    'shap':    sv_melhor,
    'grupo':   grupos_por_feat
}).sort_values('shap')

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f'SHAP por feature — {melhor_nome}  [v4: PCA automático]', fontsize=13)

axes[0].barh(feat_df['feature'], feat_df['shap'],
             color=[CORES_GRUPO[g] for g in feat_df['grupo']], height=0.6)
axes[0].set_xlabel('SHAP médio absoluto', fontsize=10)
axes[0].set_title(f'Features/PCs — {melhor_nome} (cor = grupo)', fontsize=11)
axes[0].legend(
    handles=[mpatches.Patch(color=c, label=g) for g, c in CORES_GRUPO.items()],
    fontsize=9
)

contagem_vals  = [contagem_melhor[g] for g in ['perfil', 'interacao', 'avaliacao']]
contagem_cores = [CORES_GRUPO[g]     for g in ['perfil', 'interacao', 'avaliacao']]
axes[1].pie(contagem_vals,
            labels=['perfil', 'interacao', 'avaliacao'],
            colors=contagem_cores, autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 10})
axes[1].set_title(f'Consenso SHAP — {len(SHAP_CONFIGS)} modelos', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_features_v4.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Salvo: shap_features_v4.png")



print(f"\n{SEP}")
print("ETAPA 12 — Exportando resultados")
print(SEP)

pd.DataFrame([
    {'modelo': nome, 'rank': rank, 'grupo': g, 'shap_medio': v}
    for nome, ranking in RANKING_GRUPOS_POR_MODELO.items()
    for rank, (g, v) in enumerate(
        sorted(ranking.items(), key=lambda x: -x[1]), 1)
]).to_csv(f'{OUTPUT_DIR}/shap_por_grupo_v4.csv', index=False)

feat_df.sort_values('shap', ascending=False).to_csv(
    f'{OUTPUT_DIR}/shap_features_{melhor_nome.replace(" ", "_")}_v4.csv',
    index=False
)

pd.DataFrame([
    {'grupo': g, 'vezes_melhor': v,
     'percentual': f"{v/len(SHAP_CONFIGS):.0%}"}
    for g, v in sorted(contagem_melhor.items(), key=lambda x: -x[1])
]).to_csv(f'{OUTPUT_DIR}/consenso_grupos_v4.csv', index=False)

print(f"\n{SEP}")
print("Pipeline v4 [CORRIGIDO] concluído!")
print(f"  Correções aplicadas:")
print(f"    Scaler SHAP : cada modelo usa pipeline_treinado['scaler']")
print(f"         (elimina dupla escala e garante shape correto)")
print(f"    KernelExpl. : sv_raw tratado como lista OU array + check de shape")
print(f"    LinearExpl. : idem — normaliza lista para array da classe positiva")
print(f"  Novidades v4 (mantidas):")
print(f"    PCA automático : clusters de features correlatas → PCs")
print(f"                     (>= {PCA_VAR_MIN:.0%} variância retida por cluster)")
print(f"    Seleção        : greedy F-score sobre features + PCs")
print(f"    SHAP        : 4 modelos — Tree (RF/XGB), Linear (LR), Kernel (MLP)")
print(f"    SVM          : removido")
print(f"  Melhor modelo    : {melhor_nome}")
print(f"  Melhor grupo     : {melhor_grupo_consenso}  "
      f"({contagem_melhor[melhor_grupo_consenso]}/{len(SHAP_CONFIGS)} modelos)")
print(f"  Outputs em       : {OUTPUT_DIR}/")
print(SEP)