# =============================================================================
# full_pipeline.py
#
# Kompleksowa analiza anonimizacji i podatności na atak skośności.
#
# Dla każdego z 3 datasetów:
#   1a. Adaptive search — optymalne (k, l) bez t-bliskości
#   1b. Adaptive search — optymalne (k, l, t) z t-bliskością
#   2.  Analiza siatki  — atak skośności dla wielu kombinacji (k, l)
#   3.  Pipeline        — k-anon → l-różn. → t-blisk. + atak po każdym etapie
#   4.  Wykresy         — krzywa podatności + porównanie etapów
#
# DATASETY (UCI API):
#   Adult (id=2)         — atrybut wrażliwy: income (4 wartości)
#   German Credit (id=144) — atrybut wrażliwy: checking_status (4 wartości)
#   Medical Cost (local)   — atrybut wrażliwy: charges_cat (4 kwartyle)
# =============================================================================

import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


# =============================================================================
# LOGOWANIE DO PLIKU
# =============================================================================

class Tee:
    """Przekierowuje stdout jednocześnie na konsolę i do pliku."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log.write(f'# Wyniki full_pipeline.py — {ts}\n')
        self.log.write('=' * 70 + '\n\n')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


LOG_FILE = 'wyniki_full_pipeline.txt'
tee = Tee(LOG_FILE)
sys.stdout = tee
print(f'Wyniki zapisywane do: {LOG_FILE}')

from shared_functions import (
    add_anon_id,
    clean_dataframe,
    generalize_numeric_qcut,
    apply_k_anonymity,
    apply_l_diversity,
    apply_t_closeness,
    verify_k_anonymity,
    verify_l_diversity,
    verify_t_closeness,
    print_section,
    run_skewness_attack,
    adaptive_search,
    print_adaptive_results,
)

os.makedirs('plots', exist_ok=True)

TAU = 0.10

# Siatka (k, l) do analizy podatności — l=1 oznacza tylko k-anonimowość
KL_GRID = [
    (2,  1), (5,  1), (10, 1), (20, 1),
    (2,  2), (5,  2), (10, 2), (20, 2),
]


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def prepare_work_df(df, qi_categorical, numerical_bins, sensitive_attr):
    """Generalizuje kolumny numeryczne i buduje DataFrame roboczy + listę QI."""
    df_work = df[['anon_ID'] + qi_categorical + [sensitive_attr]].copy()
    gen_cols = []
    for col, n_bins in numerical_bins.items():
        col_gen = f'{col}_gen'
        df_work[col_gen] = generalize_numeric_qcut(df[col], n_bins).values
        gen_cols.append(col_gen)
    Q = qi_categorical + gen_cols
    return df_work, Q


def run_pipeline_and_attack(df_work, Q, sensitive_attr, k, l, t,
                             dataset_name, tau):
    """
    Stosuje kolejno k-anonimowość, l-różnorodność, t-bliskość
    i przeprowadza atak skośności po każdym etapie.
    Zwraca słownik z wynikami trzech etapów.
    """
    total = len(df_work)
    results = {}

    # Etap A: k-anonimowość
    df_k = apply_k_anonymity(df_work, Q, k=k)
    pct_k = len(df_k) / total * 100
    print(f"\n  Etap A — k={k}: {len(df_k)} rekordów ({pct_k:.1f}%)")
    verify_k_anonymity(df_k, Q, k)
    r = run_skewness_attack(df_k, Q, sensitive_attr, tau,
                             dataset_name, f'k={k}')
    r.update({'k': k, 'l': 1, 't': None,
               'n_retained': len(df_k), 'pct_retained': round(pct_k, 1)})
    results['k'] = r

    # Etap B: + l-różnorodność
    df_kl = apply_l_diversity(df_k, Q, sensitive_attr, l=l)
    pct_kl = len(df_kl) / total * 100
    print(f"\n  Etap B — k={k}, l={l}: {len(df_kl)} rekordów ({pct_kl:.1f}%)")
    verify_l_diversity(df_kl, Q, sensitive_attr, l)
    r = run_skewness_attack(df_kl, Q, sensitive_attr, tau,
                             dataset_name, f'k={k}, l={l}')
    r.update({'k': k, 'l': l, 't': None,
               'n_retained': len(df_kl), 'pct_retained': round(pct_kl, 1)})
    results['kl'] = r

    # Etap C: + t-bliskość
    df_klt, gv = apply_t_closeness(df_kl, Q, sensitive_attr, t=t)
    pct_klt = len(df_klt) / total * 100
    print(f"\n  Etap C — k={k}, l={l}, t={t}: {len(df_klt)} rekordów ({pct_klt:.1f}%)")
    verify_t_closeness(df_klt, Q, sensitive_attr, t, global_values=gv)
    r = run_skewness_attack(df_klt, Q, sensitive_attr, tau,
                             dataset_name, f'k={k}, l={l}, t={t}')
    r.update({'k': k, 'l': l, 't': t,
               'n_retained': len(df_klt), 'pct_retained': round(pct_klt, 1)})
    results['klt'] = r

    return results


def run_grid_attacks(df_work, Q, sensitive_attr, kl_grid, tau, dataset_name):
    """Atak skośności dla każdej kombinacji (k, l) z siatki — bez wykresów."""
    grid_results = []
    for k, l in kl_grid:
        df_k = apply_k_anonymity(df_work, Q, k=k)
        if len(df_k) == 0:
            continue
        df_kl = apply_l_diversity(df_k, Q, sensitive_attr, l=l) if l > 1 else df_k
        if len(df_kl) == 0:
            continue
        stage = f'k={k}' if l == 1 else f'k={k}, l={l}'
        r = run_skewness_attack(df_kl, Q, sensitive_attr, tau,
                                 dataset_name, stage, plot=False)
        r.update({'k': k, 'l': l,
                   'n_retained': len(df_kl),
                   'pct_retained': round(len(df_kl)/len(df_work)*100, 1)})
        grid_results.append(r)
    return grid_results


def plot_analysis(grid_results, pipeline_results, dataset_name):
    """
    Wykres zbiorczy z dwoma panelami:
      lewy  — krzywa podatności vs k (l=1 i l=2) z siatki
      prawy — porównanie % podatnych klas i % rekordów dla 3 etapów pipeline
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Analiza podatności — {dataset_name}", fontsize=13, fontweight='bold')

    # --- Lewy panel: krzywa z siatki ---
    df_grid = pd.DataFrame([{
        'k': r['k'], 'l': r['l'],
        'pct_vulnerable': r['pct_vulnerable'],
        'pct_retained':   r['pct_retained'],
    } for r in grid_results])

    colors = {1: '#4C72B0', 2: '#C44E52'}
    ax = axes[0]
    for l_val, grp in df_grid.groupby('l'):
        grp = grp.sort_values('k')
        label = ('tylko k-anonimowość' if l_val == 1
                 else f'k-anon + l-różnorodność (l={l_val})')
        ax.plot(grp['k'], grp['pct_vulnerable'],
                marker='o', label=label, color=colors.get(l_val, 'gray'),
                linewidth=2)
    ax.set_xlabel('k')
    ax.set_ylabel('% podatnych klas')
    ax.set_title('Skuteczność ataku vs k')
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # --- Prawy panel: porównanie etapów pipeline ---
    stages = ['k-anonimowość', 'k + l-różnorodność', 'k + l + t-bliskość']
    keys   = ['k', 'kl', 'klt']
    vuln   = [pipeline_results[key]['pct_vulnerable'] for key in keys]
    retain = [pipeline_results[key]['pct_retained']   for key in keys]

    ax2 = axes[1]
    x = np.arange(len(stages))
    b1 = ax2.bar(x - 0.2, vuln,   0.35, label='% podatnych klas',
                  color='#C44E52', alpha=0.85)
    b2 = ax2.bar(x + 0.2, retain, 0.35, label='% zachowanych rekordów',
                  color='#4C72B0', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel('%')
    ax2.set_title('Etapy anonimizacji (optymalne k, l, t)')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1,
                 f'{h:.0f}%', ha='center', va='bottom', fontsize=9, color='#C44E52')
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1,
                 f'{h:.0f}%', ha='center', va='bottom', fontsize=9, color='#4C72B0')

    plt.tight_layout()
    fname = f"plots/full_analysis_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\nWykres zapisano: {fname}")
    plt.show()


def print_pipeline_summary(pipeline_results, dataset_name):
    """Drukuje podsumowanie wyników pipeline dla jednego datasetu."""
    print(f"\n{'─'*65}")
    print(f"  Podsumowanie pipeline — {dataset_name}")
    print(f"{'─'*65}")
    print(f"  {'Etap':<25} {'Rekordy%':>9} {'Klas':>6} {'Podatnych':>10} {'Max odch.':>10}")
    print(f"  {'─'*60}")
    for key, label in [('k', 'k-anonimowość'),
                        ('kl', 'k + l-różnorodność'),
                        ('klt', 'k + l + t-bliskość')]:
        r = pipeline_results[key]
        md = r['victim']['max_diff'] if r['victim'] else 0.0
        print(f"  {label:<25} {r['pct_retained']:>8.1f}% "
              f"{r['n_classes']:>6} {r['n_vulnerable']:>9} "
              f"({r['pct_vulnerable']:>5.1f}%) {md:>9.4f}")


all_pipeline_results = []


# =============================================================================
# 1. ADULT DATASET
# =============================================================================

print_section("DATASET 1/3 — ADULT (UCI id=2)")

adult = fetch_ucirepo(id=2)
df_ad = adult.data.features.copy()
df_ad['income'] = adult.data.targets
df_ad = clean_dataframe(df_ad, int_cols=['age'])
df_ad = add_anon_id(df_ad)

df_ad['education_grp'] = df_ad['education'].apply(
    lambda e: 'Higher' if e in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']
    else 'Other'
)

SENSITIVE_AD = 'income'
QI_CAT_AD   = ['education_grp', 'sex', 'race']
QI_NUM_AD   = {'age': df_ad['age']}

print(f"Rekordów: {len(df_ad)} | income: {sorted(df_ad[SENSITIVE_AD].unique())}")

# 1a. Adaptive search: k i l (bez t)
print_section("1a. Adaptive search — k i l (bez t-bliskości)")
res_ad_kl = adaptive_search(
    df=df_ad, qi_categorical=QI_CAT_AD, qi_numerical=QI_NUM_AD,
    sensitive_attr=SENSITIVE_AD,
    k_values=[2, 3, 5, 10, 15, 20],
    l_values=[2, 3, 4, 5],
    n_bins_options=[2, 3, 4, 5, 6, 8],
    use_t_closeness=False,
)
print_adaptive_results(res_ad_kl, top_n=5)

# 1b. Adaptive search: k, l i t
print_section("1b. Adaptive search — k, l i t (z t-bliskością)")
res_ad_klt = adaptive_search(
    df=df_ad, qi_categorical=QI_CAT_AD, qi_numerical=QI_NUM_AD,
    sensitive_attr=SENSITIVE_AD,
    k_values=[2, 3, 5, 10],
    l_values=[2, 3],
    n_bins_options=[2, 3, 4, 5],
    t_values=[0.1, 0.15, 0.2, 0.3],
    use_t_closeness=True,
)
print_adaptive_results(res_ad_klt, top_n=5)
best_ad = res_ad_klt.iloc[0]
k_ad = int(best_ad['k'])
l_ad = int(best_ad['l'])
t_ad = float(best_ad['t'])
age_bins_ad = int(best_ad['age_bins'])

df_ad_work, Q_AD = prepare_work_df(df_ad, QI_CAT_AD, {'age': age_bins_ad}, SENSITIVE_AD)
print(f"\nUżyte parametry: k={k_ad}, l={l_ad}, t={t_ad}, age_bins={age_bins_ad}")

# 2. Siatka
print_section("1c. Analiza siatki (k, l) — atak skośności")
grid_ad = run_grid_attacks(df_ad_work, Q_AD, SENSITIVE_AD, KL_GRID, TAU, 'Adult')

# 3. Pipeline + atak
print_section("1d. Pipeline k → k+l → k+l+t z atakiem po każdym etapie")
pipe_ad = run_pipeline_and_attack(df_ad_work, Q_AD, SENSITIVE_AD,
                                   k=k_ad, l=l_ad, t=t_ad,
                                   dataset_name='Adult', tau=TAU)
print_pipeline_summary(pipe_ad, 'Adult')
plot_analysis(grid_ad, pipe_ad, 'Adult')
all_pipeline_results.extend([pipe_ad['k'], pipe_ad['kl'], pipe_ad['klt']])


# =============================================================================
# 2. MEDICAL COST DATASET
# =============================================================================

print_section("DATASET 2/3 — MEDICAL COST (insurance.csv)")

df_med = pd.read_csv('insurance.csv')
df_med = clean_dataframe(df_med, int_cols=['age'], float_cols=['bmi'])
df_med = add_anon_id(df_med)
df_med['charges_cat'] = pd.qcut(df_med['charges'], q=4,
                                  labels=['Low', 'Medium', 'High', 'Very High'])

SENSITIVE_MED = 'charges_cat'
QI_CAT_MED   = ['sex', 'region']
QI_NUM_MED   = {'age': df_med['age'], 'bmi': df_med['bmi']}

print(f"Rekordów: {len(df_med)} | charges_cat: {list(df_med[SENSITIVE_MED].cat.categories)}")

# 1a. Adaptive search: k i l (bez t)
print_section("2a. Adaptive search — k i l (bez t-bliskości)")
res_med_kl = adaptive_search(
    df=df_med, qi_categorical=QI_CAT_MED, qi_numerical=QI_NUM_MED,
    sensitive_attr=SENSITIVE_MED,
    k_values=[2, 3, 5, 10, 15, 20],
    l_values=[2, 3, 4],
    n_bins_options=[2, 3, 4, 5, 6],
    use_t_closeness=False,
)
print_adaptive_results(res_med_kl, top_n=5)

# 1b. Adaptive search: k, l i t
print_section("2b. Adaptive search — k, l i t (z t-bliskością)")
res_med_klt = adaptive_search(
    df=df_med, qi_categorical=QI_CAT_MED, qi_numerical=QI_NUM_MED,
    sensitive_attr=SENSITIVE_MED,
    k_values=[2, 3, 5, 10],
    l_values=[2, 3],
    n_bins_options=[2, 3, 4, 5],
    t_values=[0.1, 0.15, 0.2, 0.3],
    use_t_closeness=True,
)
print_adaptive_results(res_med_klt, top_n=5)
best_med = res_med_klt.iloc[0]
k_med = int(best_med['k'])
l_med = int(best_med['l'])
t_med = float(best_med['t'])
age_bins_med = int(best_med['age_bins'])
bmi_bins_med = int(best_med['bmi_bins'])

df_med_work, Q_MED = prepare_work_df(
    df_med, QI_CAT_MED,
    {'age': age_bins_med, 'bmi': bmi_bins_med},
    SENSITIVE_MED
)
print(f"\nUżyte parametry: k={k_med}, l={l_med}, t={t_med}, "
      f"age_bins={age_bins_med}, bmi_bins={bmi_bins_med}")

# 2. Siatka
print_section("2c. Analiza siatki (k, l) — atak skośności")
grid_med = run_grid_attacks(df_med_work, Q_MED, SENSITIVE_MED, KL_GRID, TAU, 'Medical Cost')

# 3. Pipeline + atak
print_section("2d. Pipeline k → k+l → k+l+t z atakiem po każdym etapie")
pipe_med = run_pipeline_and_attack(df_med_work, Q_MED, SENSITIVE_MED,
                                    k=k_med, l=l_med, t=t_med,
                                    dataset_name='Medical Cost', tau=TAU)
print_pipeline_summary(pipe_med, 'Medical Cost')
plot_analysis(grid_med, pipe_med, 'Medical Cost')
all_pipeline_results.extend([pipe_med['k'], pipe_med['kl'], pipe_med['klt']])


# =============================================================================
# 3. GERMAN CREDIT DATASET
# =============================================================================

print_section("DATASET 3/3 — GERMAN CREDIT (UCI id=144)")

COLUMN_RENAME_GER = {
    'Attribute1': 'checking_status', 'Attribute2': 'duration',
    'Attribute3': 'credit_history',  'Attribute4': 'purpose',
    'Attribute5': 'credit_amount',   'Attribute6': 'savings_status',
    'Attribute7': 'employment',      'Attribute8': 'installment_commitment',
    'Attribute9': 'personal_status', 'Attribute10': 'other_parties',
    'Attribute11': 'residence_since','Attribute12': 'property_magnitude',
    'Attribute13': 'age',            'Attribute14': 'other_payment_plans',
    'Attribute15': 'housing',        'Attribute16': 'existing_credits',
    'Attribute17': 'job',            'Attribute18': 'num_dependents',
    'Attribute19': 'own_telephone',  'Attribute20': 'foreign_worker',
}
VALUE_LABELS_GER = {
    'checking_status': {'A11': '<0 DM', 'A12': '0-200 DM',
                        'A13': '>=200 DM', 'A14': 'no account'},
    'personal_status': {'A91': 'male divorced', 'A92': 'female div/married',
                        'A93': 'male single',   'A94': 'male married',
                        'A95': 'female single'},
    'housing':         {'A151': 'rent', 'A152': 'own', 'A153': 'for free'},
    'foreign_worker':  {'A201': 'yes',  'A202': 'no'},
}

german = fetch_ucirepo(id=144)
df_ger = german.data.features.rename(columns=COLUMN_RENAME_GER).copy()
df_ger['credit_risk'] = german.data.targets['class'].map({1: 'good', 2: 'bad'})
for col, mapping in VALUE_LABELS_GER.items():
    if col in df_ger.columns:
        df_ger[col] = df_ger[col].map(mapping).fillna(df_ger[col])

df_ger = clean_dataframe(df_ger, int_cols=['age'])
df_ger = add_anon_id(df_ger)

SENSITIVE_GER = 'checking_status'
QI_CAT_GER   = ['personal_status', 'housing', 'foreign_worker']
QI_NUM_GER   = {'age': df_ger['age']}

print(f"Rekordów: {len(df_ger)} | checking_status: {sorted(df_ger[SENSITIVE_GER].unique())}")

# 1a. Adaptive search: k i l (bez t)
print_section("3a. Adaptive search — k i l (bez t-bliskości)")
res_ger_kl = adaptive_search(
    df=df_ger, qi_categorical=QI_CAT_GER, qi_numerical=QI_NUM_GER,
    sensitive_attr=SENSITIVE_GER,
    k_values=[2, 3, 5, 10, 15, 20],
    l_values=[2, 3],
    n_bins_options=[2, 3, 4, 5, 6],
    use_t_closeness=False,
)
print_adaptive_results(res_ger_kl, top_n=5)

# 1b. Adaptive search: k, l i t
print_section("3b. Adaptive search — k, l i t (z t-bliskością)")
res_ger_klt = adaptive_search(
    df=df_ger, qi_categorical=QI_CAT_GER, qi_numerical=QI_NUM_GER,
    sensitive_attr=SENSITIVE_GER,
    k_values=[2, 3, 5, 10],
    l_values=[2, 3],
    n_bins_options=[2, 3, 4, 5],
    t_values=[0.1, 0.15, 0.2, 0.3],
    use_t_closeness=True,
)
print_adaptive_results(res_ger_klt, top_n=5)
best_ger = res_ger_klt.iloc[0]
k_ger = int(best_ger['k'])
l_ger = int(best_ger['l'])
t_ger = float(best_ger['t'])
age_bins_ger = int(best_ger['age_bins'])

df_ger_work, Q_GER = prepare_work_df(df_ger, QI_CAT_GER, {'age': age_bins_ger}, SENSITIVE_GER)
print(f"\nUżyte parametry: k={k_ger}, l={l_ger}, t={t_ger}, age_bins={age_bins_ger}")

# 2. Siatka
print_section("3c. Analiza siatki (k, l) — atak skośności")
grid_ger = run_grid_attacks(df_ger_work, Q_GER, SENSITIVE_GER, KL_GRID, TAU, 'German Credit')

# 3. Pipeline + atak
print_section("3d. Pipeline k → k+l → k+l+t z atakiem po każdym etapie")
pipe_ger = run_pipeline_and_attack(df_ger_work, Q_GER, SENSITIVE_GER,
                                    k=k_ger, l=l_ger, t=t_ger,
                                    dataset_name='German Credit', tau=TAU)
print_pipeline_summary(pipe_ger, 'German Credit')
plot_analysis(grid_ger, pipe_ger, 'German Credit')
all_pipeline_results.extend([pipe_ger['k'], pipe_ger['kl'], pipe_ger['klt']])


# =============================================================================
# ZBIORCZA TABELA — WSZYSTKIE DATASETY I ETAPY
# =============================================================================

print_section("ZBIORCZA TABELA — WSZYSTKIE DATASETY I ETAPY PIPELINE")

print(f"\n{'Dataset':<16} {'Etap':<25} {'Rekordy%':>9} {'Klas':>6} "
      f"{'Podatnych':>10} {'%podatnych':>11} {'Max odch.':>10}")
print("─" * 90)
for r in all_pipeline_results:
    md = r['victim']['max_diff'] if r['victim'] else 0.0
    print(f"{r['dataset']:<16} {r['stage']:<25} "
          f"{r['pct_retained']:>8.1f}% "
          f"{r['n_classes']:>6} {r['n_vulnerable']:>10} "
          f"{r['pct_vulnerable']:>10.1f}% {md:>10.4f}")
print("─" * 90)


# =============================================================================
# PRÓG t ELIMINUJĄCY ATAK — skanowanie wartości t
# =============================================================================

print_section("PRÓG t — PRZY JAKIM t ATAK SKOŚNOŚCI PRZESTAJE BYĆ SKUTECZNY?")

T_SCAN = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]

SCAN_CONFIGS = [
    ('Adult',         df_ad_work,  Q_AD,  SENSITIVE_AD,  k_ad,  l_ad),
    ('Medical Cost',  df_med_work, Q_MED, SENSITIVE_MED, k_med, l_med),
    ('German Credit', df_ger_work, Q_GER, SENSITIVE_GER, k_ger, l_ger),
]


def find_t_threshold(df_work, Q, sensitive_attr, k, l, t_values, tau):
    df_k  = apply_k_anonymity(df_work, Q, k=k)
    df_kl = apply_l_diversity(df_k, Q, sensitive_attr, l=l)
    if len(df_kl) == 0:
        print("  Brak rekordów po l-różnorodności.")
        return pd.DataFrame(), None

    rows = []
    threshold_t = None

    for t_val in t_values:
        df_t, gv = apply_t_closeness(df_kl, Q, sensitive_attr, t=t_val)
        if len(df_t) == 0:
            rows.append({'t': t_val, 'n_retained': 0, 'pct_retained': 0.0,
                         'n_classes': 0, 'n_vulnerable': 0,
                         'pct_vulnerable': None, 'max_diff': None})
            continue

        n_classes = df_t.groupby(Q, observed=True).ngroups
        n_vuln = 0
        max_diff_global = 0.0
        P_G = df_t[sensitive_attr].value_counts(normalize=True)

        for _, group in df_t.groupby(Q, observed=True):
            P_i = group[sensitive_attr].value_counts(normalize=True)
            md = max(abs(P_i.get(v, 0) - P_G.get(v, 0)) for v in P_G.index)
            if md > tau:
                n_vuln += 1
            max_diff_global = max(max_diff_global, md)

        pct_vuln = n_vuln / n_classes * 100 if n_classes > 0 else 0.0
        pct_ret  = len(df_t) / len(df_work) * 100

        rows.append({'t': t_val, 'n_retained': len(df_t),
                     'pct_retained': round(pct_ret, 1),
                     'n_classes': n_classes, 'n_vulnerable': n_vuln,
                     'pct_vulnerable': round(pct_vuln, 1),
                     'max_diff': round(max_diff_global, 4)})

        if n_vuln == 0 and threshold_t is None:
            threshold_t = t_val

    return pd.DataFrame(rows), threshold_t


def plot_t_threshold(results_df, threshold_t, dataset_name, tau):
    df = results_df.dropna(subset=['pct_vulnerable'])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Próg t eliminujący atak — {dataset_name} (tau={tau})",
                 fontsize=12, fontweight='bold')

    ax1.plot(df['t'], df['pct_vulnerable'], 'o-', color='#C44E52',
             linewidth=2, label='% podatnych klas')
    ax1.set_xlabel('t (próg t-bliskości)')
    ax1.set_ylabel('% podatnych klas', color='#C44E52')
    ax1.tick_params(axis='y', labelcolor='#C44E52')
    ax1.set_ylim(-5, 105)
    ax1.axhline(0, color='#C44E52', linewidth=0.8, linestyle=':')

    ax2 = ax1.twinx()
    ax2.plot(df['t'], df['pct_retained'], 's--', color='#4C72B0',
             linewidth=2, label='% zachowanych rekordów')
    ax2.set_ylabel('% zachowanych rekordów', color='#4C72B0')
    ax2.tick_params(axis='y', labelcolor='#4C72B0')
    ax2.set_ylim(-5, 105)

    ax1.axvline(tau, color='gray', linewidth=1.2, linestyle='--',
                label=f'tau={tau} (próg ataku)')
    if threshold_t is not None:
        ax1.axvline(threshold_t, color='#55A868', linewidth=1.8, linestyle='-',
                    label=f't*={threshold_t} (atak niesk.)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"plots/t_threshold_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Wykres zapisano: {fname}")
    plt.show()


print(f"\nSkanowanie t dla każdego datasetu (tau={TAU})...\n")
threshold_summary = []

for ds_name, df_w, Q_ds, sens, k_ds, l_ds in SCAN_CONFIGS:
    print(f"\n{'─'*60}")
    print(f"  {ds_name}  (k={k_ds}, l={l_ds})")
    print(f"{'─'*60}")

    scan_df, t_thresh = find_t_threshold(df_w, Q_ds, sens, k_ds, l_ds, T_SCAN, TAU)
    if scan_df.empty:
        continue

    print(f"  {'t':>6} {'Rekordy%':>9} {'Klas':>6} {'Podatnych':>10} "
          f"{'%podatnych':>11} {'Max odch.':>10}")
    print(f"  {'─'*55}")
    for _, row in scan_df.iterrows():
        if row['pct_vulnerable'] is None:
            print(f"  {row['t']:>6.2f}  — brak rekordów")
            continue
        marker = ' ← atak niesk.' if row['n_vulnerable'] == 0 else ''
        print(f"  {row['t']:>6.2f} {row['pct_retained']:>8.1f}% "
              f"{row['n_classes']:>6} {row['n_vulnerable']:>10} "
              f"{row['pct_vulnerable']:>10.1f}% {row['max_diff']:>10.4f}{marker}")

    if t_thresh is not None:
        ret = scan_df[scan_df['t'] == t_thresh]['pct_retained'].values[0]
        print(f"\n  → t* = {t_thresh}: atak nieskuteczny, zachowano {ret:.1f}% rekordów")
    else:
        print(f"\n  → Atak skuteczny dla wszystkich testowanych t (max odch. zawsze > {TAU})")

    threshold_summary.append({
        'dataset': ds_name, 'k': k_ds, 'l': l_ds,
        't_threshold': t_thresh if t_thresh is not None else '>0.40',
        'pct_retained_at_t': (
            scan_df[scan_df['t'] == t_thresh]['pct_retained'].values[0]
            if t_thresh is not None else None
        )
    })
    plot_t_threshold(scan_df, t_thresh, ds_name, TAU)


print_section("PODSUMOWANIE PROGÓW t*")
print(f"\n  {'Dataset':<16} {'k':>4} {'l':>4} {'t*':>10} {'Rekordy% przy t*':>18}")
print(f"  {'─'*56}")
for row in threshold_summary:
    ret = f"{row['pct_retained_at_t']:.1f}%" if row['pct_retained_at_t'] is not None else "—"
    print(f"  {row['dataset']:<16} {row['k']:>4} {row['l']:>4} "
          f"{str(row['t_threshold']):>10} {ret:>18}")
print(f"  {'─'*56}")
print(f"\n  Wniosek: t-bliskość eliminuje atak gdy t* ≤ tau={TAU}.")


# Zamknij plik logów
tee.close()
print(f'\nWyniki zapisane do pliku: {LOG_FILE}', file=sys.stderr)
