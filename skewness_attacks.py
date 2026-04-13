# =============================================================================
# skewness_attacks.py
#
# Atak skośności (Skewness Attack) na trzech zbiorach danych:
#   1. Adult (UCI id=2)       — atrybut wrażliwy: income
#   2. Medical Cost           — atrybut wrażliwy: charges_cat (kwartyle)
#   3. German Credit (UCI id=144) — atrybut wrażliwy: checking_status
#
# Pipeline dla każdego datasetu:
#   1. Wczytanie i czyszczenie danych
#   2. Adaptacyjne wyszukiwanie optymalnych (k, l, bins) — bez t-bliskości
#   3. Zastosowanie optymalnych parametrów:
#        (A) k-anonimowość
#        (B) k-anonimowość + l-różnorodność
#   4. Atak skośności na etapach A i B
#   5. Zbiorcza tabela wyników
# =============================================================================

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

from shared_functions import (
    add_anon_id,
    clean_dataframe,
    generalize_numeric_qcut,
    apply_k_anonymity,
    apply_l_diversity,
    print_section,
    run_skewness_attack,
    print_attack_summary_table,
    adaptive_search,
    print_adaptive_results,
)

TAU = 0.10   # próg ataku skośności — 10% odchylenie od rozkładu globalnego

all_results = []


# =============================================================================
# FUNKCJA POMOCNICZA — odtwarza anonimizację z optymalnych parametrów
# =============================================================================

def apply_optimal(df, qi_categorical, qi_numerical_cols,
                  best_row, sensitive_attr):
    """
    Odtwarza generalizację i anonimizację dla wiersza wyników z adaptive_search.
    Zwraca (df_k, df_kl, Q_best).
    """
    k_best = int(best_row['k'])
    l_best = int(best_row['l'])

    # Generalizacja kolumn numerycznych
    gen_col_names = []
    df_work = df[['anon_ID'] + qi_categorical + [sensitive_attr]].copy()

    for col in qi_numerical_cols:
        n_bins  = int(best_row[f'{col}_bins'])
        col_gen = f'{col}_gen'
        df_work[col_gen] = generalize_numeric_qcut(df[col], n_bins).values
        gen_col_names.append(col_gen)

    Q_best = qi_categorical + gen_col_names

    df_k  = apply_k_anonymity(df_work, Q_best, k=k_best)
    df_kl = apply_l_diversity(df_k, Q_best, sensitive_attr, l=l_best)

    return df_k, df_kl, Q_best, k_best, l_best


# =============================================================================
# 1. ADULT DATASET
# =============================================================================

print_section("DATASET 1/3 — ADULT (UCI id=2)")

adult   = fetch_ucirepo(id=2)
df_ad   = adult.data.features.copy()
df_ad['income'] = adult.data.targets
df_ad   = clean_dataframe(df_ad, int_cols=['age'])
df_ad   = add_anon_id(df_ad)

df_ad['education_grp'] = df_ad['education'].apply(
    lambda e: 'Higher' if e in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school'] else 'Other'
)

SENSITIVE_AD   = 'income'
QI_CAT_AD      = ['education_grp', 'sex', 'race']
QI_NUM_AD      = {'age': df_ad['age']}

# Adaptacyjne wyszukiwanie — t-bliskość wyłączona
print("\n--- Adaptacyjne wyszukiwanie (bez t-bliskości) ---")
results_ad = adaptive_search(
    df             = df_ad,
    qi_categorical = QI_CAT_AD,
    qi_numerical   = QI_NUM_AD,
    sensitive_attr = SENSITIVE_AD,
    k_values       = [2, 3, 5, 10, 15, 20],
    l_values       = [2, 3, 4, 5],
    n_bins_options = [2, 3, 4, 5, 6, 8],
    use_t_closeness = False,          # wyłącza t-bliskość
)
print_adaptive_results(results_ad, top_n=5)

# Optymalna kombinacja → anonimizacja
best_ad = results_ad.iloc[0]
df_ad_k, df_ad_kl, Q_ad, k_ad, l_ad = apply_optimal(
    df_ad, QI_CAT_AD, list(QI_NUM_AD.keys()), best_ad, SENSITIVE_AD
)

print(f"\nOptymalne parametry: k={k_ad}, l={l_ad}, age_bins={int(best_ad['age_bins'])}")
print(f"Po k-anonimowości  : {len(df_ad_k)} rekordów")
print(f"Po l-różnorodności : {len(df_ad_kl)} rekordów")

# Ataki
r_ad_k  = run_skewness_attack(df_ad_k,  Q_ad, SENSITIVE_AD, tau=TAU,
                               dataset_name='Adult', stage_name=f'k={k_ad}')
r_ad_kl = run_skewness_attack(df_ad_kl, Q_ad, SENSITIVE_AD, tau=TAU,
                               dataset_name='Adult', stage_name=f'k={k_ad}, l={l_ad}')

all_results.extend([r_ad_k, r_ad_kl])


# =============================================================================
# 2. MEDICAL COST DATASET
# =============================================================================

print_section("DATASET 2/3 — MEDICAL COST (insurance.csv)")

try:
    df_med = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku insurance.csv.")
    df_med = None

if df_med is not None:
    df_med = clean_dataframe(df_med, int_cols=['age'], float_cols=['bmi'])
    df_med = add_anon_id(df_med)

    # charges (float) → 4 kategorie kwartylowe — konieczne dla l-różnorodności
    df_med['charges_cat'] = pd.qcut(
        df_med['charges'], q=4, labels=['Low', 'Medium', 'High', 'Very High']
    )

    SENSITIVE_MED  = 'charges_cat'
    QI_CAT_MED     = ['sex', 'region']
    QI_NUM_MED     = {'age': df_med['age'], 'bmi': df_med['bmi']}

    # Adaptacyjne wyszukiwanie — t-bliskość wyłączona
    print("\n--- Adaptacyjne wyszukiwanie (bez t-bliskości) ---")
    results_med = adaptive_search(
        df             = df_med,
        qi_categorical = QI_CAT_MED,
        qi_numerical   = QI_NUM_MED,
        sensitive_attr = SENSITIVE_MED,
        k_values       = [2, 3, 5, 10, 15, 20],
        l_values       = [2, 3, 4, 5],
        n_bins_options = [2, 3, 4, 5, 6, 8],
        use_t_closeness = False,      # wyłącza t-bliskość
    )
    print_adaptive_results(results_med, top_n=5)

    best_med = results_med.iloc[0]
    df_med_k, df_med_kl, Q_med, k_med, l_med = apply_optimal(
        df_med, QI_CAT_MED, list(QI_NUM_MED.keys()), best_med, SENSITIVE_MED
    )

    print(f"\nOptymalne parametry: k={k_med}, l={l_med}, "
          f"age_bins={int(best_med['age_bins'])}, bmi_bins={int(best_med['bmi_bins'])}")
    print(f"Po k-anonimowości  : {len(df_med_k)} rekordów")
    print(f"Po l-różnorodności : {len(df_med_kl)} rekordów")

    r_med_k  = run_skewness_attack(df_med_k,  Q_med, SENSITIVE_MED, tau=TAU,
                                    dataset_name='Medical Cost', stage_name=f'k={k_med}')
    r_med_kl = run_skewness_attack(df_med_kl, Q_med, SENSITIVE_MED, tau=TAU,
                                    dataset_name='Medical Cost', stage_name=f'k={k_med}, l={l_med}')

    all_results.extend([r_med_k, r_med_kl])


# =============================================================================
# 3. GERMAN CREDIT DATASET
# =============================================================================

print_section("DATASET 3/3 — GERMAN CREDIT (UCI id=144)")

COLUMN_RENAME = {
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
VALUE_LABELS = {
    'checking_status': {'A11': '<0 DM', 'A12': '0-200 DM',
                        'A13': '>=200 DM', 'A14': 'no account'},
    'personal_status': {'A91': 'male divorced', 'A92': 'female div/married',
                        'A93': 'male single',   'A94': 'male married',
                        'A95': 'female single'},
    'housing':         {'A151': 'rent', 'A152': 'own', 'A153': 'for free'},
    'foreign_worker':  {'A201': 'yes',  'A202': 'no'},
}

german = fetch_ucirepo(id=144)
df_ger = german.data.features.rename(columns=COLUMN_RENAME).copy()
df_ger['credit_risk'] = german.data.targets['class'].map({1: 'good', 2: 'bad'})

for col, mapping in VALUE_LABELS.items():
    if col in df_ger.columns:
        df_ger[col] = df_ger[col].map(mapping).fillna(df_ger[col])

df_ger = clean_dataframe(df_ger, int_cols=['age'])
df_ger = add_anon_id(df_ger)

SENSITIVE_GER  = 'checking_status'
QI_CAT_GER     = ['personal_status', 'housing', 'foreign_worker']
QI_NUM_GER     = {'age': df_ger['age']}

# Adaptacyjne wyszukiwanie — t-bliskość wyłączona
print("\n--- Adaptacyjne wyszukiwanie (bez t-bliskości) ---")
results_ger = adaptive_search(
    df             = df_ger,
    qi_categorical = QI_CAT_GER,
    qi_numerical   = QI_NUM_GER,
    sensitive_attr = SENSITIVE_GER,
    k_values       = [2, 3, 5, 10, 15, 20],
    l_values       = [2, 3, 4, 5],
    n_bins_options = [2, 3, 4, 5, 6, 8],
    use_t_closeness = False,          # wyłącza t-bliskość
)
print_adaptive_results(results_ger, top_n=5)

best_ger = results_ger.iloc[0]
df_ger_k, df_ger_kl, Q_ger, k_ger, l_ger = apply_optimal(
    df_ger, QI_CAT_GER, list(QI_NUM_GER.keys()), best_ger, SENSITIVE_GER
)

print(f"\nOptymalne parametry: k={k_ger}, l={l_ger}, age_bins={int(best_ger['age_bins'])}")
print(f"Po k-anonimowości  : {len(df_ger_k)} rekordów")
print(f"Po l-różnorodności : {len(df_ger_kl)} rekordów")

r_ger_k  = run_skewness_attack(df_ger_k,  Q_ger, SENSITIVE_GER, tau=TAU,
                                dataset_name='German Credit', stage_name=f'k={k_ger}')
r_ger_kl = run_skewness_attack(df_ger_kl, Q_ger, SENSITIVE_GER, tau=TAU,
                                dataset_name='German Credit', stage_name=f'k={k_ger}, l={l_ger}')

all_results.extend([r_ger_k, r_ger_kl])


# =============================================================================
# ZBIORCZA TABELA WYNIKÓW — do sprawozdania
# =============================================================================

print_attack_summary_table(all_results)
