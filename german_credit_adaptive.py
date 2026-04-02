# =============================================================================
# german_credit_adaptive.py
# Adaptacyjne wyszukiwanie optymalnych parametrów k, l, t i podziałów
# dla zbioru danych German Credit Data (Statlog / UCI, id=144)
#
# Niewielki zbiór (1000 rekordów) — anonimizacja szybko niszczy użyteczność.
# Szukamy optymalnego balansu w przestrzeni (k, l, t).
# =============================================================================

import pandas as pd
from ucimlrepo import fetch_ucirepo

from shared_functions import (
    add_anon_id,
    clean_dataframe,
    apply_k_anonymity,
    apply_l_diversity,
    apply_t_closeness,
    print_section,
    print_entropy_preview,
    verify_k_anonymity,
    verify_l_diversity,
    verify_t_closeness,
    print_t_closeness_preview,
    adaptive_search,
    print_adaptive_results,
    generalize_numeric_qcut,
)

from german_credit_dataset import COLUMN_RENAME, VALUE_LABELS


# =============================================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# =============================================================================

print_section("1. WCZYTANIE I PRZYGOTOWANIE DANYCH")

german = fetch_ucirepo(id=144)
df = german.data.features.rename(columns=COLUMN_RENAME).copy()
df['credit_risk'] = german.data.targets['class'].map({1: 'good', 2: 'bad'})

for col, mapping in VALUE_LABELS.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(df[col])

print(f"Początkowa liczba rekordów: {len(df)}")
df = clean_dataframe(df, int_cols=['age'])


# =============================================================================
# 2. PSEUDONIMIZACJA
# =============================================================================

print_section("2. PSEUDONIMIZACJA")

df = add_anon_id(df)

SENSITIVE       = 'credit_risk'
QI_CATEGORICAL  = ['personal_status', 'housing', 'foreign_worker']

# Kolumna numeryczna do adaptacyjnej generalizacji
QI_NUMERICAL    = {'age': df['age']}

print(f"Quasi-identyfikatory kategoryczne : {QI_CATEGORICAL}")
print(f"Quasi-identyfikatory numeryczne   : {list(QI_NUMERICAL.keys())}")
print(f"Atrybut wrażliwy                  : {SENSITIVE}")
print(f"\nZakres age : {df['age'].min()}\u2013{df['age'].max()}")


# =============================================================================
# 3. ADAPTACYJNE WYSZUKIWANIE
# =============================================================================

print_section("3. ADAPTACYJNE WYSZUKIWANIE")

# Przestrzeń poszukiwań — mały dataset, więc nie testujemy zbyt dużych k
K_VALUES       = [2, 3, 5, 10]
L_VALUES       = [2, 3]
N_BINS_OPTIONS = [2, 3, 4, 5, 6]
T_VALUES       = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

df_results = adaptive_search(
    df             = df,
    qi_categorical = QI_CATEGORICAL,
    qi_numerical   = QI_NUMERICAL,
    sensitive_attr = SENSITIVE,
    k_values       = K_VALUES,
    l_values       = L_VALUES,
    n_bins_options = N_BINS_OPTIONS,
    t_values       = T_VALUES,
    top_n          = 15,
)

print_adaptive_results(df_results, top_n=15)


# =============================================================================
# 4. URUCHOMIENIE OPTYMALNEJ KOMBINACJI
# =============================================================================

print_section("4. WYNIK DLA OPTYMALNEJ KOMBINACJI")

if not df_results.empty:
    best      = df_results.iloc[0]
    k_best    = int(best['k'])
    l_best    = int(best['l'])
    t_best    = float(best['t'])
    age_bins  = int(best['age_bins'])

    print(f"Parametry: k={k_best}, l={l_best}, t={t_best}, age_bins={age_bins}")

    # Odtwórz generalizację z optymalnymi parametrami
    df['age_gen'] = generalize_numeric_qcut(df['age'], age_bins)
    Q_best = QI_CATEGORICAL + ['age_gen']

    df_work = df[['anon_ID'] + Q_best + [SENSITIVE]].copy()

    df_k  = apply_k_anonymity(df_work, Q_best, k=k_best)
    df_kl = apply_l_diversity(df_k, Q_best, SENSITIVE, l=l_best)
    df_klt = apply_t_closeness(df_kl, Q_best, SENSITIVE, t=t_best)

    print(f"\nRekordy przed                     : {len(df)}")
    print(f"Po k-anonimowości ({k_best})           : {len(df_k)}")
    print(f"Po l-różnorodności ({l_best})           : {len(df_kl)}")
    print(f"Po t-bliskości ({t_best})            : {len(df_klt)}")
    print()
    verify_k_anonymity(df_klt, Q_best, k_best)
    print()
    verify_l_diversity(df_klt, Q_best, SENSITIVE, l_best)
    print()
    verify_t_closeness(df_klt, Q_best, SENSITIVE, t_best)
    print()
    print_entropy_preview(df_klt, Q_best, SENSITIVE, n=5)
    print()
    print_t_closeness_preview(df_klt, Q_best, SENSITIVE, n=5)

    print("\nPrzykładowe rekordy po anonimizacji:")
    print(df_klt.head(10).to_string(index=False))

    # Pokaż jakie przedziały zostały wygenerowane przez qcut
    print("\nWygenerowane przedziały age:")
    print(df.groupby('age_gen', observed=True)['age'].agg(['min', 'max', 'count']))


# =============================================================================
# 5. PORÓWNANIE: RÓŻNE SCENARIUSZE ANONIMIZACJI
# =============================================================================

print_section("5. PORÓWNANIE SCENARIUSZY ANONIMIZACJI")

if not df_results.empty:
    from shared_functions import generalize_age

    df['age_manual'] = df['age'].apply(generalize_age)
    Q_manual = QI_CATEGORICAL + ['age_manual']
    df_manual = df[['anon_ID'] + Q_manual + [SENSITIVE]].copy()

    total = len(df)
    print(f"{'Scenariusz':<40} {'Rekordy':>8} {'%':>7}")
    print("-" * 57)

    for k_cmp, l_cmp, t_cmp in [(3, 2, 0.2), (3, 2, 0.3), (5, 2, 0.2), (5, 2, 0.3)]:
        dk = apply_k_anonymity(df_manual, Q_manual, k=k_cmp)
        dkl = apply_l_diversity(dk, Q_manual, SENSITIVE, l=l_cmp)
        dklt = apply_t_closeness(dkl, Q_manual, SENSITIVE, t=t_cmp)
        label = f"Ręczny  (k={k_cmp}, l={l_cmp}, t={t_cmp})"
        print(f"{label:<40} {len(dklt):>8} {len(dklt)/total*100:>6.1f}%")

    print()
    label = f"Adaptacyjny (k={k_best}, l={l_best}, t={t_best})"
    print(f"{label:<40} {len(df_klt):>8} {len(df_klt)/total*100:>6.1f}%")
