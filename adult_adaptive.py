# =============================================================================
# adult_adaptive.py
# Adaptacyjne wyszukiwanie optymalnych parametrów k, l i podziałów
# dla zbioru danych Adult (UCI Repository)
# =============================================================================

import pandas as pd
from ucimlrepo import fetch_ucirepo

from shared_functions import (
    add_anon_id,
    clean_dataframe,
    apply_k_anonymity,
    apply_l_diversity,
    print_section,
    print_entropy_preview,
    verify_k_anonymity,
    verify_l_diversity,
    adaptive_search,
    print_adaptive_results,
)


# =============================================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# =============================================================================

print_section("1. WCZYTANIE I PRZYGOTOWANIE DANYCH")

adult = fetch_ucirepo(id=2)
df = adult.data.features.copy()
df['income'] = adult.data.targets

print(f"Początkowa liczba rekordów: {len(df)}")
df = clean_dataframe(df, int_cols=['age'])


# =============================================================================
# 2. PSEUDONIMIZACJA
# =============================================================================

print_section("2. PSEUDONIMIZACJA")

df = add_anon_id(df)

# Kolumny kategoryczne QI — zostawiamy bez dodatkowej generalizacji.
# sex i race są już kategoriami. education ma 16 wartości — grupujemy
# w dwie klasy tak samo jak w podstawowym pipeline'ie, bo mniej niż 2
# opcje nie dają sensu, a więcej nie ma sensu przy tych danych.
df['education_grp'] = df['education'].apply(
    lambda e: 'Higher' if e in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']
    else 'Other'
)

SENSITIVE       = 'income'
QI_CATEGORICAL  = ['education_grp', 'sex', 'race']

# Kolumny numeryczne do adaptacyjnej generalizacji
QI_NUMERICAL    = {'age': df['age']}

print(f"Quasi-identyfikatory kategoryczne : {QI_CATEGORICAL}")
print(f"Quasi-identyfikatory numeryczne   : {list(QI_NUMERICAL.keys())}")
print(f"Atrybut wrażliwy                  : {SENSITIVE}")


# =============================================================================
# 3. ADAPTACYJNE WYSZUKIWANIE
# =============================================================================

print_section("3. ADAPTACYJNE WYSZUKIWANIE")

# Przestrzeń poszukiwań
K_VALUES      = [2, 3, 5, 10, 15, 20]
L_VALUES      = [2, 3, 4, 5]
N_BINS_OPTIONS = [2, 3, 4, 5, 6, 8]

# Przestrzeń: 6 × 4 × 6 = 144 kombinacje (l > k pomijane)
df_results = adaptive_search(
    df            = df,
    qi_categorical = QI_CATEGORICAL,
    qi_numerical   = QI_NUMERICAL,
    sensitive_attr = SENSITIVE,
    k_values       = K_VALUES,
    l_values       = L_VALUES,
    n_bins_options = N_BINS_OPTIONS,
    top_n          = 15,
)

print_adaptive_results(df_results, top_n=15)


# =============================================================================
# 4. URUCHOMIENIE OPTYMALNEJ KOMBINACJI
# =============================================================================

print_section("4. WYNIK DLA OPTYMALNEJ KOMBINACJI")

if not df_results.empty:
    best = df_results.iloc[0]
    k_best     = int(best['k'])
    l_best     = int(best['l'])
    age_bins   = int(best['age_bins'])

    print(f"Parametry: k={k_best}, l={l_best}, age_bins={age_bins}")

    # Odtwórz generalizację z optymalnymi parametrami
    from shared_functions import generalize_numeric_qcut

    df['age_gen'] = generalize_numeric_qcut(df['age'], age_bins)
    Q_best = QI_CATEGORICAL + ['age_gen']

    df_work = df[['anon_ID'] + Q_best + [SENSITIVE]].copy()

    df_k = apply_k_anonymity(df_work, Q_best, k=k_best)
    df_kl = apply_l_diversity(df_k, Q_best, SENSITIVE, l=l_best)

    print(f"\nRekordy przed : {len(df)}")
    print(f"Po k-anonimowości ({k_best}) : {len(df_k)}")
    print(f"Po l-różnorodności ({l_best}) : {len(df_kl)}")
    print()
    verify_k_anonymity(df_kl, Q_best, k_best)
    print()
    verify_l_diversity(df_kl, Q_best, SENSITIVE, l_best)
    print()
    print_entropy_preview(df_kl, Q_best, SENSITIVE, n=5)

    print("\nPrzykładowe rekordy po anonimizacji:")
    print(df_kl.head(10).to_string(index=False))

