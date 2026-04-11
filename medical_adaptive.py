# =============================================================================
# medical_adaptive.py
# Adaptacyjne wyszukiwanie optymalnych parametrów k, l i podziałów
# dla zbioru danych Medical Cost Personal Dataset (Kaggle)
# =============================================================================

import pandas as pd

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
    generalize_numeric_qcut,
)


# =============================================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# =============================================================================

print_section("1. WCZYTANIE I PRZYGOTOWANIE DANYCH")

DATASET_PATH = 'insurance.csv'
df = pd.read_csv(DATASET_PATH)

print(f"Początkowa liczba rekordów: {len(df)}")
df = clean_dataframe(df, int_cols=['age'], float_cols=['bmi'])


# =============================================================================
# 2. PSEUDONIMIZACJA
# =============================================================================

print_section("2. PSEUDONIMIZACJA")

df = add_anon_id(df)

# Kolumny kategoryczne QI — sex i region są już kategoriami, zostawiamy as-is.
SENSITIVE       = 'charges'
QI_CATEGORICAL  = ['sex', 'region']

# Dwie kolumny numeryczne do adaptacyjnej generalizacji.
# age: 18–64, bmi: 15.96–53.13
QI_NUMERICAL    = {
    'age': df['age'],
    'bmi': df['bmi'],
}

print(f"Quasi-identyfikatory kategoryczne : {QI_CATEGORICAL}")
print(f"Quasi-identyfikatory numeryczne   : {list(QI_NUMERICAL.keys())}")
print(f"Atrybut wrażliwy                  : {SENSITIVE}")
print(f"\nZakres age : {df['age'].min()}–{df['age'].max()}")
print(f"Zakres bmi : {df['bmi'].min():.2f}–{df['bmi'].max():.2f}")


# =============================================================================
# 3. ADAPTACYJNE WYSZUKIWANIE
# =============================================================================

print_section("3. ADAPTACYJNE WYSZUKIWANIE")

# Przestrzeń poszukiwań
# Medical Cost jest małym datasetem (1337 rekordów) więc nie ma sensu
# testować zbyt dużych k — przy k=20 i 4 QI prawie wszystko odpada.
K_VALUES       = [2, 3, 5, 10]
L_VALUES       = [2, 3, 4]
N_BINS_OPTIONS = [2, 3, 4, 5, 6]

# Przestrzeń: 4 × 3 × 5² = 300 kombinacji (l > k pomijane) → ~180 efektywnie
df_results = adaptive_search(
    df             = df,
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
    best      = df_results.iloc[0]
    k_best    = int(best['k'])
    l_best    = int(best['l'])
    age_bins  = int(best['age_bins'])
    bmi_bins  = int(best['bmi_bins'])

    print(f"Parametry: k={k_best}, l={l_best}, age_bins={age_bins}, bmi_bins={bmi_bins}")

    # Odtwórz generalizację z optymalnymi parametrami
    df['age_gen'] = generalize_numeric_qcut(df['age'], age_bins)
    df['bmi_gen'] = generalize_numeric_qcut(df['bmi'], bmi_bins)

    Q_best = QI_CATEGORICAL + ['age_gen', 'bmi_gen']
    df_work = df[['anon_ID'] + Q_best + [SENSITIVE]].copy()

    df_k  = apply_k_anonymity(df_work, Q_best, k=k_best)
    df_kl = apply_l_diversity(df_k, Q_best, SENSITIVE, l=l_best)

    print(f"\nRekordy przed                     : {len(df)}")
    print(f"Po k-anonimowości ({k_best})           : {len(df_k)}")
    print(f"Po l-różnorodności ({l_best})           : {len(df_kl)}")
    print()
    verify_k_anonymity(df_kl, Q_best, k_best)
    print()
    verify_l_diversity(df_kl, Q_best, SENSITIVE, l_best)
    print()
    print_entropy_preview(df_kl, Q_best, SENSITIVE, n=5)

    print("\nPrzykładowe rekordy po anonimizacji:")
    print(df_kl.head(10).to_string(index=False))

    # Pokaż jakie przedziały zostały wygenerowane przez qcut
    print("\nWygenerowane przedziały age:")
    print(df.groupby('age_gen', observed=True)['age'].agg(['min', 'max', 'count']))
    print("\nWygenerowane przedziały bmi:")
    print(df.groupby('bmi_gen', observed=True)['bmi'].agg(['min', 'max', 'count']))


# =============================================================================
# 5. PORÓWNANIE Z PODSTAWOWYM PIPELINE'EM
# =============================================================================

print_section("5. PORÓWNANIE: ADAPTACYJNY vs RĘCZNY PIPELINE")

if not df_results.empty:
    # Ręczny pipeline z podstawowych pliku — dekady wiekowe + kategorie WHO BMI
    from shared_functions import generalize_age

    def generalize_bmi_who(bmi):
        if bmi < 18.5:   return 'Underweight'
        elif bmi < 25.0: return 'Normal'
        elif bmi < 30.0: return 'Overweight'
        else:            return 'Obese'

    df['age_manual'] = df['age'].apply(generalize_age)
    df['bmi_manual'] = df['bmi'].apply(generalize_bmi_who)
    Q_manual = QI_CATEGORICAL + ['age_manual', 'bmi_manual']
    df_manual = df[['anon_ID'] + Q_manual + [SENSITIVE]].copy()

    for k_cmp, l_cmp in [(3, 2), (10, 2)]:
        dk = apply_k_anonymity(df_manual, Q_manual, k=k_cmp)
        dkl = apply_l_diversity(dk, Q_manual, SENSITIVE, l=l_cmp)
        print(f"Ręczny  (k={k_cmp}, l={l_cmp}): {len(dkl):>5} rekordów ({len(dkl)/len(df)*100:.1f}%)")

    print()
    print(f"Adaptacyjny (k={k_best}, l={l_best}): {len(df_kl):>5} rekordów ({len(df_kl)/len(df)*100:.1f}%)")