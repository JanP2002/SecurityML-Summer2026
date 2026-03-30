# =============================================================================
# medical_cost_dataset.py
# K-anonimowość + L-różnorodność na zbiorze Medical Cost Personal Dataset
# https://www.kaggle.com/datasets/mirichoi0218/insurance
# =============================================================================


import pandas as pd

from shared_functions import (
    add_anon_id,
    clean_dataframe,
    generalize_age,
    generate_equivalence_classes,
    apply_k_anonymity,
    apply_l_diversity,
    print_section,
    print_eq_class_preview,
    print_k_anonymity_summary,
    print_l_diversity_summary,
    print_entropy_preview,
    verify_k_anonymity,
    verify_l_diversity,
)

# =============================================================================
# 1. WCZYTANIE DANYCH
# =============================================================================
 
print_section("1. WCZYTANIE DANYCH")
 
# Umieść plik insurance.csv w tym samym folderze co skrypt
# https://www.kaggle.com/datasets/mirichoi0218/insurance
DATASET_PATH = 'insurance.csv'
 
df = pd.read_csv(DATASET_PATH)
 
print(f"Początkowa liczba rekordów: {len(df)}")
print(f"Kolumny: {list(df.columns)}")
print(df.head(3))
print()
print("Podstawowe statystyki:")
print(df.describe())


# =============================================================================
# 2. CZYSZCZENIE DANYCH
# =============================================================================

print_section("2. CZYSZCZENIE DANYCH")

df = clean_dataframe(df, int_cols=['age'], float_cols=['bmi'])


# =============================================================================
# 3. PSEUDONIMIZACJA I WYBÓR ATRYBUTÓW
# =============================================================================

print_section("3. PSEUDONIMIZACJA I WYBÓR ATRYBUTÓW")

Q         = ['age', 'sex', 'bmi_category', 'region']
SENSITIVE = 'charges'

df = add_anon_id(df)
print(f"Quasi-identyfikatory : {Q}")
print(f"Atrybut wrażliwy     : {SENSITIVE}")
print(df[['anon_ID', 'age', 'sex', 'bmi', 'region', 'charges', 'smoker']].head(3))


# =============================================================================
# 4. GENERALIZACJA
# =============================================================================

print_section("4. GENERALIZACJA")


def generalize_bmi(bmi):
    """
    Generalizuje BMI do kategorii WHO:
      < 18.5      -> Underweight
      18.5–24.9   -> Normal
      25.0–29.9   -> Overweight
      >= 30.0     -> Obese
    """
    if pd.isna(bmi):
        return 'Unknown'
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25.0:
        return 'Normal'
    elif bmi < 30.0:
        return 'Overweight'
    else:
        return 'Obese'


df['age']          = df['age'].apply(generalize_age)
df['bmi_category'] = df['bmi'].apply(generalize_bmi)

df_anon = df[['anon_ID'] + Q + [SENSITIVE]].copy()

print("Dane po generalizacji (wiek w dekadach, BMI w kategoriach WHO):")
print(df_anon.head(5))
print(f"\nUnikalne wartości 'age'          : {sorted(df_anon['age'].unique())}")
print(f"Unikalne wartości 'bmi_category' : {df_anon['bmi_category'].unique()}")
print(f"Unikalne wartości 'region'       : {df_anon['region'].unique()}")


# =============================================================================
# 5. KLASY RÓWNOWAŻNOŚCI (przed k-anonimowością)
# =============================================================================

print_section("5. KLASY RÓWNOWAŻNOŚCI — PRZED K-ANONIMIZACJĄ")

all_eq_classes = generate_equivalence_classes(df_anon, Q)
print_eq_class_preview(all_eq_classes, n=3)


# =============================================================================
# 6. K-ANONIMOWOŚĆ DLA k = 10
# =============================================================================

print_section("6. K-ANONIMOWOŚĆ  (k = 10)")

K = 10
df_k10 = apply_k_anonymity(df_anon, Q, k=K)

print_k_anonymity_summary(df_anon, df_k10, k=K)
print()
verify_k_anonymity(df_k10, Q, k=K)

print("\nPrzykładowa klasa (wiek 30-39, mężczyzna, Obese, southeast):")
sample = df_k10[
    (df_k10['age'] == '30-39') &
    (df_k10['sex'] == 'male') &
    (df_k10['bmi_category'] == 'Obese') &
    (df_k10['region'] == 'southeast')
]
print(sample.head(10))


# =============================================================================
# 7. K-ANONIMOWOŚĆ DLA k = 3
# =============================================================================

print_section("7. K-ANONIMOWOŚĆ  (k = 3)")

K = 3
df_k3 = apply_k_anonymity(df_anon, Q, k=K)

print_k_anonymity_summary(df_anon, df_k3, k=K)
print()
verify_k_anonymity(df_k3, Q, k=K)

print("\nPrzykładowa klasa (wiek 30-39, mężczyzna, Obese, southeast):")
sample = df_k3[
    (df_k3['age'] == '30-39') &
    (df_k3['sex'] == 'male') &
    (df_k3['bmi_category'] == 'Obese') &
    (df_k3['region'] == 'southeast')
]
print(sample.head(10))


# =============================================================================
# 8. L-RÓŻNORODNOŚĆ NA ZBIORZE k = 10
# =============================================================================

print_section("8. L-RÓŻNORODNOŚĆ  (k = 10, l = 2)")

L = 2
df_k10_l2 = apply_l_diversity(df_k10, Q, SENSITIVE, l=L)

print_l_diversity_summary(df_k10, df_k10_l2, l=L)
print()
verify_l_diversity(df_k10_l2, Q, SENSITIVE, l=L)

print()
print_entropy_preview(df_k10_l2, Q, SENSITIVE, n=5)


# =============================================================================
# 9. L-RÓŻNORODNOŚĆ NA ZBIORZE k = 3
# =============================================================================

print_section("9. L-RÓŻNORODNOŚĆ  (k = 3, l = 2)")

df_k3_l2 = apply_l_diversity(df_k3, Q, SENSITIVE, l=L)

print_l_diversity_summary(df_k3, df_k3_l2, l=L)
print()
verify_l_diversity(df_k3_l2, Q, SENSITIVE, l=L)

print()
print_entropy_preview(df_k3_l2, Q, SENSITIVE, n=5)
