# =============================================================================
# adult_dataset.py
# K-anonimowość + L-różnorodność na zbiorze danych Adult (UCI Repository)
# Autor oryginału: Jan Poręba  |  Przepisano do wspólnego pipeline'u
# =============================================================================

import pandas as pd
from ucimlrepo import fetch_ucirepo

from shared_functions import (
    add_anon_id,
    clean_dataframe,
    generalize_age,
    generate_equivalence_classes,
    apply_k_anonymity,
    apply_l_diversity,
    apply_t_closeness,
    print_section,
    print_eq_class_preview,
    print_k_anonymity_summary,
    print_l_diversity_summary,
    print_t_closeness_summary,
    print_entropy_preview,
    verify_k_anonymity,
    verify_l_diversity,
    verify_t_closeness,
    print_t_closeness_preview,
)


# =============================================================================
# 1. WCZYTANIE DANYCH
# =============================================================================

print_section("1. WCZYTANIE DANYCH")

adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets

df = X.copy()
df['income'] = y

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

df = clean_dataframe(df, int_cols=['age'])


# =============================================================================
# 3. PSEUDONIMIZACJA I WYBÓR ATRYBUTÓW
# =============================================================================

print_section("3. PSEUDONIMIZACJA I WYBÓR ATRYBUTÓW")

Q         = ['age', 'education', 'sex', 'race']
SENSITIVE = 'income'

df = add_anon_id(df)
df_anon = df[['anon_ID'] + Q + [SENSITIVE]].copy()

print(f"Quasi-identyfikatory : {Q}")
print(f"Atrybut wrażliwy     : {SENSITIVE}")
print(df_anon.head(3))


# =============================================================================
# 4. GENERALIZACJA
# =============================================================================

print_section("4. GENERALIZACJA")


def generalize_education(edu):
    """Grupuje wykształcenie w dwie szerokie kategorie."""
    higher = ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']
    return 'Higher Education' if edu in higher else 'Up to College'


df_anon['age']       = df_anon['age'].apply(generalize_age)
df_anon['education'] = df_anon['education'].apply(generalize_education)

print("Dane po generalizacji (wiek w dekadach, wykształcenie w 2 kategoriach):")
print(df_anon.head(5))
print(f"\nUnikalne wartości 'age'       : {sorted(df_anon['age'].unique())}")
print(f"Unikalne wartości 'education' : {df_anon['education'].unique()}")


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

print("\nPrzykładowa klasa (wiek 30-39, kobieta, biała, wyższe wykształcenie):")
sample = df_k10[
    (df_k10['age'] == '30-39') &
    (df_k10['sex'] == 'Female') &
    (df_k10['race'] == 'White') &
    (df_k10['education'] == 'Higher Education')
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

print("\nPrzykładowa klasa (wiek 30-39, kobieta, biała, wyższe wykształcenie):")
sample = df_k3[
    (df_k3['age'] == '30-39') &
    (df_k3['sex'] == 'Female') &
    (df_k3['race'] == 'White') &
    (df_k3['education'] == 'Higher Education')
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


# =============================================================================
# 10. T-BLISKOŚĆ NA ZBIORZE k = 10, l = 2
# =============================================================================

print_section("10. T-BLISKOŚĆ  (k = 10, l = 2, t = 0.2)")

T = 0.2
df_k10_l2_t = apply_t_closeness(df_k10_l2, Q, SENSITIVE, t=T)

print_t_closeness_summary(df_k10_l2, df_k10_l2_t, t=T)
print()
verify_t_closeness(df_k10_l2_t, Q, SENSITIVE, t=T)

print()
print_t_closeness_preview(df_k10_l2_t, Q, SENSITIVE, n=5)


# =============================================================================
# 11. T-BLISKOŚĆ NA ZBIORZE k = 3, l = 2
# =============================================================================

print_section("11. T-BLISKOŚĆ  (k = 3, l = 2, t = 0.2)")

df_k3_l2_t = apply_t_closeness(df_k3_l2, Q, SENSITIVE, t=T)

print_t_closeness_summary(df_k3_l2, df_k3_l2_t, t=T)
print()
verify_t_closeness(df_k3_l2_t, Q, SENSITIVE, t=T)

print()
print_t_closeness_preview(df_k3_l2_t, Q, SENSITIVE, n=5)
