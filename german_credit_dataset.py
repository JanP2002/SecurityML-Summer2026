# =============================================================================
# german_credit_dataset.py
# K-anonimowość + L-różnorodność + T-bliskość na zbiorze German Credit Data
# https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
#
# Niewielki (1000 rekordów), wysoce wrażliwy zbiór finansowy.
# Ze względu na ograniczoną wielkość próbek, anonimizacja natychmiastowo
# niszczy użyteczność danych — wymagające środowisko dla balansu (k, l, t).
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
# MAPOWANIA ATRYBUTÓW (UCI Statlog -> czytelne nazwy)
# =============================================================================

COLUMN_RENAME = {
    'Attribute1':  'checking_status',
    'Attribute2':  'duration',
    'Attribute3':  'credit_history',
    'Attribute4':  'purpose',
    'Attribute5':  'credit_amount',
    'Attribute6':  'savings_status',
    'Attribute7':  'employment',
    'Attribute8':  'installment_rate',
    'Attribute9':  'personal_status',
    'Attribute10': 'other_parties',
    'Attribute11': 'residence_since',
    'Attribute12': 'property_magnitude',
    'Attribute13': 'age',
    'Attribute14': 'other_payment_plans',
    'Attribute15': 'housing',
    'Attribute16': 'existing_credits',
    'Attribute17': 'job',
    'Attribute18': 'num_dependents',
    'Attribute19': 'own_telephone',
    'Attribute20': 'foreign_worker',
}

VALUE_LABELS = {
    'checking_status': {
        'A11': '<0 DM', 'A12': '0-200 DM', 'A13': '>=200 DM', 'A14': 'brak konta',
    },
    'credit_history': {
        'A30': 'brak kredytów', 'A31': 'spłacone', 'A32': 'bieżące spłacane',
        'A33': 'opóźnienia', 'A34': 'konto krytyczne',
    },
    'purpose': {
        'A40': 'samochód nowy', 'A41': 'samochód używany', 'A42': 'meble',
        'A43': 'RTV', 'A44': 'AGD', 'A45': 'naprawy',
        'A46': 'edukacja', 'A48': 'przekwalifikowanie',
        'A49': 'biznes', 'A410': 'inne',
    },
    'savings_status': {
        'A61': '<100 DM', 'A62': '100-500 DM', 'A63': '500-1000 DM',
        'A64': '>=1000 DM', 'A65': 'brak/nieznane',
    },
    'employment': {
        'A71': 'bezrobotny', 'A72': '<1 rok', 'A73': '1-4 lata',
        'A74': '4-7 lat', 'A75': '>=7 lat',
    },
    'personal_status': {
        'A91': 'mężczyzna rozwiedziony', 'A92': 'kobieta',
        'A93': 'mężczyzna wolny', 'A94': 'mężczyzna żonaty',
    },
    'other_parties': {
        'A101': 'brak', 'A102': 'współwnioskodawca', 'A103': 'poręczyciel',
    },
    'property_magnitude': {
        'A121': 'nieruchomość', 'A122': 'ubezpieczenie/oszczędności',
        'A123': 'samochód/inne', 'A124': 'brak/nieznane',
    },
    'other_payment_plans': {
        'A141': 'bank', 'A142': 'sklep', 'A143': 'brak',
    },
    'housing': {
        'A151': 'wynajem', 'A152': 'własność', 'A153': 'bezpłatne',
    },
    'job': {
        'A171': 'bezrobotny/niewykwalifikowany',
        'A172': 'niewykwalifikowany rezydent',
        'A173': 'wykwalifikowany', 'A174': 'kierownictwo/samozatrudniony',
    },
    'own_telephone': {
        'A191': 'brak', 'A192': 'tak',
    },
    'foreign_worker': {
        'A201': 'tak', 'A202': 'nie',
    },
}


# =============================================================================
# 1. WCZYTANIE DANYCH
# =============================================================================

print_section("1. WCZYTANIE DANYCH")

german = fetch_ucirepo(id=144)
X = german.data.features
y = german.data.targets

df = X.rename(columns=COLUMN_RENAME).copy()
df['credit_risk'] = y['class'].map({1: 'good', 2: 'bad'})

# Dekoduj wartości atrybutów kategorycznych
for col, mapping in VALUE_LABELS.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(df[col])

print(f"Początkowa liczba rekordów: {len(df)}")
print(f"Kolumny: {list(df.columns)}")
print(df.head(3))
print()
print("Rozkład credit_risk:")
print(df['credit_risk'].value_counts())
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

Q         = ['age', 'personal_status', 'housing', 'foreign_worker']
SENSITIVE = 'credit_risk'

df = add_anon_id(df)
df_anon = df[['anon_ID'] + Q + [SENSITIVE]].copy()

print(f"Quasi-identyfikatory : {Q}")
print(f"Atrybut wrażliwy     : {SENSITIVE}")
print(df_anon.head(3))


# =============================================================================
# 4. GENERALIZACJA
# =============================================================================

print_section("4. GENERALIZACJA")

df_anon['age'] = df_anon['age'].apply(generalize_age)

print("Dane po generalizacji (wiek w dekadach):")
print(df_anon.head(5))
print(f"\nUnikalne wartości 'age'             : {sorted(df_anon['age'].unique())}")
print(f"Unikalne wartości 'personal_status' : {df_anon['personal_status'].unique()}")
print(f"Unikalne wartości 'housing'         : {df_anon['housing'].unique()}")
print(f"Unikalne wartości 'foreign_worker'  : {df_anon['foreign_worker'].unique()}")


# =============================================================================
# 5. KLASY RÓWNOWAŻNOŚCI (przed k-anonimowością)
# =============================================================================

print_section("5. KLASY RÓWNOWAŻNOŚCI — PRZED K-ANONIMIZACJĄ")

all_eq_classes = generate_equivalence_classes(df_anon, Q)
print_eq_class_preview(all_eq_classes, n=5)

sizes = [len(g) for g in all_eq_classes.values()]
print(f"\nStatystyki rozmiarów klas:")
print(f"  min: {min(sizes)}, max: {max(sizes)}, "
      f"średnia: {sum(sizes)/len(sizes):.1f}, mediana: {sorted(sizes)[len(sizes)//2]}")


# =============================================================================
# 6. K-ANONIMOWOŚĆ DLA k = 5
# =============================================================================

print_section("6. K-ANONIMOWOŚĆ  (k = 5)")

K = 5
df_k5 = apply_k_anonymity(df_anon, Q, k=K)

print_k_anonymity_summary(df_anon, df_k5, k=K)
print()
verify_k_anonymity(df_k5, Q, k=K)

print("\nPrzykładowa klasa (wiek 30-39, kobieta, własność, zagraniczny):")
sample = df_k5[
    (df_k5['age'] == '30-39') &
    (df_k5['personal_status'] == 'kobieta') &
    (df_k5['housing'] == 'własność') &
    (df_k5['foreign_worker'] == 'tak')
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

print("\nPrzykładowa klasa (wiek 30-39, kobieta, własność, zagraniczny):")
sample = df_k3[
    (df_k3['age'] == '30-39') &
    (df_k3['personal_status'] == 'kobieta') &
    (df_k3['housing'] == 'własność') &
    (df_k3['foreign_worker'] == 'tak')
]
print(sample.head(10))


# =============================================================================
# 8. L-RÓŻNORODNOŚĆ NA ZBIORZE k = 5
# =============================================================================

print_section("8. L-RÓŻNORODNOŚĆ  (k = 5, l = 2)")

L = 2
df_k5_l2 = apply_l_diversity(df_k5, Q, SENSITIVE, l=L)

print_l_diversity_summary(df_k5, df_k5_l2, l=L)
print()
verify_l_diversity(df_k5_l2, Q, SENSITIVE, l=L)

print()
print_entropy_preview(df_k5_l2, Q, SENSITIVE, n=5)


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
# 10. T-BLISKOŚĆ NA ZBIORZE k = 5, l = 2
# =============================================================================

print_section("10. T-BLISKOŚĆ  (k = 5, l = 2, t = 0.2)")

T = 0.2
df_k5_l2_t = apply_t_closeness(df_k5_l2, Q, SENSITIVE, t=T)

print_t_closeness_summary(df_k5_l2, df_k5_l2_t, t=T)
print()
verify_t_closeness(df_k5_l2_t, Q, SENSITIVE, t=T)

print()
print_t_closeness_preview(df_k5_l2_t, Q, SENSITIVE, n=5)


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


# =============================================================================
# 12. PODSUMOWANIE — WPŁYW ANONIMIZACJI NA UŻYTECZNOŚĆ
# =============================================================================

print_section("12. PODSUMOWANIE — WPŁYW (k, l, t) NA UŻYTECZNOŚĆ")

total = len(df_anon)
print(f"{'Etap':<35} {'Rekordy':>8} {'%':>7}")
print("-" * 52)
print(f"{'Oryginał':<35} {total:>8} {100.0:>6.1f}%")
for label, d in [
    ("k=5", df_k5), ("k=5, l=2", df_k5_l2), ("k=5, l=2, t=0.2", df_k5_l2_t),
    ("k=3", df_k3), ("k=3, l=2", df_k3_l2), ("k=3, l=2, t=0.2", df_k3_l2_t),
]:
    print(f"{label:<35} {len(d):>8} {len(d)/total*100:>6.1f}%")
