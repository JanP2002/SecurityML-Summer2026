# =============================================================================
# shared_functions.py
# Wspólne funkcje dla pipeline'u k-anonimowości i l-różnorodności
# =============================================================================

import uuid
import pandas as pd


# -----------------------------------------------------------------------------
# PSEUDONIMIZACJA
# -----------------------------------------------------------------------------

def add_anon_id(df):
    """Nadaje każdemu rekordowi anonimowy identyfikator (pierwsze 8 znaków UUID)."""
    df = df.copy()
    df['anon_ID'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    return df


# -----------------------------------------------------------------------------
# GENERALIZACJA — WIEK
# -----------------------------------------------------------------------------

def generalize_age(age):
    """Zamienia dokładny wiek na przedział dekadowy, np. 34 -> '30-39'."""
    if pd.isna(age):
        return 'Unknown'
    decade = (int(age) // 10) * 10
    return f"{decade}-{decade + 9}"


# -----------------------------------------------------------------------------
# KLASY RÓWNOWAŻNOŚCI
# -----------------------------------------------------------------------------

def generate_equivalence_classes(df, quasi_identifiers):
    """
    Generuje wszystkie klasy równoważności dla zadanego zbioru danych.

    Zwraca słownik:
      klucz  -> krotka wartości quasi-identyfikatorów
      wartość -> DataFrame z rekordami tej klasy
    """
    classes = {}
    for key, group in df.groupby(quasi_identifiers):
        classes[key] = group
    return classes


# -----------------------------------------------------------------------------
# K-ANONIMOWOŚĆ (supresja)
# -----------------------------------------------------------------------------

def apply_k_anonymity(df, quasi_identifiers, k):
    """
    Stosuje k-anonimowość przez supresję: usuwa rekordy należące do klas
    równoważności liczących mniej niż k elementów.

    Zwraca przefiltrowany DataFrame.
    """
    ec_sizes = df.groupby(quasi_identifiers).size().reset_index(name='ec_size')
    df_merged = pd.merge(df, ec_sizes, on=quasi_identifiers)
    df_result = df_merged[df_merged['ec_size'] >= k].copy()
    df_result = df_result.drop(columns=['ec_size'])
    return df_result


# -----------------------------------------------------------------------------
# WYDRUK WYNIKÓW
# -----------------------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_eq_class_preview(eq_classes, n=3):
    """Drukuje podgląd pierwszych n klas równoważności."""
    print(f"Łącznie klas równoważności: {len(eq_classes)}")
    print(f"Podgląd {n} pierwszych klas:")
    for i, (key, group) in enumerate(eq_classes.items()):
        if i >= n:
            break
        print(f"  -> Klasa {key}: {len(group)} rekordów")


def print_k_anonymity_summary(df_before, df_after, k):
    """Drukuje podsumowanie k-anonimizacji."""
    lost = len(df_before) - len(df_after)
    print(f"Parametr k          : {k}")
    print(f"Rekordy przed       : {len(df_before)}")
    print(f"Rekordy po supresji : {len(df_after)}")
    print(f"Usunięto rekordów   : {lost} ({lost / len(df_before) * 100:.1f}%)")


def verify_k_anonymity(df, quasi_identifiers, k):
    """Weryfikuje czy zbiór spełnia k-anonimowość i drukuje wynik."""
    min_ec = df.groupby(quasi_identifiers).size().min()
    satisfied = min_ec >= k
    print(f"Minimalna klasa równoważności: {min_ec} rekordów")
    if satisfied:
        print(f"✓  Zbiór spełnia {k}-anonimowość")
    else:
        print(f"✗  Zbiór NIE spełnia {k}-anonimowości!")
    return satisfied


# -----------------------------------------------------------------------------
# CZYSZCZENIE DANYCH
# -----------------------------------------------------------------------------

def clean_dataframe(df, int_cols=None, float_cols=None):
    """
    Usuwa braki i duplikaty, opcjonalnie wymusza typy kolumn.

    Parametry:
      int_cols   -- lista kolumn do rzutowania na int   (np. ['age'])
      float_cols -- lista kolumn do rzutowania na float (np. ['bmi'])

    Zwraca oczyszczony DataFrame.
    """
    before = len(df)
    df = df.dropna()
    print(f"Usunięto wierszy z brakującymi wartościami : {before - len(df)}")

    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Usunięto duplikatów                        : {dupes}")
    print(f"Rekordy po czyszczeniu                     : {len(df)}")

    for col in (int_cols or []):
        df[col] = df[col].astype(int)
    for col in (float_cols or []):
        df[col] = df[col].astype(float)

    return df


# -----------------------------------------------------------------------------
# L-RÓŻNORODNOŚĆ
# -----------------------------------------------------------------------------

import numpy as np


def compute_entropy(group, sensitive_attr):
    """Oblicza entropię rozkładu atrybutu wrażliwego w danej klasie."""
    probs = group[sensitive_attr].value_counts(normalize=True)
    return -np.sum(probs * np.log(probs))


def apply_l_diversity(df, quasi_identifiers, sensitive_attr, l):
    """
    Stosuje l-różnorodność przez supresję: usuwa klasy równoważności,
    których entropia rozkładu atrybutu wrażliwego jest mniejsza niż log(l).

    Warunek: H(EC_i) = -sum(P(s|EC_i) * log P(s|EC_i)) >= log(l)

    Zwraca przefiltrowany DataFrame.
    """
    log_l = np.log(l)

    ec_entropy = (
        df.groupby(quasi_identifiers)
        .apply(lambda g: compute_entropy(g, sensitive_attr), include_groups=False)
        .reset_index(name='entropy')
    )

    df_merged = pd.merge(df, ec_entropy, on=quasi_identifiers)
    df_result = df_merged[df_merged['entropy'] >= log_l].drop(columns=['entropy'])
    return df_result


def print_l_diversity_summary(df_before, df_after, l):
    """Drukuje podsumowanie l-dywersyfikacji."""
    lost = len(df_before) - len(df_after)
    print(f"Parametr l             : {l}")
    print(f"Rekordy przed          : {len(df_before)}")
    print(f"Rekordy po supresji    : {len(df_after)}")
    print(f"Usunięto rekordów      : {lost} ({lost / len(df_before) * 100:.1f}%)")


def verify_l_diversity(df, quasi_identifiers, sensitive_attr, l):
    """Weryfikuje czy zbiór spełnia l-różnorodność i drukuje wynik."""
    log_l = np.log(l)
    entropies = df.groupby(quasi_identifiers).apply(
        lambda g: compute_entropy(g, sensitive_attr), include_groups=False
    )
    min_entropy = entropies.min()
    satisfied = min_entropy >= log_l
    print(f"Minimalna entropia w klasie : {min_entropy:.4f}  (log({l}) = {log_l:.4f})")
    if satisfied:
        print(f"✓  Zbiór spełnia {l}-różnorodność")
    else:
        print(f"✗  Zbiór NIE spełnia {l}-różnorodności!")
    return satisfied


def print_entropy_preview(df, quasi_identifiers, sensitive_attr, n=5):
    """Drukuje entropię dla n najmniejszych klas — przydatne do debugowania."""
    entropies = (
        df.groupby(quasi_identifiers)
        .apply(lambda g: pd.Series({
            'entropy': compute_entropy(g, sensitive_attr),
            'size': len(g),
            'unique_values': g[sensitive_attr].nunique(),
        }), include_groups=False)
        .reset_index()
        .sort_values('entropy')
    )
    print(f"Klasy z najniższą entropią (podgląd {n}):")
    print(entropies.head(n).to_string(index=False))