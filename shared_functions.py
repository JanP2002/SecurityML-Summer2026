# =============================================================================
# shared_functions.py
# Wspólne funkcje dla pipeline'u k-anonimowości
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
    for key, group in df.groupby(quasi_identifiers, observed=True):
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
    ec_sizes = df.groupby(quasi_identifiers, observed=True).size().reset_index(name='ec_size')
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
    min_ec = df.groupby(quasi_identifiers, observed=True).size().min()
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
        df.groupby(quasi_identifiers, observed=True)
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
    entropies = df.groupby(quasi_identifiers, observed=True).apply(
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
        df.groupby(quasi_identifiers, observed=True)
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


# -----------------------------------------------------------------------------
# T-BLISKOŚĆ (t-closeness)
# -----------------------------------------------------------------------------


def compute_t_closeness_distance(group_values, global_values, method='auto'):
    """
    Oblicza dystans między rozkładem atrybutu wrażliwego w grupie
    a rozkładem globalnym.

    Metody:
      'auto'        — EMD dla danych numerycznych, variational dla kategorycznych
      'emd'         — Earth Mover's Distance (dla danych uporządkowanych)
      'variational' — dystans wariacyjny (dla danych kategorycznych)
    """
    if method == 'auto':
        if pd.api.types.is_numeric_dtype(global_values):
            method = 'emd'
        else:
            method = 'variational'

    if method == 'variational':
        group_dist = group_values.value_counts(normalize=True)
        global_dist = global_values.value_counts(normalize=True)
        all_vals = set(group_dist.index) | set(global_dist.index)
        return 0.5 * sum(
            abs(group_dist.get(v, 0) - global_dist.get(v, 0))
            for v in all_vals
        )

    if method == 'emd':
        ordered = sorted(global_values.unique())
        m = len(ordered)
        if m <= 1:
            return 0.0
        group_hist = group_values.value_counts(normalize=True)
        global_hist = global_values.value_counts(normalize=True)
        cdf_g, cdf_gl, emd = 0.0, 0.0, 0.0
        for v in ordered:
            cdf_g += group_hist.get(v, 0)
            cdf_gl += global_hist.get(v, 0)
            emd += abs(cdf_g - cdf_gl)
        return emd / (m - 1)

    raise ValueError(f"Nieznana metoda: {method}")


def apply_t_closeness(df, quasi_identifiers, sensitive_attr, t, method='auto'):
    """
    Stosuje t-bliskość przez supresję: usuwa klasy równoważności,
    w których dystans rozkładu atrybutu wrażliwego od rozkładu globalnego
    przekracza próg t.

    Warunek: d(P(S|EC_i), P(S)) <= t

    Zwraca przefiltrowany DataFrame.
    """
    global_values = df[sensitive_attr]

    def _distance(group):
        return compute_t_closeness_distance(
            group[sensitive_attr], global_values, method
        )

    distances = (
        df.groupby(quasi_identifiers, observed=True)
        .apply(_distance, include_groups=False)
        .reset_index(name='t_distance')
    )

    df_merged = pd.merge(df, distances, on=quasi_identifiers)
    df_result = df_merged[df_merged['t_distance'] <= t].drop(columns=['t_distance'])
    return df_result


def print_t_closeness_summary(df_before, df_after, t):
    """Drukuje podsumowanie t-bliskości."""
    lost = len(df_before) - len(df_after)
    print(f"Parametr t             : {t}")
    print(f"Rekordy przed          : {len(df_before)}")
    print(f"Rekordy po supresji    : {len(df_after)}")
    print(f"Usunięto rekordów      : {lost} ({lost / len(df_before) * 100:.1f}%)")


def verify_t_closeness(df, quasi_identifiers, sensitive_attr, t, method='auto'):
    """Weryfikuje czy zbiór spełnia t-bliskość i drukuje wynik."""
    global_values = df[sensitive_attr]

    distances = df.groupby(quasi_identifiers, observed=True).apply(
        lambda g: compute_t_closeness_distance(
            g[sensitive_attr], global_values, method
        ),
        include_groups=False,
    )
    max_dist = distances.max()
    satisfied = max_dist <= t
    print(f"Maksymalny dystans w klasie : {max_dist:.4f}  (próg t = {t})")
    if satisfied:
        print(f"✓  Zbiór spełnia t-bliskość (t = {t})")
    else:
        print(f"✗  Zbiór NIE spełnia t-bliskości (t = {t})!")
    return satisfied


def print_t_closeness_preview(df, quasi_identifiers, sensitive_attr, n=5,
                              method='auto'):
    """Drukuje dystanse t-bliskości dla n klas z największym dystansem."""
    global_values = df[sensitive_attr]
    distances = (
        df.groupby(quasi_identifiers, observed=True)
        .apply(lambda g: pd.Series({
            't_distance': compute_t_closeness_distance(
                g[sensitive_attr], global_values, method
            ),
            'size': len(g),
            'unique_values': g[sensitive_attr].nunique(),
        }), include_groups=False)
        .reset_index()
        .sort_values('t_distance', ascending=False)
    )
    print(f"Klasy z największym dystansem t-bliskości (podgląd {n}):")
    print(distances.head(n).to_string(index=False))


# -----------------------------------------------------------------------------
# ADAPTACYJNE WYSZUKIWANIE OPTYMALNYCH PARAMETRÓW
# -----------------------------------------------------------------------------

import itertools


def generalize_numeric_qcut(series, n_bins):
    """
    Generalizuje kolumnę numeryczną do n_bins kwantylowych przedziałów.
    Używa pd.qcut — każdy przedział zawiera podobną liczbę rekordów.
    Zwraca Series z etykietami 'cat_1', 'cat_2', ... lub None jeśli się nie uda.
    """
    try:
        labels = [f'cat_{i + 1}' for i in range(n_bins)]
        return pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # Za mało unikalnych wartości żeby stworzyć n_bins przedziałów
        return None


def adaptive_search(df, qi_categorical, qi_numerical, sensitive_attr,
                    k_values, l_values, n_bins_options, t_values=None,
                    top_n=15):
    """
    Przeszukuje przestrzeń parametrów (k, l, podziały kolumn numerycznych)
    szukając kombinacji która maksymalizuje liczbę zachowanych rekordów
    przy spełnieniu k-anonimowości i l-różnorodności.

    Parametry:
      df             -- DataFrame z kolumną anon_ID i atrybutem wrażliwym
      qi_categorical -- lista kolumn kategorycznych QI (bez generalizacji)
      qi_numerical   -- słownik {nazwa_col: Series} dla kolumn numerycznych
      sensitive_attr -- nazwa kolumny z atrybutem wrażliwym
      k_values       -- lista wartości k do sprawdzenia, np. [2, 3, 5, 10]
      l_values       -- lista wartości l do sprawdzenia, np. [2, 3, 4]
      n_bins_options -- lista liczb kubełków do próbowania, np. [2, 3, 4, 5]
      top_n          -- ile najlepszych wyników wydrukować

    Zwraca DataFrame z wynikami posortowanymi od najlepszego.
    """
    total = len(df)
    num_cols = list(qi_numerical.keys())
    results = []

    # Wszystkie kombinacje liczby kubełków dla kolumn numerycznych
    bin_combinations = list(itertools.product(n_bins_options, repeat=len(num_cols)))
    total_combinations = len(k_values) * len(l_values) * len(bin_combinations)
    if t_values is not None:
        total_combinations *= len(t_values)

    print(f"Przestrzeń poszukiwań:")
    print(f"  k: {k_values}")
    print(f"  l: {l_values}")
    if t_values is not None:
        print(f"  t: {t_values}")
    print(f"  kubełki: {n_bins_options} dla każdej z {len(num_cols)} kolumn numerycznych")
    print(f"  łącznie kombinacji: {total_combinations}")
    print(f"\nPrzeszukiwanie...", end='', flush=True)

    checked = 0
    for k in k_values:
        for l in l_values:
            # l > k jest teoretycznie bez sensu — przy l wartościach wrażliwych
            # potrzebujesz co najmniej l rekordów, więc k >= l jest naturalne
            if l > k:
                continue

            for bin_combo in bin_combinations:
                checked += 1
                if checked % 20 == 0:
                    print('.', end='', flush=True)

                # Generalizuj kolumny numeryczne
                generalized_cols = {}
                valid = True
                for col, n_bins in zip(num_cols, bin_combo):
                    gen = generalize_numeric_qcut(qi_numerical[col], n_bins)
                    if gen is None:
                        valid = False
                        break
                    generalized_cols[col] = gen

                if not valid:
                    continue

                # Zbuduj roboczy DataFrame z wygeneralizowanymi kolumnami
                cat_col_names = [f'{col}_g{n}' for col, n in zip(num_cols, bin_combo)]
                df_work = df[['anon_ID'] + qi_categorical + [sensitive_attr]].copy()
                for col, cat_name, gen_series in zip(num_cols, cat_col_names, generalized_cols.values()):
                    df_work[cat_name] = gen_series.values

                Q = qi_categorical + cat_col_names

                # k-anonimowość
                df_k = apply_k_anonymity(df_work, Q, k)
                if len(df_k) == 0:
                    continue

                # l-różnorodność
                df_l = apply_l_diversity(df_k, Q, sensitive_attr, l)

                if t_values is not None:
                    if len(df_l) == 0:
                        continue
                    for t_val in t_values:
                        df_t = apply_t_closeness(df_l, Q, sensitive_attr, t_val)
                        retained = len(df_t)
                        row = {
                            'k': k, 'l': l, 't': t_val,
                            'retained': retained,
                            'pct_retained': round(retained / total * 100, 1),
                            'removed': total - retained,
                        }
                        for col, n in zip(num_cols, bin_combo):
                            row[f'{col}_bins'] = n
                        results.append(row)
                else:
                    retained = len(df_l)
                    row = {
                        'k': k,
                        'l': l,
                        'retained': retained,
                        'pct_retained': round(retained / total * 100, 1),
                        'removed': total - retained,
                    }
                    for col, n in zip(num_cols, bin_combo):
                        row[f'{col}_bins'] = n
                    results.append(row)

    print(f'\nSprawdzono {checked} kombinacji.\n')

    if not results:
        print("Brak wyników — żadna kombinacja nie spełniła warunków.")
        return pd.DataFrame()

    sort_cols = ['retained', 'k', 'l']
    sort_asc = [False, False, False]
    if t_values is not None:
        sort_cols.append('t')
        sort_asc.append(True)

    df_results = (
        pd.DataFrame(results)
        .sort_values(sort_cols, ascending=sort_asc)
        .reset_index(drop=True)
    )

    return df_results


def print_adaptive_results(df_results, top_n=15):
    """Drukuje wyniki adaptacyjnego wyszukiwania."""
    if df_results.empty:
        return

    print(f"Top {min(top_n, len(df_results))} kombinacji (posortowane wg zachowanych rekordów):")
    print(df_results.head(top_n).to_string(index=False))

    best = df_results.iloc[0]
    print(f"\n{'=' * 50}")
    print(f"OPTIMUM:")
    print(f"  k               = {best['k']}")
    print(f"  l               = {best['l']}")
    if 't' in df_results.columns:
        print(f"  t               = {best['t']}")
    for col in df_results.columns:
        if col.endswith('_bins'):
            print(f"  {col:<15} = {int(best[col])}")
    print(f"  Zachowane rekordy: {int(best['retained'])} / {int(best['retained'] + best['removed'])} ({best['pct_retained']}%)")
    print(f"{'=' * 50}")
