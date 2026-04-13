# =============================================================================
# shared_functions.py
# Wspólne funkcje dla pipeline'u k-anonimowości, l-różnorodności,
# t-bliskości i ataku skośności
# =============================================================================

import uuid
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -----------------------------------------------------------------------------
# PSEUDONIMIZACJA
# -----------------------------------------------------------------------------

def add_anon_id(df):
    df = df.copy()
    df['anon_ID'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    return df


# -----------------------------------------------------------------------------
# GENERALIZACJA — WIEK
# -----------------------------------------------------------------------------

def generalize_age(age):
    if pd.isna(age):
        return 'Unknown'
    decade = (int(age) // 10) * 10
    return f"{decade}-{decade + 9}"


# -----------------------------------------------------------------------------
# KLASY RÓWNOWAŻNOŚCI
# -----------------------------------------------------------------------------

def generate_equivalence_classes(df, quasi_identifiers):
    classes = {}
    for key, group in df.groupby(quasi_identifiers, observed=True):
        classes[key] = group
    return classes


# -----------------------------------------------------------------------------
# K-ANONIMOWOŚĆ (supresja)
# -----------------------------------------------------------------------------

def apply_k_anonymity(df, quasi_identifiers, k):
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
    print(f"Łącznie klas równoważności: {len(eq_classes)}")
    print(f"Podgląd {n} pierwszych klas:")
    for i, (key, group) in enumerate(eq_classes.items()):
        if i >= n:
            break
        print(f"  -> Klasa {key}: {len(group)} rekordów")


def print_k_anonymity_summary(df_before, df_after, k):
    lost = len(df_before) - len(df_after)
    print(f"Parametr k          : {k}")
    print(f"Rekordy przed       : {len(df_before)}")
    print(f"Rekordy po supresji : {len(df_after)}")
    print(f"Usunięto rekordów   : {lost} ({lost / len(df_before) * 100:.1f}%)")


def verify_k_anonymity(df, quasi_identifiers, k):
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

def compute_entropy(group, sensitive_attr):
    probs = group[sensitive_attr].value_counts(normalize=True)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def apply_l_diversity(df, quasi_identifiers, sensitive_attr, l):
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
    lost = len(df_before) - len(df_after)
    print(f"Parametr l             : {l}")
    print(f"Rekordy przed          : {len(df_before)}")
    print(f"Rekordy po supresji    : {len(df_after)}")
    print(f"Usunięto rekordów      : {lost} ({lost / len(df_before) * 100:.1f}%)")


def verify_l_diversity(df, quasi_identifiers, sensitive_attr, l):
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
    if method == 'auto':
        method = 'emd' if pd.api.types.is_numeric_dtype(global_values) else 'variational'

    if method == 'variational':
        group_dist  = group_values.value_counts(normalize=True)
        global_dist = global_values.value_counts(normalize=True)
        all_vals = set(group_dist.index) | set(global_dist.index)
        return 0.5 * sum(abs(group_dist.get(v, 0) - global_dist.get(v, 0)) for v in all_vals)

    if method == 'emd':
        ordered = sorted(global_values.unique())
        m = len(ordered)
        if m <= 1:
            return 0.0
        group_hist  = group_values.value_counts(normalize=True)
        global_hist = global_values.value_counts(normalize=True)
        cdf_g = cdf_gl = emd = 0.0
        for v in ordered:
            cdf_g  += group_hist.get(v, 0)
            cdf_gl += global_hist.get(v, 0)
            emd    += abs(cdf_g - cdf_gl)
        return emd / (m - 1)

    raise ValueError(f"Nieznana metoda: {method}")


def apply_t_closeness(df, quasi_identifiers, sensitive_attr, t, method='auto'):
    global_values = df[sensitive_attr].copy()

    def _distance(group):
        return compute_t_closeness_distance(group[sensitive_attr], global_values, method)

    distances = (
        df.groupby(quasi_identifiers, observed=True)
        .apply(_distance, include_groups=False)
        .reset_index(name='t_distance')
    )
    df_merged = pd.merge(df, distances, on=quasi_identifiers)
    df_result = df_merged[df_merged['t_distance'] <= t].drop(columns=['t_distance'])
    return df_result, global_values


def print_t_closeness_summary(df_before, df_after, t):
    lost = len(df_before) - len(df_after)
    print(f"Parametr t             : {t}")
    print(f"Rekordy przed          : {len(df_before)}")
    print(f"Rekordy po supresji    : {len(df_after)}")
    print(f"Usunięto rekordów      : {lost} ({lost / len(df_before) * 100:.1f}%)")


def verify_t_closeness(df, quasi_identifiers, sensitive_attr, t,
                       global_values=None, method='auto'):
    if global_values is None:
        global_values = df[sensitive_attr]
    distances = df.groupby(quasi_identifiers, observed=True).apply(
        lambda g: compute_t_closeness_distance(g[sensitive_attr], global_values, method),
        include_groups=False,
    )
    max_dist  = distances.max()
    satisfied = max_dist <= t
    print(f"Maksymalny dystans w klasie : {max_dist:.4f}  (próg t = {t})")
    if satisfied:
        print(f"✓  Zbiór spełnia t-bliskość (t = {t})")
    else:
        print(f"✗  Zbiór NIE spełnia t-bliskości (t = {t})!")
    return satisfied


def print_t_closeness_preview(df, quasi_identifiers, sensitive_attr, n=5,
                              global_values=None, method='auto'):
    if global_values is None:
        global_values = df[sensitive_attr]
    distances = (
        df.groupby(quasi_identifiers, observed=True)
        .apply(lambda g: pd.Series({
            't_distance':    compute_t_closeness_distance(g[sensitive_attr], global_values, method),
            'size':          len(g),
            'unique_values': g[sensitive_attr].nunique(),
        }), include_groups=False)
        .reset_index()
        .sort_values('t_distance', ascending=False)
    )
    print(f"Klasy z największym dystansem t-bliskości (podgląd {n}):")
    print(distances.head(n).to_string(index=False))


# -----------------------------------------------------------------------------
# ATAK SKOŚNOŚCI (Skewness Attack)
# -----------------------------------------------------------------------------

def run_skewness_attack(df_anon, Q, sensitive_attr, tau,
                        dataset_name='', stage_name='', plot=True):
    """
    Przeprowadza atak skośności na zanonimizowanym zbiorze danych.

    Dla każdej klasy EC_i porównuje lokalny rozkład P_i(s) atrybutu wrażliwego
    z rozkładem globalnym P_G(s). Jeśli max|P_i(s) - P_G(s)| > tau — atak udany.

    Parametry:
      df_anon        -- zanonimizowany DataFrame
      Q              -- lista quasi-identyfikatorów
      sensitive_attr -- nazwa kolumny z atrybutem wrażliwym
      tau            -- próg pewności atakującego (np. 0.10)
      dataset_name   -- nazwa datasetu (do tytułu wykresu)
      stage_name     -- etap anonimizacji (do tytułu wykresu), np. 'k=5'
      plot           -- czy rysować wykres

    Zwraca słownik z wynikami (do użycia w raporcie).
    """
    P_G = df_anon[sensitive_attr].value_counts(normalize=True).sort_index()
    n_classes = df_anon.groupby(Q, observed=True).ngroups
    successful_attacks = []

    for ec_name, group in df_anon.groupby(Q, observed=True):
        P_i = group[sensitive_attr].value_counts(normalize=True)

        max_diff     = 0.0
        skewed_value = None

        for val in P_G.index:
            diff = abs(P_i.get(val, 0) - P_G.get(val, 0))
            if diff > max_diff:
                max_diff     = diff
                skewed_value = val

        if max_diff > tau:
            successful_attacks.append({
                'ec_name':      ec_name,
                'size':         len(group),
                'P_i':          P_i,
                'max_diff':     max_diff,
                'skewed_value': skewed_value,
            })

    n_vulnerable = len(successful_attacks)
    pct_vulnerable = n_vulnerable / n_classes * 100 if n_classes > 0 else 0.0

    # --- Drukowanie wyników ---
    title = f"{dataset_name} | {stage_name}"
    print(f"\n{'─' * 60}")
    print(f"  ATAK SKOŚNOŚCI — {title}")
    print(f"{'─' * 60}")
    print(f"Próg tau              : {tau}")
    print(f"Klas równoważności    : {n_classes}")
    print(f"Klas podatnych        : {n_vulnerable} ({pct_vulnerable:.1f}%)")
    print(f"\nRozkład globalny P_G(s):")
    for val, prob in P_G.items():
        print(f"  {str(val):<20} {prob:.4f}")

    victim = None
    if successful_attacks:
        successful_attacks.sort(key=lambda x: x['max_diff'], reverse=True)
        victim = successful_attacks[0]

        print(f"\nNajbardziej podatna klasa:")
        for q, v in zip(Q, victim['ec_name'] if isinstance(victim['ec_name'], tuple) else (victim['ec_name'],)):
            print(f"  {q}: {v}")
        print(f"  Rozmiar grupy : {victim['size']} rekordów")
        print(f"\nRozkład lokalny P_i(s) w tej klasie:")
        for val, prob in victim['P_i'].sort_index().items():
            marker = " ← max odchylenie" if val == victim['skewed_value'] else ""
            print(f"  {str(val):<20} {prob:.4f}{marker}")
        print(f"\nMaksymalne odchylenie : {victim['max_diff']:.4f} > tau={tau}  →  ATAK UDANY")
    else:
        print(f"\nŻadna klasa nie przekroczyła progu tau={tau}.")
        print("Zbiór jest odporny na atak skośności przy tych parametrach.")

    # --- Wykres ---
    if plot and victim is not None:
        labels      = [str(v) for v in P_G.index]
        global_vals = [float(P_G.get(v, 0)) for v in P_G.index]
        local_vals  = [float(victim['P_i'].get(v, 0)) for v in P_G.index]

        x     = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Atak skośności — {title}", fontsize=13, fontweight='bold')

        # Lewy panel: porównanie rozkładów
        ax = axes[0]
        bars1 = ax.bar(x - width / 2, global_vals, width, label='Globalny $P_G(s)$',  color='#4C72B0', alpha=0.85)
        bars2 = ax.bar(x + width / 2, local_vals,  width, label='Lokalny $P_i(s)$', color='#C44E52', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Prawdopodobieństwo')
        ax.set_title('Porównanie rozkładów')
        ax.legend()
        ax.set_ylim(0, min(1.0, max(global_vals + local_vals) * 1.25))
        for bar in bars1:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=8)

        # Prawy panel: odchylenia |P_i - P_G|
        ax2   = axes[1]
        diffs = [abs(lv - gv) for lv, gv in zip(local_vals, global_vals)]
        colors = ['#C44E52' if d > tau else '#4C72B0' for d in diffs]
        ax2.bar(x, diffs, color=colors, alpha=0.85)
        ax2.axhline(y=tau, color='black', linestyle='--', linewidth=1.2, label=f'tau = {tau}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax2.set_ylabel('|P_i(s) - P_G(s)|')
        ax2.set_title('Odchylenie od rozkładu globalnego')
        ax2.legend()
        for i, (bar, d) in enumerate(zip(ax2.patches, diffs)):
            ax2.text(bar.get_x() + bar.get_width() / 2, d + 0.005, f'{d:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        filename = f"attack_{dataset_name.lower().replace(' ', '_')}_{stage_name.replace('=', '').replace(',', '').replace(' ', '_')}.png"
        plt.savefig("plots/" + filename, dpi=150, bbox_inches='tight')
        print(f"\nWykres zapisano: {filename}")
        plt.show()

    return {
        'dataset':       dataset_name,
        'stage':         stage_name,
        'tau':           tau,
        'n_classes':     n_classes,
        'n_vulnerable':  n_vulnerable,
        'pct_vulnerable': pct_vulnerable,
        'P_G':           P_G,
        'victim':        victim,
    }


def print_attack_summary_table(results):
    """
    Drukuje zbiorczą tabelę wyników ataków dla raportu.
    Przyjmuje listę słowników zwróconych przez run_skewness_attack.
    """
    print("\n" + "=" * 70)
    print("  ZBIORCZE WYNIKI ATAKÓW SKOŚNOŚCI")
    print("=" * 70)
    header = f"{'Dataset':<22} {'Etap':<20} {'tau':>5} {'Klas':>6} {'Podatnych':>10} {'%':>7} {'Max odch.':>10}"
    print(header)
    print("─" * 70)
    for r in results:
        max_diff = r['victim']['max_diff'] if r['victim'] else 0.0
        print(
            f"{r['dataset']:<22} {r['stage']:<20} {r['tau']:>5.2f} "
            f"{r['n_classes']:>6} {r['n_vulnerable']:>10} {r['pct_vulnerable']:>6.1f}% "
            f"{max_diff:>10.4f}"
        )
    print("─" * 70)


# -----------------------------------------------------------------------------
# ADAPTACYJNE WYSZUKIWANIE
# -----------------------------------------------------------------------------

def generalize_numeric_qcut(series, n_bins):
    try:
        labels = [f'cat_{i + 1}' for i in range(n_bins)]
        return pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        return None


def adaptive_search(df, qi_categorical, qi_numerical, sensitive_attr,
                    k_values, l_values, n_bins_options,
                    t_values=None, use_t_closeness=True, top_n=15):
    """
    use_t_closeness=False wylacza t-bliskosc nawet jesli t_values jest podane.
    Rownowazne z t_values=None, ale bardziej czytelne.
    """
    if not use_t_closeness:
        t_values = None
    total    = len(df)
    num_cols = list(qi_numerical.keys())
    results  = []

    bin_combinations    = list(itertools.product(n_bins_options, repeat=len(num_cols)))
    total_combinations  = len(k_values) * len(l_values) * len(bin_combinations)
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
            if l > k:
                continue
            for bin_combo in bin_combinations:
                checked += 1
                if checked % 20 == 0:
                    print('.', end='', flush=True)

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

                cat_col_names = [f'{col}_g{n}' for col, n in zip(num_cols, bin_combo)]
                df_work = df[['anon_ID'] + qi_categorical + [sensitive_attr]].copy()
                for col, cat_name, gen_series in zip(num_cols, cat_col_names, generalized_cols.values()):
                    df_work[cat_name] = gen_series.values

                Q    = qi_categorical + cat_col_names
                df_k = apply_k_anonymity(df_work, Q, k)
                if len(df_k) == 0:
                    continue
                df_l = apply_l_diversity(df_k, Q, sensitive_attr, l)

                if t_values is not None:
                    if len(df_l) == 0:
                        continue
                    for t_val in t_values:
                        df_t, _ = apply_t_closeness(df_l, Q, sensitive_attr, t_val)
                        retained = len(df_t)
                        row = {'k': k, 'l': l, 't': t_val,
                               'retained': retained,
                               'pct_retained': round(retained / total * 100, 1),
                               'removed': total - retained}
                        for col, n in zip(num_cols, bin_combo):
                            row[f'{col}_bins'] = n
                        results.append(row)
                else:
                    retained = len(df_l)
                    row = {'k': k, 'l': l,
                           'retained': retained,
                           'pct_retained': round(retained / total * 100, 1),
                           'removed': total - retained}
                    for col, n in zip(num_cols, bin_combo):
                        row[f'{col}_bins'] = n
                    results.append(row)

    print(f'\nSprawdzono {checked} kombinacji.\n')

    if not results:
        print("Brak wyników — żadna kombinacja nie spełniła warunków.")
        return pd.DataFrame()

    sort_cols = ['retained', 'k', 'l']
    sort_asc  = [False, False, False]
    if t_values is not None:
        sort_cols.append('t')
        sort_asc.append(True)

    return (pd.DataFrame(results)
            .sort_values(sort_cols, ascending=sort_asc)
            .reset_index(drop=True))


def print_adaptive_results(df_results, top_n=15):
    if df_results.empty:
        return
    print(f"Top {min(top_n, len(df_results))} kombinacji (posortowane wg zachowanych rekordów):")
    print(df_results.head(top_n).to_string(index=False))
    best = df_results.iloc[0]
    print(f"\n{'=' * 50}")
    print(f"OPTIMUM:")
    print(f"  k = {best['k']}")
    print(f"  l = {best['l']}")
    if 't' in df_results.columns:
        print(f"  t = {best['t']}")
    for col in df_results.columns:
        if col.endswith('_bins'):
            print(f"  {col:<15} = {int(best[col])}")
    print(f"  Zachowane rekordy: {int(best['retained'])} / "
          f"{int(best['retained'] + best['removed'])} ({best['pct_retained']}%)")
    print(f"{'=' * 50}")
