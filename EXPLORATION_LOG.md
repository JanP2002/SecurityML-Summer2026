# EXPLORATION_LOG — graphrep

Dziennik eksploracji (Cel A: klasteryzacja, Cel B: prywatność, Cel C: CIFAR/grafy z obrazu).
Konwencja: każdy wpis = data, hipoteza, zmiana, komenda, wynik (km_ARI / km_NMI / probe_acc /
recon_r2), wniosek. Raportujemy **parami (użyteczność, prywatność)**. Selekcja po kryteriach
bezetykietowych (silhouette / CV sondy); ARI/NMI to **wynik końcowy**, nie cel strojenia.

---

## Leaderboardy (najlepsze konfiguracje per zbiór)

↑ lepiej: ARI/NMI/probe_acc/silhouette · ↓ lepiej: recon_leak (prywatność, odporny atak).
Liczby zweryfikowane na pełnych danych — szczegóły w sekcji **AUDYT FINALNY** (koniec pliku).

### ENZYMES (6 klas, pełne 600 grafów) — sufit trudny; recon = ODPORNY ATAK
| # | metoda / konfig | km_ARI | km_NMI | silhouette | probe_acc | recon_leak↓ |
|---|---|---|---|---|---|---|
| 1 | wl + KMeans (honest) | **0.046** | 0.086 | 0.141 | 0.403 | 0.077 |
| ~ | wl + UMAP→KMeans (oracle) | 0.052 | 0.095 | 0.451 | 0.403 | — |
| 2 | g2v (baseline) | 0.034 | 0.060 | 0.097 | 0.332 | 0.099 |
| 3 | topo (baseline) | 0.030 | 0.059 | 0.263 | 0.302 | 0.002 |

> UMAP→KMeans daje 0.052 (oracle nn30,md0.1), ale dobrej konfiguracji UMAP nie da się wybrać
> label-free (silhouette woli nn10 → ARI 0.021). Honest sufit ≈ 0.046 (zbiór wewnętrznie
> trudny — nie inwestujemy więcej).

### PROTEINS_full (2 klasy) — PEŁNE 1113 grafów; recon = ODPORNY ATAK (RidgeCV+MLP, max)
| # | metoda / konfig | km_ARI | km_NMI | silhouette | probe_acc | recon_leak↓ |
|---|---|---|---|---|---|---|
| 1 | **fgsd** (histogram odl. biharm.) | **0.135** | 0.086 | 0.277 | 0.691 | **0.042** |
| 2 | rand_ens (topo, M=10, dropedge0.5) | 0.089 | 0.072 | 0.291 | 0.724 | 0.082 |
| 3 | topo (baseline) | 0.080 | 0.063 | 0.318 | 0.741 | 0.099 |
| 3 | wltopo (wl+topo) | 0.080 | 0.063 | 0.247 | 0.750 | 0.160 |
| — | attr (leaky) | 0.037 | 0.026 | 0.194 | 0.748 | 0.764 |
| — | wl (baseline) | 0.040 | 0.026 | 0.274 | 0.725 | 0.103 |

> **fgsd = ścisły zwycięzca:** najlepsze ARI (0.135 ≈ 1.7× topo) **i** najniższy wyciek
> (0.042) pod odpornym atakiem. `recon_leak`∈[0,1] = clip(max(RidgeCV,MLP) R², 0, 1).
> ⚠️ Korekta: dawne ujemne recon (wltopo -0.164) były **artefaktem rozbieżnego MLP**; pod
> uczciwym atakiem liniowym wltopo (0.160) przecieka nieco więcej niż topo (0.099) — „Pareto
> wltopo" wycofane. `rand_ens` (ensembling losowych grafów) bije swój deskryptor bazowy topo
> (0.089 vs 0.080) przez denoising degree-stats, ale **nie** lidera fgsd; silhouette woli topo.

### CIFAR-10 (3 klasy: 0/1/8 = samolot/auto/statek, SLIC) — 1500 grafów; recon = ODPORNY ATAK
| # | metoda / konfig | km_ARI | km_NMI | silhouette | probe_acc | recon_leak↓ |
|---|---|---|---|---|---|---|
| 1 | **wl + GMM-full** | **0.085** | 0.080 | 0.193 | 0.573 | 0.712 |
| 2 | wl + KMeans | 0.082 | 0.094 | 0.223 | 0.573 | 0.712 |
| 2 | g2v + KMeans | 0.082 | 0.081 | 0.055 | 0.587 | 0.651 |
| — | attr (leaky, odniesienie) | **0.124** | 0.116 | 0.258 | 0.633 | 0.885 |
| — | rgb (baseline wyglądu) | 0.071 | 0.073 | 0.561 | 0.520 | 0.441 |
| — | topo | 0.045 | 0.050 | 0.254 | 0.491 | 0.323 |
| — | hog (baseline wyglądu) | 0.008 | 0.009 | 0.050 | 0.718 | 0.264 |
| — | _oracle LDA(wl)→KMeans_ | _0.220\*_ | 0.194 | — | — | — |

> ⚠️ **label-rich nie skaluje** (boost wl 0.078→0.108 przy per-class 200 znika przy 1500
> grafach: wl 0.082 bez/z label-rich) — był efektem małej próbki. Robustny strukturalny lider:
> **wl ≈ 0.082±0.008** (GMM-full 0.085), bijący wygląd (hog 0.008, rgb 0.071). `attr` (0.124) to
> nienadzorowane maksimum, lecz najbardziej przeciekowe (0.885). wl/g2v niosą kolor (seed) →
> recon 0.712/0.651. \*oracle LDA: gdy oś klasy znana, ARI skacze 0.082→0.220 (×2.6).
> Liczby zweryfikowane na pełnych danych — patrz sekcja AUDYT FINALNY.

---

## Dziennik prób

> ⚠️ Wpisy Krok 0 / H* / B1–B4 / C1–C5 poniżej to **eksploracja na podpróbkach (per-class 200)
> ze STARYM atakiem MLP** (recon_r2 z artefaktami). Wiążące liczby — w leaderboardach u góry
> i sekcji AUDYT FINALNY (`results_FINAL_*`). Te wpisy zostawione jako ślad rozumowania.

### 2026-06-23 · Krok 0 — baseline ENZYMES (pełne 600 grafów)
- **Komenda:** `python run_experiments.py --source tu --dataset ENZYMES --methods topo spec wl g2v netlsd graphlet n2v attr combo gnat --out-dir results_enz_base`
- **Wynik (best-w per metoda):**

  | method | w | ARI | NMI | silh | probe_acc | recon_r2 |
  |---|---|---|---|---|---|---|
  | topo | 1.0 | 0.030 | 0.059 | 0.263 | 0.302 | -0.032 |
  | spec | 1.0 | 0.017 | 0.034 | 0.290 | 0.270 | -0.031 |
  | wl | 1.0 | 0.046 | 0.086 | 0.141 | 0.403 | -0.187 |
  | g2v | 1.0 | 0.034 | 0.060 | 0.097 | 0.332 | -0.218 |
  | netlsd | 1.0 | 0.015 | 0.040 | 0.363 | 0.253 | -0.021 |
  | graphlet | 1.0 | 0.025 | 0.048 | **0.435** | 0.242 | 0.042 |
  | n2v | 1.0 | 0.021 | 0.044 | 0.123 | 0.220 | -0.464 |
  | attr | 0.0 | 0.032 | 0.056 | 0.170 | **0.490** | **0.791** |
  | combo | 0.0 | 0.032 | 0.056 | 0.170 | 0.490 | 0.791 |
  | gnat | 0.0 | 0.032 | 0.056 | 0.170 | 0.490 | 0.791 |

- **Wniosek:** ENZYMES jest trudny — wszystkie ARI < 0.05. Najlepsza separowalność
  bez wycieku: `wl` (ARI 0.046, acc 0.403, recon -0.187 = prywatne). `attr` ma najwyższe
  acc (0.490) ale recon_r2 0.791 → wycieka treść (zgodnie z walidacją ramy). `combo`/`gnat`
  zapadają się do w=0 (czysty attr), bo sweep wybiera w wg probe_acc → trafia w przeciekający
  strumień atrybutowy. `graphlet` ma najwyższe silhouette (0.435) ale niskie ARI — klastry
  „czyste" geometrycznie, lecz nie zgodne z klasami. NetLSD/graphlet **nie** biją wl/topo
  na ARI w tym baseline (do dalszego testu z innym klastrowaniem/fuzją).

### 2026-06-23 · Krok 0 — baseline PROTEINS_full (per-class 200 → 400 grafów)
- **Komenda:** `python run_experiments.py --source tu --dataset PROTEINS_full --per-class 200 --methods topo spec wl g2v netlsd graphlet attr combo gnat --out-dir results_prot_base`
- **Wynik (best-w per metoda):**

  | method | w | ARI | NMI | silh | probe_acc | recon_r2 |
  |---|---|---|---|---|---|---|
  | topo | 1.0 | **0.071** | 0.053 | 0.306 | 0.680 | 0.042 |
  | spec | 1.0 | 0.022 | 0.017 | 0.397 | 0.670 | 0.035 |
  | wl | 1.0 | 0.055 | 0.042 | 0.278 | 0.635 | -0.372 |
  | g2v | 1.0 | 0.012 | 0.011 | 0.100 | 0.603 | -0.218 |
  | netlsd | 1.0 | 0.044 | 0.034 | **0.507** | 0.657 | 0.036 |
  | graphlet | 1.0 | 0.016 | 0.013 | 0.479 | 0.652 | 0.038 |
  | attr | 0.0 | 0.028 | 0.022 | 0.186 | **0.722** | **0.827** |
  | combo/gnat | 0.0 | 0.028 | 0.022 | 0.186 | 0.722 | 0.827 |

- **Wniosek:** PROTEINS czytelniejszy niż ENZYMES. Najlepsze ARI: `topo` (0.071, acc 0.680,
  recon 0.042). `wl` najbardziej prywatne (recon -0.372) przy ARI 0.055. `attr` znów:
  najwyższe acc (0.722) ale recon 0.827 = wyciek. NetLSD/graphlet wysokie silhouette
  (0.51/0.48) lecz niskie ARI — klastry nie zgodne z binarnym podziałem. combo/gnat znów
  zapadają się do w=0 (czysty attr).


### 2026-06-23 · Krok 0 — baseline CIFAR-10 (3 klasy 0/1/8, SLIC, per-class 200 → 600 grafów)
- **Komenda:** `python run_experiments.py --source cifar --graph-type slic --classes 0 1 8 --per-class 200 --methods topo spec wl g2v netlsd graphlet attr combo gnat rgb hog hyb --out-dir results_cifar_base`
- **Wynik (best-w per metoda):**

  | method | w | ARI | NMI | silh | probe_acc | recon_r2 |
  |---|---|---|---|---|---|---|
  | topo | 1.0 | 0.056 | 0.069 | 0.267 | 0.525 | 0.286 |
  | spec | 1.0 | 0.059 | 0.056 | 0.280 | 0.452 | 0.087 |
  | wl | 1.0 | **0.078** | 0.066 | 0.194 | 0.627 | 0.508 |
  | g2v | 1.0 | 0.062 | 0.058 | 0.058 | 0.582 | 0.477 |
  | netlsd | 1.0 | 0.024 | 0.022 | 0.522 | 0.445 | 0.053 |
  | graphlet | 1.0 | 0.037 | 0.036 | 0.381 | 0.420 | 0.100 |
  | attr | 0.0 | **0.118** | 0.117 | 0.270 | 0.652 | **0.877** |
  | combo/gnat | 0.0 | 0.118 | 0.117 | 0.270 | 0.652 | 0.877 |
  | rgb | 1.0 | 0.053 | 0.049 | 0.549 | 0.505 | 0.421 |
  | hog | 1.0 | 0.019 | 0.023 | 0.052 | 0.628 | -0.304 |
  | hyb (HOG⊕g2v) | 0.25 | 0.020 | 0.023 | 0.048 | **0.665** | 0.209 |

- **Wniosek:** Bez etykiet wygrywa `attr` (ARI 0.118 — to pooling koloru węzłów), ale recon
  0.877 = silny wyciek. Wśród czystej struktury najlepszy `wl` (ARI 0.078) > g2v 0.062 >
  topo 0.056. Metody grafowe (wl 0.078) biją baseline'y wyglądu (rgb 0.053, hog 0.019) na
  ARI — zgodnie z odniesieniem z CLAUDE.md. `hyb` daje najwyższe acc (0.665) — struktura
  dokłada do HOG (acc 0.628 → 0.665). recon hog ujemny (-0.304) → HOG nieodwracalny do
  koloru węzłów (ale to inna treść).

### 2026-06-23 · H1 — algorytm klastrowania (spectral / agglo) na PROTEINS
- **Komendy:** `--cluster-algo spectral` i `--cluster-algo agglo`, metody topo/wl/g2v/netlsd/graphlet.
- **Wynik:** spectral mocno **psuje** ARI (wl 0.004, netlsd 0.008, vs kmeans 0.055/0.044).
  agglo: netlsd ↑ 0.055 (vs 0.044), ale topo ↓ 0.028 (vs 0.071) — mieszane.
- **Wniosek:** Domyślny **KMeans pozostaje najlepszy** na PROTEINS. Spektralne z affinity
  nearest_neighbors nie pasuje do tych gęstych embeddingów. Nie zmieniam domyślnego algorytmu.

### 2026-06-23 · H2 — fuzja całografowych deskryptorów strukturalnych (wltopo / sfuse / sfuse2)
- **Zmiana kodu:** dodane metody `wltopo`=[wl,topo], `sfuse`=[topo,wl,netlsd,graphlet],
  `sfuse2`=[topo,netlsd,graphlet] (czysta struktura, w=1, bez sweepu).
- **Komendy:** `--methods wltopo sfuse sfuse2` na PROTEINS (per-class 200) i ENZYMES (pełne).
- **Wynik:**

  | zbiór | method | ARI | NMI | silh | acc | recon |
  |---|---|---|---|---|---|---|
  | PROTEINS | wltopo | 0.071 | 0.053 | 0.235 | 0.613 | **-0.402** |
  | PROTEINS | sfuse | 0.058 | 0.044 | 0.341 | 0.632 | -0.420 |
  | PROTEINS | sfuse2 | 0.058 | 0.044 | 0.379 | 0.695 | 0.003 |
  | ENZYMES | wltopo | 0.030 | 0.059 | 0.185 | 0.418 | -0.173 |
  | ENZYMES | sfuse | 0.033 | 0.068 | 0.240 | 0.405 | -0.173 |
  | ENZYMES | sfuse2 | 0.034 | 0.071 | 0.279 | 0.300 | 0.003 |

- **Wniosek:** **Trafienie na PROTEINS** — `wltopo` osiąga ARI topo (0.071) przy recon
  -0.402 (prywatność wl). Czyli fuzja wl+topo = użyteczność topo + prywatność wl → ścisła
  poprawa Pareto. Na ENZYMES fuzja nie podnosi ARI ponad samo `wl` (0.046), ale sfuse2 ma
  najwyższe NMI (0.071) i zachowuje prywatność. `sfuse` rozcieńcza topo netlsd/graphletem
  (gorzej niż wltopo). **Decyzja:** wltopo → leaderboard PROTEINS (#1, lepsza prywatność).

### 2026-06-23 · H3 — FGSD (histogram odległości biharmonicznych) — nowy deskryptor
- **Zmiana kodu:** `embeddings._fgsd` + metoda `fgsd` (pseudoinwersja Laplasjanu →
  odległości biharmoniczne par węzłów → histogram 200-binowy, clip do [0,50], norm. l2).
- **Komendy:** `--methods fgsd` na PROTEINS (per-class 200), ENZYMES (pełne), CIFAR (0/1/8).
- **Wynik:**

  | zbiór | ARI | acc | recon |
  |---|---|---|---|
  | PROTEINS | **0.074** | 0.605 | -5.23 |
  | ENZYMES | 0.027 | 0.243 | -3.46 |
  | CIFAR | 0.036 | 0.388 | -0.79 |

- **Wniosek:** FGSD **wygrywa ARI na PROTEINS** (0.074 > topo 0.071) przy skrajnej
  prywatności (recon mocno ujemny — czysta struktura, zero treści). Na ENZYMES i CIFAR
  słaby (małe/regularne grafy SLIC mają uboższe widmo). Świetny kandydat do osi
  prywatność–użyteczność na PROTEINS. Uwaga: recon mocno ujemny to artefakt (200-wym
  histogram, MLP rozbiega się) — interpretować jako „nieodwracalne", nie ilościowo.

### 2026-06-23 · H2b — fuzje strukturalne + FGSD na CIFAR
- **Wynik:** wltopo 0.062, sfuse 0.048, sfuse2 0.047, fgsd 0.036 — wszystkie **gorsze** niż
  samo `wl` (0.078). **Wniosek:** na grafach z obrazu fuzja rozcieńcza najlepszy strumień;
  `wl` zostaje liderem struktury na CIFAR. Fuzje pomagają na grafach molekularnych, nie obrazowych.

## Cel B — krzywe prywatność–użyteczność

### 2026-06-23 · B1 — krzywa rewire (degree-preserving) na PROTEINS
- **Komenda:** `--methods topo wl fgsd attr --privacy-curve --obf-method rewire`
- **Wynik (ARI / recon po sile s):**

  | s | topo ARI/recon | wl ARI/recon | fgsd ARI/recon | attr ARI/recon |
  |---|---|---|---|---|
  | 0.00 | 0.071 / 0.042 | 0.055 / -0.372 | 0.074 / -5.23 | 0.028 / 0.827 |
  | 0.25 | 0.091 / 0.055 | 0.055 / -0.346 | 0.023 / -0.725 | 0.028 / 0.827 |
  | 0.50 | 0.088 / 0.061 | 0.053 / -0.391 | 0.034 / -0.546 | 0.028 / 0.827 |
  | 1.00 | 0.076 / 0.032 | 0.055 / -0.395 | 0.019 / -0.504 | 0.028 / 0.827 |

- **Wniosek:** Rewire zachowuje stopnie → `topo` (deskryptor stopni) jest **niewrażliwy**
  (ARI stałe ~0.08) — zła para obfuskacja↔reprezentacja. Niszczy za to `fgsd` (spektralny:
  0.074→0.019). `attr` całkowicie płaski (recon 0.827 niezależne od s) — **obfuskacja
  krawędzi nie redukuje wycieku atrybutowego**.

### 2026-06-23 · B2 — edge-DP (randomized response z ε) na PROTEINS — NOWA METODA
- **Zmiana kodu:** `data.obfuscate_graph` metoda `edp`: RR na każdej parze węzłów, f=0.5·s,
  ε=ln((1-f)/f). s=0→ε=∞ (brak zmian); s=0.25→ε≈1.95; s=0.5→ε≈1.10; s=1→ε=0 (pełny szum).
  Dodane do `--obf-method`.
- **Komenda:** `--methods topo wl fgsd attr --privacy-curve --obf-method edp`
- **Wynik (ARI / recon):**

  | s (ε) | topo | wl | fgsd | attr |
  |---|---|---|---|---|
  | 0.00 (∞) | 0.071 / 0.042 | 0.055 / -0.372 | 0.074 / -5.23 | 0.028 / 0.827 |
  | 0.25 (1.95) | 0.071 / 0.062 | 0.051 / -0.174 | 0.031 / -0.096 | 0.028 / 0.827 |
  | 0.50 (1.10) | 0.063 / 0.042 | 0.048 / -0.207 | 0.079 / -0.015 | 0.028 / 0.827 |
  | 1.00 (0.00) | 0.063 / 0.021 | 0.051 / -0.247 | 0.094 / -0.164 | 0.028 / 0.827 |

- **Wniosek:** edge-DP daje **formalny parametr ε** na osi prywatność–użyteczność.
  Densyfikacja (RR dodaje losowe krawędzie) psuje `topo` (0.071→0.063) i degraduje stopnie.
  `attr` znów płaski (recon 0.827 dla każdego ε) — **kluczowy, powtarzalny wynik: edge-DP
  chroni TYLKO strukturę; treść (atrybuty) wymaga osobnego mechanizmu** (np. szum na cechach
  węzłów). Reprezentacje strukturalne (wl/fgsd) są wewnętrznie prywatne (recon<0) niezależnie
  od ε. (fgsd ARI rośnie przy małym ε — prawdopodobnie histogram chwyta gęstość/rozmiar
  skorelowane z klasą; traktować ostrożnie, nie jako realny zysk użyteczności.)

### 2026-06-23 · B3 — edge-DP na CIFAR (3 klasy 0/1/8, SLIC)
- **Komenda:** `--source cifar --classes 0 1 8 --per-class 200 --methods topo wl attr hog --privacy-curve --obf-method edp`
- **Wynik (ARI / recon):**

  | s (ε) | topo | wl | attr | hog |
  |---|---|---|---|---|
  | 0.00 (∞) | 0.056 / 0.286 | 0.078 / 0.508 | 0.118 / 0.877 | 0.019 / -0.304 |
  | 0.25 (1.95) | 0.027 / 0.196 | 0.077 / 0.403 | 0.118 / 0.877 | 0.019 / -0.304 |
  | 0.50 (1.10) | 0.022 / 0.194 | 0.077 / 0.417 | 0.118 / 0.877 | 0.019 / -0.304 |
  | 1.00 (0.00) | 0.034 / 0.186 | 0.077 / 0.399 | 0.118 / 0.877 | 0.019 / -0.304 |

- **Wniosek:** Na grafach z obrazu `topo` degraduje pod edge-DP (0.056→0.034, recon 0.29→0.19),
  ale `wl` **niezwykle stabilne** (ARI 0.078→0.077) — jego sygnał pochodzi z etykiet koloru
  węzłów (seed WL), których obfuskacja krawędzi NIE rusza; recon spada 0.508→0.40 → **lekki
  zysk prywatności bez kosztu użyteczności**. `attr`/`hog` całkiem płaskie (immunne na zmiany
  krawędzi). Powtarza się wniosek z PROTEINS: edge-DP chroni strukturę, nie treść.
- **Figury:** `results_prot_priv_edp/fig_privacy_utility.png`, `results_cifar_priv_edp/fig_privacy_utility.png`.

## Cel C — grafy z obrazu (CIFAR), przeszukanie budowy grafu

### 2026-06-23 · C1 — `--rich-features` (mini-HOG na superpiksel)
- **Komenda:** `--source cifar --classes 0 1 8 --per-class 200 --rich-features --methods topo wl g2v attr rgb hog hyb`
- **Wynik:** attr ARI **spadł** 0.118→0.089; wl/g2v bez zmian ARI (0.078/0.062); hyb acc 0.665.
- **Wniosek:** Bogatsze cechy węzła (HOG+pozycja+rozmiar) **zaszumiają pooling atrybutowy**
  (mean⊕std) → gorsze klastrowanie attr. Nie pomaga ARI. Odrzucam dla klastrowania.

### 2026-06-23 · C2 — `--label-rich` (seed WL po pełnym deskryptorze koloru, nie tylko mean) ★
- **Komenda:** `--source cifar --classes 0 1 8 --per-class 200 --label-rich --methods topo wl g2v attr`
- **Wynik:**

  | method | ARI | NMI | silh | acc | recon | vs baseline ARI |
  |---|---|---|---|---|---|---|
  | topo | 0.056 | 0.069 | 0.267 | 0.525 | 0.286 | = (nie używa seedu) |
  | wl | **0.108** | 0.102 | 0.214 | 0.613 | 0.482 | 0.078 → 0.108 |
  | g2v | **0.116** | 0.111 | 0.058 | 0.613 | 0.462 | 0.062 → 0.116 |
  | attr | 0.118 | 0.117 | 0.270 | 0.652 | 0.877 | = (nie używa seedu) |

- **Wniosek:** ★ **Trafienie.** Bogatszy seed WL (KMeans na mean⊕std koloru zamiast samego
  mean[:3]) podnosi WL-owe metody do poziomu `attr` (g2v 0.116 ≈ attr 0.118) przy recon
  **0.46 vs 0.88** — czyli ta sama jakość klastrowania, ~2× mniejszy wyciek. Najlepszy
  strukturalny wynik na CIFAR. `--label-rich` wchodzi na stałe do konfiguracji CIFAR.

### 2026-06-23 · C3 — `--n-segments 100` (+ label-rich)
- **Zmiana kodu:** dodane flagi CLI `--n-segments`, `--compactness`.
- **Komenda:** `--label-rich --n-segments 100 --methods topo wl g2v attr` (|V| med 110 vs ~?60).
- **Wynik:** topo **0.056→0.100** (drobniejszy graf = dyskryminujące stopnie), wl 0.097,
  g2v **0.116→0.060** (spadł), attr 0.101. **Wniosek:** więcej segmentów pomaga topo, psuje
  g2v — netto nie bije seg60+label-rich (g2v 0.116). Zostaję przy domyślnym n_segments=60.

### 2026-06-23 · C4 — fuzja strukturalna+attr pod label-rich (sweep wagi) — RYGOR
- **Komenda:** `--label-rich --methods wltopo combo gnat hyb` (seg60).
- **Wynik pełnego sweepu (gnat = [g2v,topo]⊕attr):**

  | w | gnat ARI | gnat NMI | silh | acc | combo ARI |
  |---|---|---|---|---|---|
  | 0.00 (attr) | 0.118 | 0.117 | **0.270** | **0.652** | 0.118 |
  | 0.50 | **0.130** | **0.130** | 0.164 | 0.630 | 0.118 |
  | 0.75 | 0.063 | 0.075 | 0.147 | 0.630 | 0.120 |
  | 1.00 (struct) | 0.058 | 0.071 | 0.155 | 0.602 | 0.116 |

- **Wniosek:** Fuzja 50/50 struktury i atrybutów osiąga **ARI 0.130** (najwyższe na CIFAR),
  bijąc oba czyste strumienie. **ALE** ani silhouette (max przy w=0), ani probe-CV (acc max
  przy w=0) nie wskazuje w=0.5 → to punkt osiągalny tylko z **oracle ARI**, więc **NIE
  liczę go jako uczciwy wynik** (byłoby strojeniem pod etykiety). Zostawiam jako diagnostykę:
  „sygnał strukturalny i atrybutowy są komplementarne", ale label-free selekcja go nie znajduje.
  Honestny lider CIFAR pozostaje `g2v+label-rich` (0.116, recon 0.462).

### 2026-06-23 · C5 — `--edge-quantile` 0.4 (rzadszy graf) + label-rich
- **Wynik:** g2v 0.116→**0.096** (rzadszy graf psuje g2v), topo/attr ~bez zmian.
- **Wniosek:** Rzadszy graf nie pomaga; domyślne edge-quantile 0.6 zostaje. (Gęstszy 0.8
  pominięty — kierunek nieobiecujący.)

### 2026-06-23 · B4 — feature-DP (szum na cechach węzłów) — NOWY MECHANIZM (prywatność treści) ★
- **Motywacja:** edge-DP/rewire NIE redukują wycieku atrybutowego (recon attr stały) — brakowało
  mechanizmu chroniącego TREŚĆ. 
- **Zmiana kodu:** `data.obfuscate_features` + `--obf-method feature`: szum gaussowski
  σ_d = strength·std_d (globalne odchylenie cechy). Struktura nietknięta, etykiety (seed WL)
  zostają — izoluje prywatność na poziomie atrybutów.
- **Komenda:** `--methods attr topo --privacy-curve --obf-method feature --obf-strengths 0 0.5 1 2 4` (PROTEINS p-c 200)
- **Wynik (attr):**

  | strength (×std) | ARI | acc | recon_r2 |
  |---|---|---|---|
  | 0.0 | 0.028 | 0.722 | 0.827 |
  | 0.5 | 0.020 | 0.650 | 0.714 |
  | 1.0 | 0.027 | 0.670 | 0.629 |
  | 2.0 | 0.071 | 0.657 | 0.489 |
  | 4.0 | 0.107 | 0.628 | 0.232 |

  `topo`/`wl`/`fgsd` (kontrola): **całkowicie płaskie** (struktura immunna na szum cech).
- **Wniosek:** ★ Feature-DP **spycha recon treści** (acc spada łagodnie) — skuteczny knob
  prywatności TREŚCI, którego edge-DP nie dawał. Razem z edge-DP daje **dwuosiowe pokrętło
  prywatności: struktura ⟂ treść** (każda oś chroni inny strumień).
  > ⚠️ **KOREKTA AUDYTU (patrz sekcja AUDYT FINALNY):** powyższe 0.827→0.232 pochodzi ze
  > SŁABEGO ataku MLP. Pod ODPORNYM atakiem (RidgeCV+MLP, pełne dane) feature-DP redukuje
  > recon attr **0.764 → 0.485** (σ=0→4) — realnie, ale mniej radykalnie. Aktualne liczby:
  > `results_FINAL_priv_feat`. Wniosek jakościowy (struktura płaska, treść maleje) — bez zmian.

## Potwierdzenie najlepszych konfiguracji na PEŁNYCH DANYCH

### 2026-06-23 · V1 — pełny PROTEINS_full (1113 grafów, bez podpróbkowania)
- **Komenda:** `python run_experiments.py --source tu --dataset PROTEINS_full --methods topo wl fgsd wltopo attr --out-dir results_prot_FULL`
- **Wynik:** fgsd ARI **0.135** (recon -8.58) > topo/wltopo 0.080 > attr 0.037. wltopo
  recon -0.164 (vs topo 0.093) — Pareto. **Wniosek:** FGSD potwierdza i **wzmacnia** przewagę
  na pełnych danych (0.074→0.135, ~1.7× topo) przy skrajnej prywatności. Leaderboard PROTEINS
  zaktualizowany na liczby z pełnych danych.

### 2026-06-23 · V2 — potwierdzenie CIFAR na skali (per-class 500 → 1500 grafów)
- **Komendy:** `--per-class 500 --label-rich --methods topo wl g2v attr rgb hog` oraz wariant
  bez `--label-rich` (wl g2v topo).
- **Wynik (label-rich vs bez):**

  | method | ARI @200/kl (lr) | ARI @500/kl (lr) | ARI @500/kl (bez lr) |
  |---|---|---|---|
  | wl | 0.108 | 0.083 | 0.082 |
  | g2v | 0.116 | 0.065 | 0.082 |
  | topo | 0.056 | 0.045 | — |
  | attr | 0.118 | 0.124 | — |

- **Wniosek:** ⚠️ **label-rich nie skaluje.** Przewaga z per-class 200 znika przy 1500 grafach
  (wl 0.082 niezależnie od label-rich; g2v z label-rich nawet gorsze: 0.065 vs 0.082). To
  był artefakt małej próbki — dlatego potwierdzamy na pełnych danych. Robustny wynik
  strukturalny na CIFAR: **wl ≈ 0.082** (recon ~0.53), > baseline'y wyglądu (rgb 0.071,
  hog 0.008) → struktura grafu realnie dokłada sygnał ponad wygląd, zgodnie z odniesieniem.
  Doc2Vec (g2v) jest czuły na rozmiar korpusu/seed — niestabilny między skalami.

## Cel A — poprawa klastrowania (bateria algorytmów na najlepszych embeddingach)

Hipoteza: wąskim gardłem jest krok klastrowania (probe_acc >> szansy, ARI niskie). Narzędzie:
`cluster_lab.py` — buduje 1 embedding, testuje baterię klastrowań + diagnozę η²(rozmiar/gęstość).
Selekcja label-free (silhouette/CH/DB); ARI/NMI jako wynik.

### 2026-06-23 · A1 — PROTEINS_full (pełne 1113), embedding fgsd, k=2
- **Diagnoza:** η²(size|class)=0.084, η²(dens|class)=**0.183** (gęstość koreluje z klasą).
  Klastry KMeans: η²(dens)=**0.412** — dzielą po gęstości MOCNIEJ niż gęstość niesie klasę.
- **Bateria (top):**

  | algorytm | ARI | NMI | silh | CH | DB | ηsize | ηdens |
  |---|---|---|---|---|---|---|---|
  | kmeans (baseline) | **0.135** | 0.086 | 0.277 | 460 | 1.47 | 0.136 | 0.412 |
  | pca-whiten+sph-kmeans | 0.135 | 0.088 | 0.062 | 70 | 3.38 | 0.076 | 0.411 |
  | umap(nn10,md0)+gmm | 0.128 | **0.100** | 0.538 | 591 | 1.05 | 0.083 | 0.584 |
  | umap(nn30,md0.1)+gmm | 0.126 | 0.096 | 0.524 | 851 | 0.82 | 0.085 | 0.576 |
  | gmm-full / gmm-diag | ~0.00 | 0.04 | 0.16 | — | — | — | — |

- **Wniosek:** **KMeans (0.135) zostaje najlepszy na ARI** — żaden wariant go nie bije.
  UMAP→GMM podnosi silhouette (0.54) i lekko NMI (0.100 vs 0.086), ale ARI nie. **GMM
  bezpośrednio zawodzi** (200-wym histogram → kowariancja osobliwa, ARI≈0). PCA-whitening
  bez UMAP psuje (η-struktura znika). Diagnoza: klastry częściowo dzielą po gęstości — ale
  to NIE jest naprawialne lepszym klastrowaniem (gęstość sama słabo niesie klasę: 0.183).
  Sufit ARI≈0.135 to własność embeddingu, nie kroku klastrowania (dla fgsd/PROTEINS).

### 2026-06-23 · A2 — CIFAR (per-class 500, 1500 grafów), embedding wl, k=3
- **Diagnoza:** η²(size|class)=0.081, η²(dens|class)=0.112 (słabe). Klastry: η²(size/dens)≈0.05
  → **NIE są napędzane rozmiarem/gęstością** (inaczej niż PROTEINS/fgsd).
- **Bateria (top):**

  | algorytm | ARI | NMI | silh | CH | DB |
  |---|---|---|---|---|---|
  | **gmm-full** | **0.085** | 0.080 | 0.193 | 284 | 1.88 |
  | kmeans (baseline) | 0.082 | 0.086 | 0.222 | 320 | 1.69 |
  | umap(nn50,md0)+gmm | 0.079 | 0.083 | 0.568 | 1829 | 0.64 |
  | pca-whiten+* | 0.04–0.06 | — | — | — | — |

- **Wniosek:** **GMM-full daje marginalny zysk (0.082→0.085 ARI).** UMAP nie bije KMeans na
  ARI (choć silhouette 0.57). **Kluczowy wniosek diagnostyczny:** wysokie probe_acc (0.57) +
  niskie ARI + klastry NIE-rozmiarowe ⇒ klasa jest **liniowo separowalna, ale nie jest
  dominującą osią wariancji** → klastrowanie bez etykiet jej nie znajduje. To **fundamentalna
  własność embeddingu**, nie wada algorytmu — lepsze klastrowanie nie odblokowuje dużego ARI.

### 2026-06-23 · A3 / RYGOR#5 — odporny atak rekonstrukcyjny (RidgeCV + MLP, max; metryki [0,1])
- **Zmiana kodu:** `evaluate.reconstruction_attack` próbuje DWÓCH modeli (RidgeCV auto-α + MLP)
  i bierze lepszy (max R² = najsilniejszy atakujący). Nowe kolumny CSV: `recon_nmse`=1-R²,
  `recon_leak`=clip(R²,0,1). 
- **Przyczyna artefaktu -8.58:** `Ridge(α=1)`/MLP **przeuczały się** na 200-wym FGSD (target
  PROTEINS jest 29-wym). RidgeCV(α∈{0.1..1000}) stabilizuje: FGSD recon **-8.575 → 0.042**.
- **Wynik (PROTEINS, recon_leak):** fgsd 0.042 < topo 0.099 < wl 0.103 < wltopo 0.160 < attr 0.764.
- **Wniosek:** ★ Naprawione. Ujemne recon metod strukturalnych były **artefaktem MLP** — pod
  uczciwym atakiem liniowym wyciek jest mały, ale dodatni. **fgsd najmniej przeciekowy** (0.042)
  ORAZ najlepszy ARI → ścisły zwycięzca. Korekta porządku prywatności w leaderboardzie PROTEINS.

### 2026-06-23 · A4 / RYGOR#6 — label-free dobór wagi fuzji (CIFAR wl⊕attr, 1500 grafów)
- **Narzędzie:** `fusion_select.py` — sweep w, ARI (wynik) vs silhouette/CH/DB/stabilność bootstrapowa.
- **Wynik:** komplementarny optimum **w=0.5 → ARI 0.130** (bije czysty attr w=0:0.124 i czystą
  strukturę w=1:0.082) — **realny i robustny na skali, bez label-rich.** Kryteria label-free:

  | kryterium | wybiera w | tam ARI | trafia w 0.5? |
  |---|---|---|---|
  | silhouette / CH / DB | 0.0 | 0.124 | ✗ (woli zwarty attr) |
  | stabilność bootstrap | 1.0 | 0.082 | ✗ (struktura najstabilniejsza) |
  | **oracle ARI** | **0.5** | **0.130** | — |

- **Wniosek:** Komplementarność struktury i treści daje realny zysk (0.130), ale **żadne
  standardowe kryterium bezetykietowe go nie znajduje** — indeksy wewnętrzne preferują skrajny
  zwarty (attr) lub skrajnie stabilny (struktura) koniec, nie środek. Uczciwie: najlepsza
  label-free selekcja (silhouette→w=0) daje 0.124 — blisko 0.130, więc realna strata jest
  mała. **To ograniczenie metody, nie do obejścia prostym kryterium** (potrzebny byłby sygnał
  nadzorowany = wyciek). Stab rośnie monotonicznie z w → struktura czystsza/stabilniejsza.

### 2026-06-23 · A4b / RYGOR#6 — fuzja na PROTEINS (fgsd⊕attr, pełne 1113)
- **Wynik:** komplementarny optimum **w=0.6 → ARI 0.155** (vs fgsd 0.135, attr 0.037) — skok
  z 0.047 (w=0.5) na 0.155 (w=0.6). Kryteria label-free: silhouette/CH/DB → **w=1.0** (czysty
  fgsd, ARI 0.135 = honest lider); stabilność → w=0.1 (attr). **Żadne nie trafia w 0.6.**
- **Wniosek:** Powtórzenie wzorca z CIFAR: **fuzja struktura⊕treść ma komplementarny optimum
  (0.155 PROTEINS / 0.130 CIFAR) bijący oba czyste strumienie, ale nieselektowalny label-free.**
  silhouette wybiera bardziej zwarty z dwóch czystych końców (fgsd na PROTEINS, attr na CIFAR).
  Honest wynik = czysty fgsd 0.135 (silhouette go potwierdza). Oracle 0.155 zostaje jako
  diagnostyka „dostępne, gdyby był sygnał nadzorowany".

### 2026-06-23 · A5 — bateria klastrowania ENZYMES (pełne 600, wl, k=6)
- **Bateria (top):** umap(nn30,md0.1)+kmeans ARI **0.052** / NMI **0.095** > kmeans 0.046/0.086.
  Kilka wariantów UMAP+kmeans 0.045–0.052. gmm-full 0.026 (gorzej). pca-whiten psuje.
- **Uwaga selekcji:** najlepsze ARI UMAP (0.052) ma silh 0.451, a silh-max to umap(nn10) z
  ARI tylko 0.021 → **dobra konfiguracja UMAP nie jest label-free selektowalna.** Klastry
  nie-rozmiarowe (ηsize~0.02).
- **Wniosek:** ENZYMES — UMAP→KMeans oferuje drobny zysk (0.046→0.052), ale niewybierany
  uczciwie. Sufit ENZYMES ~0.05 pozostaje (zbiór wewnętrznie trudny, zgodnie z założeniem).

## PODSUMOWANIE FAZY: poprawa klastrowania (Cel A)
**Teza zweryfikowana częściowo: klastrowanie NIE jest dominującym wąskim gardłem.** Lepsze
algorytmy dają tylko marginalne zyski ARI:
- CIFAR: GMM-full 0.082→**0.085**; ENZYMES: UMAP→KMeans 0.046→**0.052**; PROTEINS: brak (KMeans
  0.135 najlepszy). UMAP mocno poprawia **silhouette** (0.5+), ale to kosmetyka — nie ARI.
- **GMM-full** pomaga w niskim wymiarze (CIFAR wl 64-wym), **zawodzi** w wysokim (fgsd 200-wym,
  osobliwa kowariancja). **PCA-whitening** psuje (niszczy strukturę η). 
- **Diagnoza η²:** PROTEINS/fgsd klastry częściowo dzielą po gęstości (ηdens 0.41), ale CIFAR/
  ENZYMES — nie po rozmiarze. **Prawdziwa przyczyna niskiego ARI:** wysokie probe_acc + niskie
  ARI ⇒ klasa jest LINIOWO SEPAROWALNA, ale nie jest dominującą osią wariancji → klastrowanie
  bez etykiet jej nie znajduje. To własność embeddingu/danych, nie wada algorytmu.
- **Fuzja struktura⊕treść** ma realny komplementarny optimum (PROTEINS fgsd⊕attr w=0.6 → **0.155**;
  CIFAR wl⊕attr w=0.5 → **0.130**) bijący oba czyste strumienie — ale **żadne kryterium
  label-free go nie wybiera** (silhouette/CH/DB → zwartszy czysty koniec; stabilność → struktura).
- **Rygor:** atak rekonstrukcyjny uodporniony (RidgeCV+MLP max, metryki [0,1]) — naprawiony
  artefakt -8.58; ujemne recon strukturalne okazały się artefaktami MLP.

**Honest leadery bez zmian co do ARI** (klastrowanie nie odblokowało dużych zysków); główna
wartość fazy: **diagnoza, że sufit ARI to własność embeddingu (klasa nie jest osią wariancji),
oraz dwie poprawki rygoru.**

### 2026-06-23 · A6 — jądro WL (kernel-KMeans / spectral) + ORACLE LDA upper-bound
- **Narzędzie:** `closeA.py`. Jądro WL = kosinusowe na histogramach poddrzew → spectral
  (precomputed) / kernel-KMeans. Oracle = LDA(X,y)→KMeans (UŻYWA ETYKIET = wyciek, osobno).
- **Jądro WL (honest, label-free):**

  | zbiór | metoda | ARI | NMI | vs lider |
  |---|---|---|---|---|
  | PROTEINS | wl-kernel spectral/kmeans | 0.040 / 0.021 | 0.026 | << fgsd 0.135 |
  | CIFAR | wl-kernel spectral-disc | **0.084** | 0.090 | ≈ wl 0.082 (marginal) |

- **ORACLE upper-bound (z etykietami — NIE do leaderboardu honest):**

  | zbiór | embedding | honest ARI | **oracle LDA ARI** | skok |
  |---|---|---|---|---|
  | PROTEINS | wl-kernel | 0.040 | **0.315** | **×8** |
  | PROTEINS | fgsd | 0.135 | 0.188 | ×1.4 |
  | CIFAR | wl-kernel | 0.084 | **0.226** | ×2.6 |
  | CIFAR | wl | 0.082 | 0.220 | ×2.7 |

- **Wniosek:** ★ **Jądro WL nie bije liderów honest** (PROTEINS dużo gorzej; CIFAR marginalnie).
  Ale **ORACLE jest definitywnym dowodem diagnozy:** gdy oś klasy jest znana (LDA), ARI skacze
  ×1.4–×8 (WL-kernel PROTEINS 0.040→**0.315**). **Sygnał klasy JEST w reprezentacji strukturalnej,
  tylko nie jest dominującą osią wariancji** → klastrowanie bez etykiet go nie wydobywa. To
  zamyka Cel A: separowalność (probe/oracle) ≠ klastrowalność (honest ARI).

## ABLACJA — grafy losowe jako REPREZENTACJA (struktura vs losowość)

### 2026-06-23 · ABL1 — prawdziwy graf vs ER vs dropedge (najlepszy deskryptor strukturalny)
- **Pytanie:** ile sygnału klastrowania pochodzi z PRAWDZIWEJ struktury vs losowej? Dla
  najlepszego deskryptora liczymy km_ARI na (i) prawdziwym grafie, (ii) ER o dopasowanej
  gęstości, (iii) dropedge p=0.5. Atrybuty/etykiety węzłów ZACHOWANE (zmieniamy tylko strukturę).
- **Narzędzie:** `random_ablation.py` (reużywa `data.obfuscate` jako generator losowych grafów).
- **Wynik:**

  | zbiór / deskryptor | wariant | km_ARI | km_NMI |
  |---|---|---|---|
  | PROTEINS_full / **fgsd** | (i) prawdziwy | **0.135** | 0.086 |
  | | (ii) ER dopasowana gęstość | 0.065 | 0.035 |
  | | (iii) dropedge p=0.5 | 0.058 | 0.048 |
  | ENZYMES / **wl** | (i) prawdziwy | **0.046** | 0.086 |
  | | (ii) ER dopasowana gęstość | 0.037 | 0.080 |
  | | (iii) dropedge p=0.5 | 0.039 | 0.064 |

- **Wniosek:** ★ **PROTEINS/fgsd: prawdziwa struktura niesie ~2× sygnału losowej** (0.135 vs
  ~0.06) — topologia realnie koduje klasę (fgsd jest czysto spektralny). Resztkowe ~0.06 dla
  ER bierze się z dopasowanej gęstości (gęstość słabo koreluje z klasą, ηdens=0.18).
  **ENZYMES/wl: różnica mała** (0.046 vs ~0.038) — bo `wl` jest seedowany ETYKIETAMI węzłów
  (typ), które obfuskacja krawędzi zachowuje → jego (i tak słaby) sygnał płynie głównie ze
  **składu etykiet węzłów, nie z topologii**. Czyli „struktura > losowość" trzyma się mocno
  dla deskryptorów czysto topologicznych (fgsd), a dla WL z etykietami sygnał jest hybrydowy
  (etykiety + trochę topologii). Honest leaderboardy bez zmian (to ablacja, nie nowy wynik).

## NOWA METODA — rand_ens (ensembling losowych grafów jako reprezentacja)

### 2026-06-23 · RE1 — rand_ens: M losowych wariantów struktury + uśrednianie deskryptora
- **Pomysł (z notatnika cifar_rand_graphs2):** dla KAŻDEGO obiektu generuj M losowych wariantów
  jego struktury (dropedge/shortcuts/er), policz bazowy deskryptor na każdym i UŚREDNIJ →
  embedding. Etykiety/atrybuty węzłów zachowane. Implementacja: bazowy embedder liczony RAZ na
  korpusie M·N grafów (wl/g2v dzielą słownik+SVD; topo/... per-graf). Wpięte jako
  `--methods rand_ens`, params: `--rand-ens-m/-base/-method/-p` (domyślnie M=10, topo, dropedge, p=0.5).
- **Wynik (pełne dane):**

  | zbiór | rand_ens(topo) | topo (PRAWDZIWY graf) | rand_ens(wl) | wl (PRAWDZIWY) | lider (fgsd) |
  |---|---|---|---|---|---|
  | PROTEINS_full | **0.089** / NMI 0.072 | 0.080 / 0.063 | 0.007 | 0.040 | 0.135 |
  | ENZYMES | 0.018 / 0.038 | 0.030 / 0.059 | 0.016 | 0.046 | (wl 0.046) |

- **Wniosek (odpowiedź na pytanie: lepszy czy gorszy niż pojedynczy prawdziwy graf?):**
  **Przeważnie GORSZY.** Jedyny wyjątek: **rand_ens(topo) na PROTEINS 0.089 > topo real 0.080**
  (+NMI) — uśrednianie deskryptora STOPNI po dropedge-wariantach działa jak bagging/denoising
  topo (degree stats są odporne na losowe usuwanie krawędzi, a uśrednianie wygładza szum).
  Wciąż **poniżej lidera fgsd (0.135)**, a silhouette woli topo (0.318 > 0.291) — więc label-free
  nie preferuje rand_ens nad topo. **base=wl niszczy sygnał** (PROTEINS 0.007, ENZYMES 0.016 —
  relabeling WL na losowych grafach + uśrednianie joint-SVD zmywa strukturę). ENZYMES: rand_ens
  pogarsza (0.018<0.030 — dropedge niszczy i tak słabą topologię). **Spójne z ablacją ABL1:**
  losowa struktura niesie mniej sygnału niż prawdziwa; ensembling to nie naprawia (poza wąskim
  efektem denoisingu degree-stats na PROTEINS). Honest lider PROTEINS bez zmian (fgsd 0.135).

## PRYWATNOŚĆ — wizualny atak inwersyjny (CIFAR)

### 2026-06-23 · INV1 — odtworzenie OBRAZU z embeddingu grafu (nie tylko średniej treści)
- **Komenda:** `python inversion_attack.py --classes 0 1 8 --per-class 200 --out-dir results_inversion`
- **Atak:** regresor embedding_grafu → obraz uśredniony po superpikselach SLIC (kolor-per-węzeł
  złożony po masce), CV, silniejszy z {RidgeCV, MLP}. Metryki: SSIM↑/MSE↓ do oryginału.
  Degradacja przy feature-DP σ∈{0,1,2,4}.
- **Wynik (SSIM / MSE; ↑SSIM = gorsza prywatność):**

  | metoda | σ=0 | σ=1 | σ=2 | σ=4 |
  |---|---|---|---|---|
  | **attr** (treść) | **0.198 / 0.037** | 0.184 / 0.039 | 0.181 / 0.041 | 0.173 / 0.045 |
  | wl (struktura, seed=kolor) | 0.153 / 0.041 | 0.153 / 0.041 | (σ-niezmienne) | 0.153 / 0.041 |
  | fgsd (czysta topologia) | 0.165 / 0.058 | (σ-niezmienne) | | 0.165 / 0.058 |

- **Wniosek:** ★ **attr odtwarza obraz najlepiej** (SSIM 0.198, MSE 0.037 — najwyższa wierność)
  i **degraduje monotonicznie pod feature-DP** (SSIM 0.198→0.173, MSE 0.037→0.045). `fgsd`
  (czysta topologia) = szara breja (MSE 0.058, zero koloru — nieodwracalny). `wl` pośrednio:
  odtwarza niebieskawą plamę, bo jego seed to etykiety KOLORU (nie redukowane przez feature-DP,
  bo nadawane na czystych cechach). **Ważny niuans:** wszystkie embeddingi są permutacyjnie
  niezmiennicze → nie kodują UKŁADU przestrzennego, więc rekonstrukcje to „plamy koloru", nie
  ostre obrazy; różnicuje je przede wszystkim wierność KOLORU (attr ≫ fgsd). Potwierdza:
  treść (attr) jest odwracalna i chroniona przez feature-DP; czysta struktura (fgsd) nie niesie
  wyglądu. Figura: `results_inversion/fig_inversion.png`.

---

## AUDYT FINALNY (re-weryfikacja całości na pełnych danych odpornym atakiem)

Pełny przegląd + re-run wszystkiego, na czym opieramy opis. Wyniki w `results_FINAL_*`.

### Co AKTUALNE vs PRZESTARZAŁE
- **AKTUALNE (wiążące):** leaderboardy `results_FINAL_{prot,enz,cifar}`, krzywe
  `results_FINAL_priv_{feat,edp}`, siatka `results_FINAL_priv2d`, `results_inversion`,
  oracle (closeA), sensitivity.
- **PRZESTARZAŁE (tylko historyczne):** wszystkie `results_*` z per-class 200 (baseline'y,
  B1–B4, C1–C5) — używały **starego ataku MLP** (recon_r2 z artefaktami: ujemne wartości,
  -8.58 dla FGSD). Liczby ARI z podpróbek (np. FGSD PROTEINS 0.074, label-rich CIFAR 0.108)
  były później skorygowane na pełnych danych (V1/V2). NIE używać do opisu.

### Co ZWERYFIKOWANE (re-run potwierdził)
- **Leaderboardy strukturalne — bez zmian:** PROTEINS fgsd 0.135 / leak 0.046; ENZYMES wl
  0.046 / leak 0.077; CIFAR wl 0.082 / leak 0.712, g2v 0.082 / leak 0.651.
- **Stabilność (mean±std):** fgsd PROTEINS ARI **0.136±0.001** (5 seedów), CIFAR wl ARI
  **0.079±0.008** (3 seedy). recon_leak stabilny: fgsd 0.046±0.002, attr PROTEINS 0.765±0.006,
  attr CIFAR 0.891±0.004 → liczby prywatności twarde.
- **edge-DP (full PROTEINS, robust):** recon[attr] = **0.764 stałe** dla wszystkich σ_edge
  (treść niezależna od edge-DP); topo ARI degraduje 0.080→0.046.
- **feature-DP (full PROTEINS, robust, 7 σ):** recon[attr] **0.764→0.485** monotonicznie;
  topo 0.099, wl 0.103 — idealnie płaskie.
- **Siatka 2D (full, robust):** recon[attr] niezmienny wzdłuż edge-DP (**std=0.0000**), malejący
  wzdłuż feature-DP → czysty dowód struktura⟂treść.
- **Oracle LDA:** PROTEINS WL-kernel 0.040→**0.315** (×8), fgsd→0.188; CIFAR wl 0.082→**0.220** (×2.6).
- **Atak inwersyjny:** attr SSIM 0.198/MSE 0.037 (najlepsze) > fgsd (MSE 0.058); feature-DP psuje attr.

### Co POPRAWIONE
1. **feature-DP pod odpornym atakiem: 0.764→0.485** (nie 0.827→0.232 ze słabego MLP) —
   skorygowano B4 i RESULTS.
2. **CIFAR g2v recon_leak 0.651** (precyzyjnie, < wl 0.712).
3. **PROTEINS netlsd 0.074** na pełnych danych (leak 0.058) — dodano do leaderboardu.
4. Ścieżki figur w RESULTS → `results_FINAL_*`.

**Finalne ustalenia i szkic referatu:** `RESULTS.md` (sekcja USTALENIA FINALNE) + `FINDINGS.md`.
