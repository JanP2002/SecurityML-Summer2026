# RESULTS — graphrep: reprezentacje grafowe pod kątem użyteczności i prywatności

Podsumowanie eksploracji (pełny dziennik: [`EXPLORATION_LOG.md`](EXPLORATION_LOG.md)).
Zbiory: **ENZYMES** (6 klas, 600 grafów), **PROTEINS_full** (2 klasy, 1113), **CIFAR-10**
grafy z obrazu (3 klasy 0/1/8, SLIC, 1500). Metryki: klastrowanie `km_ARI`/`km_NMI`
(nienadzorowane, raportowane jako **wynik** — nie cel strojenia), separowalność `probe_acc`
(sonda CV), prywatność `recon_leak ∈ [0,1]` (odporny atak rekonstrukcyjny, ↓ = prywatniej).

---

# USTALENIA FINALNE (po pełnym audycie i re-weryfikacji)

Wszystkie liczby poniżej **przeliczone od nowa na PEŁNYCH danych aktualnym kodem** (odporny
atak RidgeCV+MLP) w katalogach `results_FINAL_*`. Stare katalogi (`results_*_base`, per-class
200, stary atak MLP) uznane za PRZESTARZAŁE i niewiążące.

## Status weryfikacji (krok 1–2)
- ✅ **Leaderboardy strukturalne (ENZYMES/PROTEINS/CIFAR) — POTWIERDZONE** co do liczby
  (`results_FINAL_prot/enz/cifar`). fgsd PROTEINS 0.135, wl ENZYMES 0.046, wl/g2v CIFAR 0.082 —
  bez zmian.
- ✅ **edge-DP, feature-DP, 2D grid, atak inwersyjny, oracle LDA — POTWIERDZONE** na pełnych
  danych odpornym atakiem.

## Korekty wprowadzone w audycie
1. **feature-DP pod ODPORNYM atakiem jest mniej radykalny niż sugerował słaby MLP.** Stary wpis
   (B4) podawał recon attr 0.827 → **0.232** (σ=4) — to był słaby atak MLP. Robust (RidgeCV+MLP)
   na pełnych danych: **0.764 → 0.485**. Wyciek spada realnie, ale silny atakujący wyciąga więcej.
   Skorygowano w leaderboardach/krzywych.
2. **CIFAR `g2v` recon_leak = 0.651** (precyzyjnie; wcześniej „~0.7") — nieco prywatniejszy niż
   wl (0.712) przy tym samym ARI 0.082.
3. **PROTEINS `netlsd` na pełnych danych: ARI 0.074** (leak 0.058) — wysokie i prywatne; dodane
   do leaderboardu (per-class 200 dawało tylko 0.044).

## Wrażliwość na seed (krok 3; mean ± std po 3–5 seedach)
| metryka | wartość | uwaga |
|---|---|---|
| fgsd PROTEINS km_ARI | **0.136 ± 0.001** (0.135–0.138) | skrajnie stabilne — nie artefakt seeda |
| fgsd PROTEINS recon_leak | 0.046 ± 0.002 | stabilne, prywatne |
| attr PROTEINS recon_leak | 0.765 ± 0.006 | stabilne, przeciekowe |
| wl CIFAR km_ARI | **0.079 ± 0.008** (0.069–0.087) | umiarkowana wariancja (z podpróbki) |
| wl CIFAR recon_leak | 0.719 ± 0.007 | stabilne |
| attr CIFAR recon_leak | 0.891 ± 0.004 | stabilne |

**Wniosek:** liczby prywatności (recon_leak) są bardzo stabilne (±0.002–0.007). ARI fgsd/PROTEINS
jest twarde; ARI wl/CIFAR ma rozrzut ±0.008 (raportować jako ~0.08).

## Minimalny zestaw figur opowiadający historię (krok 4)
| # | figura | co pokazuje | ścieżka |
|---|---|---|---|
| 1 | benchmark PROTEINS | fgsd: najlepsze ARI **i** najniższy wyciek wśród struktury | `results_FINAL_prot/fig_benchmark.png` |
| 2 | **siatka 2D prywatności** | struktura ⟂ treść: recon[attr] niezależny od edge-DP, malejący z feature-DP | `results_FINAL_priv2d/fig_privacy_2d.png` |
| 3 | krzywa feature-DP | attr recon 0.76→0.49 z σ, struktura płaska | `results_FINAL_priv_feat/fig_privacy_utility.png` |
| 4 | **wizualny atak inwersyjny** | attr odtwarza obraz, struktura = śmieci; feature-DP psuje | `results_inversion/fig_inversion.png` |

**Tła klastrowania (tabele, nie figury):** leaderboardy + tabela honest vs oracle LDA.

**ODRZUCAM jako nadmiarowe:** krzywa edge-DP osobno (`results_FINAL_priv_edp` — siatka 2D już
pokazuje niezmienniczość względem edge-DP); benchmarki CIFAR/edge/feat (leaderboard wystarcza);
`fig_morphospace`; wszystkie `results_*` per-class 200 (stary atak — zastąpione przez `results_FINAL_*`).

## Tabela: wniosek × zbiór (krok 5)
| wniosek | ENZYMES | PROTEINS_full | CIFAR-10 |
|---|---|---|---|
| **A** separowalność ≠ klastrowalność | ✓ sufit ARI 0.046, probe 0.40 | ✓✓ **oracle ×8** (0.040→0.315) | ✓ oracle ×2.6 (0.082→0.220) |
| **B** struktura nieodwracalna vs attr wyciek | ✓ wl leak 0.077 vs attr 0.791 | ✓✓ **fgsd 0.046 vs attr 0.765** | ✓ fgsd 0.048, topo 0.32 vs attr 0.885 (wl 0.71 — wyjątek: seed=kolor) |
| **C** 2-osiowe pokrętło struktura⟂treść | — (świadomie nieobecny; PROTEINS reprezentatywny) | ✓✓ **krzywa feature-DP + siatka 2D + edge-DP** | ✓ edge-DP (krzywa); feature-DP demonstrowany na PROTEINS |
| **D** struktura > wygląd | — (brak osi wyglądu w TU) | — (brak wyglądu) | ✓✓ **wl 0.079 > rgb 0.071 > hog 0.008** + ablacja struktura vs losowość |

Legenda: ✓✓ pokazane najmocniej · ✓ pokazane · — świadomie nieobecne (z powodem).

## Negatywy / zastrzeżenia (krok 6) — pełna lista w [`FINDINGS.md`](FINDINGS.md)
ARI bezwzględnie niskie (zbiory trudne); `--label-rich` nie skaluje; komplementarny optimum
fuzji nieselektowalny label-free; `rand_ens` nie bije lidera; wizualny atak — niskie bezwzględne
SSIM (permutacyjna niezmienniczość ⇒ brak układu przestrzennego); feature-DP pod silnym atakiem
redukuje wyciek umiarkowanie (nie do zera); oracle LDA używa etykiet (górna granica, nie honest);
wl/CIFAR ARI wrażliwe na seed (±0.008).

Szkic pod referat: [`FINDINGS.md`](FINDINGS.md).

---

## Finalne leaderboardy (honest = selekcja label-free; oracle = z etykietami, osobno)

### PROTEINS_full (1113) — `results_FINAL_prot`
| metoda | km_ARI | km_NMI | probe_acc | recon_leak↓ |
|---|---|---|---|---|
| **fgsd** (histogram odl. biharmonicznych) | **0.135** (±0.001) | 0.086 | 0.691 | **0.046** (±0.002) |
| rand_ens (topo, M=10, dropedge0.5) | 0.089 | 0.072 | 0.724 | 0.082 |
| topo | 0.080 | 0.063 | 0.741 | 0.099 |
| wltopo (wl+topo) | 0.080 | 0.063 | 0.750 | 0.160 |
| netlsd | 0.074 | 0.056 | 0.733 | 0.058 |
| wl | 0.040 | 0.026 | 0.725 | 0.103 |
| attr (treść — odniesienie) | 0.037 | 0.026 | 0.748 | 0.765 (±0.006) |
| _oracle LDA(wl-kernel)→KMeans_ | _0.315_ | _0.227_ | — | — |

### CIFAR-10 (1500, grafy z obrazu) — `results_FINAL_cifar`
| metoda | km_ARI | km_NMI | probe_acc | recon_leak↓ |
|---|---|---|---|---|
| **wl + GMM-full** | **0.085** | 0.080 | 0.573 | 0.712 |
| wl + KMeans | 0.082 (±0.008) | 0.086 | 0.573 | 0.712 (±0.007) |
| g2v + KMeans | 0.082 | 0.081 | 0.587 | 0.651 |
| attr (treść — odniesienie) | 0.124 | 0.116 | 0.633 | 0.885 (±0.004) |
| rgb (wygląd — baseline) | 0.071 | 0.073 | 0.520 | 0.441 |
| topo | 0.045 | 0.050 | 0.491 | 0.323 |
| fgsd (słabe na obrazach) | 0.023 | 0.026 | 0.420 | 0.048 |
| hog (wygląd — baseline) | 0.008 | 0.009 | 0.718 | 0.264 |
| _oracle LDA(wl)→KMeans_ | _0.220_ | _0.194_ | — | — |

### ENZYMES (600) — sufit trudny — `results_FINAL_enz`
| metoda | km_ARI | km_NMI | probe_acc | recon_leak↓ |
|---|---|---|---|---|
| wl + KMeans | **0.046** | 0.086 | 0.403 | 0.077 |
| g2v | 0.034 | 0.060 | 0.332 | 0.099 |
| topo | 0.030 | 0.059 | 0.302 | 0.002 |
| attr (treść — odniesienie) | 0.032 | 0.056 | 0.490 | 0.791 |
| wl + UMAP→KMeans (oracle hp) | 0.052 | 0.095 | 0.403 | — |

## Główne wnioski

### (a) Separowalność ≠ klastrowalność
Na **wszystkich** zbiorach `probe_acc` jest wyraźnie powyżej szansy, a `km_ARI` niskie.
Bateria klastrowań (GMM-full, UMAP, PCA-whitening, spectral, kernel-KMeans na jądrze WL)
daje co najwyżej **marginalne** zyski ARI (CIFAR 0.082→0.085; ENZYMES 0.046→0.052; PROTEINS
0.135 bez zmian). **Diagnoza dowodowa — oracle LDA** (rzut na oś klasy, używa etykiet): ARI
skacze ×1.4–×8 (WL-kernel PROTEINS **0.040 → 0.315**; CIFAR wl 0.082 → 0.220). **Sygnał
klasy JEST w reprezentacji strukturalnej, ale nie jest dominującą osią wariancji** — dlatego
klastrowanie bez etykiet go nie wydobywa. Wąskim gardłem nie jest algorytm klastrowania ani
deskryptor, lecz geometria embeddingu (klasa = kierunek o małej wariancji).
*(Diagnoza η²: tylko PROTEINS/fgsd klastry częściowo dzielą po gęstości grafu, ηdens=0.41;
CIFAR/ENZYMES nie są napędzane rozmiarem.)*

**Ablacja struktura vs losowość** (graf losowy jako reprezentacja): dla `fgsd` na PROTEINS
prawdziwy graf daje ARI 0.135 vs ER o dopasowanej gęstości 0.065 i dropedge p=0.5 0.058 —
**prawdziwa topologia niesie ~2× sygnału losowej**. Na ENZYMES różnica jest mała (wl 0.046 vs
ER 0.037), bo `wl` jest seedowany etykietami węzłów (typ), które obfuskacja krawędzi zachowuje
→ jego słaby sygnał płynie głównie ze składu etykiet, nie z topologii.

### (b) Reprezentacje strukturalne są nieodwracalne (prywatne); `attr` wycieka
Pod **odpornym atakiem rekonstrukcyjnym** (RidgeCV+MLP, bierze silniejszy; `recon_leak∈[0,1]`)
strumień atrybutowy wycieka silnie (PROTEINS 0.764, CIFAR 0.885 — z treści da się odtworzyć
obiekt), a deskryptory czysto strukturalne są **prawie nieodwracalne**: PROTEINS fgsd 0.042,
topo 0.099, wl 0.103; ENZYMES wl 0.077. **`fgsd` na PROTEINS to ścisły zwycięzca** — najlepsze
ARI (0.136 ± 0.001) *i* najniższy wyciek (0.046 ± 0.002), oba stabilne po 5 seedach. *(Uwaga
metodyczna: słaby atak MLP dawał absurdalne ujemne R²=-8.58 na 200-wym FGSD — artefakt
przeuczenia; RidgeCV to naprawia.)*

**Wizualny atak inwersyjny (CIFAR).** Atak odtwarzający OBRAZ (regresor embedding_grafu →
obraz uśredniony po superpikselach SLIC) potwierdza to wizualnie: z `attr` da się odtworzyć
rozpoznawalną paletę/wygląd obiektu (SSIM 0.198, MSE 0.037 — najwyższa wierność), z `fgsd`
(czysta topologia) wychodzi szara breja (MSE 0.058, zero koloru), `wl` pośrednio (odtwarza
kolor, bo jego seed to etykiety koloru). Rosnący **feature-DP monotonicznie psuje** odtworzenie
z `attr` (SSIM 0.198→0.173, MSE 0.037→0.045), nie ruszając metod strukturalnych. Niuans:
embeddingi całografowe są permutacyjnie niezmiennicze → nie kodują układu przestrzennego, więc
rekonstrukcje to „plamy koloru" — różnicuje je wierność KOLORU, nie ostrość. Figura:
`results_inversion/fig_inversion.png`.

### (c) Dwuosiowe pokrętło prywatności: struktura ⟂ treść
Dwa niezależne, formalne mechanizmy obfuskacji:
- **edge-DP** (randomized response na krawędziach, parametr ε) — chroni **strukturę**;
- **feature-DP** (kalibrowany szum gaussowski na cechach węzłów) — chroni **treść**.

Siatka 2D (`results_FINAL_priv2d/fig_privacy_2d.png`, pełne dane, odporny atak) pokazuje to
wprost: `recon_leak` strumienia treści zależy **wyłącznie** od feature-DP (**0.76 → 0.49** przy
szumie 0→4×std) i jest **dokładnie niezmienny** względem edge-DP (std=0.000 wzdłuż osi krawędzi —
attr czyta tylko cechy węzłów, których edge-DP nie rusza). Edge-DP nie rusza osi treści; dopiero
feature-DP ją redukuje. Razem dają sterowalną osobno krzywą prywatność–użyteczność na dwóch
ortogonalnych osiach. *(Korekta audytu: pod ODPORNYM atakiem feature-DP redukuje wyciek
umiarkowanie do 0.49, nie do 0.23 jak sugerował słaby MLP — silny atakujący wyciąga więcej.)*

### (d) Struktura grafu dokłada sygnał ponad wygląd (CIFAR)
Na grafach z obrazu czysto strukturalne `wl` (ARI 0.082) **bije baseline'y wyglądu**:
`rgb` 0.071, `hog` 0.008. Reprezentacja grafowa niesie informację o klasie, której sam
deskryptor wyglądu (HOG/RGB-mean) nie chwyta w trybie nienadzorowanym — potwierdza sens
kodowania obrazu jako grafu obfuskującego.

## Figury (finalne, `results_FINAL_*`)
- `results_FINAL_prot/fig_benchmark.png` — ranking reprezentacji (PROTEINS, pełne dane).
- `results_FINAL_priv2d/fig_privacy_2d.png` — **2D pokrętło prywatności** (edge-DP ⟂ feature-DP).
- `results_FINAL_priv_feat/fig_privacy_utility.png` — krzywa feature-DP (attr maleje, struktura płaska).
- `results_inversion/fig_inversion.png` — **wizualny atak inwersyjny** (CIFAR): odtworzenie
  obrazu z embeddingu, attr vs wl vs fgsd × σ feature-DP.
- (pomocniczo) `results_FINAL_priv_edp/fig_privacy_utility.png` — krzywa edge-DP (redundantna
  wobec siatki 2D); `results_FINAL_cifar/fig_benchmark.png`.

## Wkład tej eksploracji (kod)
- **Nowe deskryptory/metody klastrowania:** `fgsd` (jądro biharmoniczne), fuzje strukturalne
  `wltopo`/`sfuse`; narzędzia `cluster_lab.py` (bateria + diagnoza η²), `closeA.py` (jądro WL
  + oracle LDA), `fusion_select.py` (label-free dobór wagi).
- **Prywatność:** `edge-DP` (`--obf-method edp`) i `feature-DP` (`--obf-method feature`);
  `privacy2d.py` (siatka 2D); odporny atak rekonstrukcyjny (`RidgeCV+MLP`, kolumny
  `recon_nmse`/`recon_leak`).
- **CLI:** `--n-segments`, `--compactness`.

## Negatywne / uczciwe ustalenia (czego NIE da się obronić)
- `--label-rich` na CIFAR **nie skaluje** (boost znika przy 1500 grafach — był efektem małej próbki).
- Komplementarny optimum fuzji struktura⊕treść jest realny (PROTEINS 0.155, CIFAR 0.130),
  ale **żadne kryterium label-free go nie wybiera** → nie liczony jako honest wynik.
- Lepsze klastrowanie/jądro WL **nie** odblokowuje dużych zysków ARI (patrz wniosek a).
