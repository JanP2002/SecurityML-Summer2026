# FINDINGS — szkic pod referat (graphrep)

> Baza treści do złożenia dokumentu. Bez LaTeX — same punkty. Liczby zweryfikowane na PEŁNYCH
> danych odpornym atakiem (RidgeCV+MLP), z kontrolą wrażliwości na seed. Priorytet narracji:
> **PRYWATNOŚĆ/bezpieczeństwo** (graf jako reprezentacja ukrywająca obiekt); klasteryzacja w tle.

## 0. Kontekst — co było wcześniej
- **Nasza ścieżka (grafy gotowe TU):** klasteryzacja reprezentacji grafowych enzymów (ENZYMES,
  6 klas) i białek (PROTEINS_full, 2 klasy) — deskryptory całografowe (topo, WL, graph2vec,
  spektralne) + sonda nadzorowana + atak rekonstrukcyjny jako metryka odwracalności.
- **Ścieżka kolegów (grafy z obrazu):** CIFAR-10 → graf SLIC (węzeł = superpiksel, krawędź =
  bliskość + podobieństwo koloru); pomysł grafów LOSOWYCH (ER/dropedge/small-world) jako
  „pokrętła" obfuskacji i jako reprezentacji (notatnik cifar_rand_graphs2).
- **Myśl przewodnia:** surowy obiekt (białko, obraz) kodujemy jako abstrakcyjny graf, który
  „ukrywa" oryginał. Pytamy: ile da się odzyskać (prywatność) i czy klasa zostaje wykrywalna
  bez etykiet (użyteczność).

## 1. Co nowego (wkład tej iteracji)
- **Nowy deskryptor strukturalny:** FGSD (histogram odległości biharmonicznych) — najlepsze
  ARI na PROTEINS i jednocześnie skrajnie nieodwracalny.
- **Dwa formalne mechanizmy prywatności:** edge-DP (randomized response na krawędziach, ε) i
  **feature-DP** (szum gaussowski na cechach węzłów) — dwie ortogonalne osie: struktura ⟂ treść.
- **Odporny atak rekonstrukcyjny:** RidgeCV+MLP (silniejszy), metryka `recon_leak ∈ [0,1]` —
  naprawia artefakt słabego MLP (R²=-8.58 na FGSD).
- **Wizualny atak inwersyjny** (odtworzenie OBRAZU, nie tylko średniej treści) z SSIM/MSE.
- **Diagnostyka klastrowania:** bateria (GMM/UMAP/whitening/jądro WL) + **oracle LDA** dowodzący,
  że klasa jest w embeddingu, lecz nie jest osią wariancji.
- **rand_ens:** ensembling losowych grafów jako reprezentacja (ablacja struktura vs losowość).

## 2. Cztery główne wnioski (teza + liczby + figura)

### Wniosek A — Separowalność ≠ klastrowalność (tło, ważne dla uczciwości)
- **Teza:** klasa jest LINIOWO SEPAROWALNA (sonda/oracle), ale NIE jest dominującą osią
  wariancji → klastrowanie bez etykiet wydobywa mało. Wąskie gardło to geometria embeddingu,
  nie algorytm.
- **Liczby:** probe_acc ≫ szansy przy niskim ARI na wszystkich zbiorach. Bateria klastrowań
  (GMM-full, UMAP, jądro WL) daje co najwyżej marginał (CIFAR 0.082→0.085). **ORACLE LDA**
  (rzut na oś klasy, używa etykiet): PROTEINS WL-kernel honest 0.040 → oracle **0.315** (×8);
  CIFAR wl honest 0.082 → oracle **0.220** (×2.6).
- **Figura:** tabela honest vs oracle (RESULTS).

### Wniosek B — Reprezentacje strukturalne są nieodwracalne (prywatne); `attr` wycieka
- **Teza:** z deskryptora czysto strukturalnego prawie nie da się odtworzyć treści; ze strumienia
  atrybutowego (treść/kolor) — tak.
- **Liczby (recon_leak↓, odporny atak):** PROTEINS fgsd **0.046 ± 0.002**, topo 0.099, wl 0.103
  vs attr **0.765 ± 0.006**. ENZYMES wl 0.077 vs attr 0.791. CIFAR fgsd 0.048, topo 0.323 vs
  attr **0.885 ± 0.004** (wl 0.71 — niesie kolor przez seed). **fgsd = ścisły zwycięzca na
  PROTEINS:** najlepsze ARI (0.136 ± 0.001) i najniższy wyciek (0.046).
- **Figura:** `results_FINAL_prot/fig_benchmark.png` + leaderboardy.

### Wniosek C — Dwuosiowe pokrętło prywatności: struktura ⟂ treść
- **Teza:** edge-DP chroni STRUKTURĘ, feature-DP chroni TREŚĆ; osie niezależne.
- **Liczby:** feature-DP (full PROTEINS, robust): attr recon_leak **0.764 → 0.485** monotonicznie
  (σ=0→4), struktura PŁASKA (topo 0.099, wl 0.103 — stałe). Siatka 2D: recon_leak[attr]
  **dokładnie niezmienny** wzdłuż osi edge-DP (std=0.000) i malejący wzdłuż feature-DP.
- **Figury:** `results_FINAL_priv_feat` (krzywa feature-DP), `results_FINAL_priv2d/fig_privacy_2d.png`
  (siatka 2D), `results_FINAL_priv_edp` (krzywa edge-DP).

### Wniosek D — Struktura grafu dokłada sygnał ponad wygląd (CIFAR)
- **Teza:** czysto strukturalne `wl` bije baseline'y wyglądu (HOG/RGB) w klastrowaniu bez etykiet.
- **Liczby:** wl ARI **0.079 ± 0.008** (>) rgb 0.071 (>) hog 0.008. Ablacja struktura vs
  losowość: prawdziwa topologia (fgsd PROTEINS 0.135) ≈ 2× losowej (ER 0.065, dropedge 0.058).
- **Figura:** leaderboard CIFAR + ablacja ABL1.

### (uzupełnienie prywatności) Wizualny atak inwersyjny
- Odtworzenie obrazu z embeddingu: attr SSIM **0.198**/MSE 0.037 (rozpoznawalna paleta) ≫ fgsd
  (szara breja, MSE 0.058); feature-DP psuje attr (SSIM 0.198→0.173). Niuans: embeddingi
  permutacyjnie niezmiennicze → rekonstrukcja to „plama koloru", różnicuje wierność KOLORU.
- **Figura:** `results_inversion/fig_inversion.png`.

## 3. Negatywy / zastrzeżenia (uczciwość)
- ARI bezwzględnie niskie (zbiory trudne); ENZYMES sufit ~0.046 (nie inwestujemy).
- `--label-rich` na CIFAR **nie skaluje** (boost z małej próbki znika przy 1500 grafach).
- Komplementarny optimum fuzji struktura⊕attr (PROTEINS 0.155, CIFAR 0.130) **nieselektowalny
  label-free** → nie liczony jako honest wynik.
- `rand_ens` bije bazowe topo na PROTEINS (0.089 vs 0.080), ale nie lidera i silhouette go
  nie preferuje; base=wl szkodzi.
- Wizualny atak: bezwzględne SSIM niskie (~0.2) — permutacyjna niezmienniczość zabrania
  rekonstrukcji układu przestrzennego; pokazujemy wierność koloru, nie kształtu.
- feature-DP pod ODPORNYM atakiem redukuje wyciek umiarkowanie (0.764→0.485), nie do zera —
  silny atakujący wyciąga więcej niż słaby MLP sugerował.
- Wrażliwość na seed: fgsd PROTEINS bardzo stabilny (±0.001); CIFAR wl umiarkowanie (±0.008,
  z podpróbki). recon_leak stabilny (±0.002–0.007).
- Oracle LDA używa ETYKIET (górna granica, nie wynik honest).
