# Przewodnik po projekcie (E2E) — Lista 3 (wersja szczegółowa)

Pełny, techniczny opis całego zadania: pytanie, podstawa teoretyczna, dokładne algorytmy
budowy grafów (z formułami i pseudokodem), embeddingi, ewaluacja, wyniki, **wizualizacje**
oraz sekcja **obrony** (przewidywane pytania prowadzącego + odpowiedzi). Dokument scala
[`RESULTS.md`](RESULTS.md), [`METODY.md`](METODY.md) i kod.

Oznaczenia: `|V|` — liczba węzłów, `|E|` — liczba krawędzi, `d` — odległość cech, `σ` — skala
jądra Gaussa, `w` — waga fuzji strumieni.

---

## 0. TL;DR

- **Pytanie:** czy o klasie obiektu decyduje **struktura jego grafu**, wykrywalna **bez etykiet**?
- **Zbiory:** CIFAR-10 (obraz→graf) i PROTEINS_full (białko = gotowy graf).
- **Pipeline:** obiekt → graf → embedding (wektor) → klasteryzacja nienadzorowana + sonda nadzorowana → metryki.
- **Dwie linie:** (a) nasz `.py` — graf deterministyczny (twardy próg) + bogaty zestaw embeddingów;
  (b) notatnik finalny — graf **losowy** (RAG + DropEdge + small-world) + **ensemble** + cechy topologiczne.
- **Wnioski:** struktura niesie sygnał o klasie; hybryda struktury z HOG bije sam HOG; nienadzorowanie graf > baseline'y.

---

## 1. Pytanie badawcze, hipoteza, podstawa teoretyczna

**Hipoteza:** dla obiektu, który da się sensownie zamienić w graf (obraz po segmentacji,
białko), **rozkład wzorców łączności** (podgrafy, komponenty, stopnie) zależy od klasy na
tyle, że klasy są separowalne w przestrzeni cech grafu — także bez etykiet.

**Dlaczego to ma prawo działać (teoria):**
- **Jądra grafowe i test Weisfeilera–Lehmana (WL).** Iteracyjne przeetykietowanie WL to
  klasyczny heurystyczny test izomorfizmu grafów; rozkład powstałych „poddrzew" (wzorców
  sąsiedztwa) jest silnym deskryptorem strukturalnym. WL feature map / graph2vec wprost z tego
  korzystają — dlatego są mocniejsze niż uśrednianie embeddingów węzłów.
- **Twardy warunek na krawędź** sprawia, że graf obrazu **fragmentuje się na regiony ≈ obiekty**.
  Liczba i rozmiary tych komponentów to graf-natywny sygnał o klasie (np. „zwarty obiekt na
  jednolitym tle" vs „scena rozdrobniona").
- **Analogia morfologiczna:** rzutując embeddingi do 2D (PCA) oglądamy „morfoprzestrzeń" —
  czy obiekty jednej klasy zajmują wspólny obszar.

**Rola etykiet:** żadna na etapie budowy reprezentacji. Etykiety wchodzą **tylko** w ocenie:
(a) nienadzorowanej — jak klastry KMeans pokrywają się z klasami (ARI/NMI), (b) sondzie
nadzorowanej — jak dobrze klasy są w ogóle separowalne (CV).

---

## 2. Przepływ danych E2E

```
        (1) BUDOWA GRAFU            (2) EMBEDDING               (3) METODY              (4) EWALUACJA
 obiekt ──────────────────────► ──────────────────────► ──────────────────────► ─────────────────────────
         węzeł + atrybuty;        wektor na obiekt:          fuzja strumieni         NIENADZOROWANA: KMeans
         krawędź = bliskość       Node2Vec / WL /            (struktura ⊕ atrybuty,  (k = #klas) → ARI, NMI,
         ORAZ podobieństwo        graph2vec / topo /         waga w) + baseline'y    silhouette
         (twardy próg lub         spektrum  ALBO             (RGB, HOG)              NADZOROWANA (sonda):
         losowanie)               ensemble cech topo                                 5-fold CV, LogReg/RF/SVM
```

Każdy krok ma osobną sekcję (§5–§8). Krok (4) nienadzorowany to bezpośrednia odpowiedź na pytanie.

---

## 3. Zbiory danych

| zbiór | obiekt | |V| typowo | węzeł | krawędź | klasy | źródło |
|-------|--------|-----------|-------|---------|-------|--------|
| **CIFAR-10** | obraz 32×32 RGB | 1024 (pixel) / 225 (patch) / ~60 (slic) | piksel / patch / superpiksel | budowana (patrz §5) | 10 | torchvision → `./data` |
| **PROTEINS_full** | białko | ~9–620 (śr. ~39) | aminokwas/SSE z etykietą 0/1/2 | gotowy kontakt | 2 (enzym/nie-enzym) | mirror GraphRNN → `data/` (format Dortmund) |

CIFAR: obrazy ładowane jako `float32` w `[0,1]` (`load_cifar10`); podzbiór klas i liczba na
klasę sterowane `--classes` / `--per-class`, ziarno stałe.

---

## 4. Mapa plików

| plik | rola |
|------|------|
| [`cifar_graph_clustering.py`](cifar_graph_clustering.py) | **nasz pipeline CIFAR**: twardy próg + Node2Vec/WL/graph2vec/topo/spec/combo/gnat/hybryda. |
| [`tu_graph_clustering.py`](tu_graph_clustering.py) | benchmark całografowy TUDataset (ENZYMES/PROTEINS), 5 reprezentacji. |
| `cifar6_3graphs.ipynb` | notatnik Jana: trzy konstrukcje (pixel/patch/slic), wagi miękkie. |
| `cifar_rand_graphs2_description.ipynb` | **finalny notatnik Jana**: graf LOSOWY (RAG+DropEdge+small-world+ensemble) + opisy. |
| [`wizualizacja_grafow.py`](wizualizacja_grafow.py) | generator 26 figur → `results_wiz/`. |
| [`raport_html.py`](raport_html.py) → [`raport.html`](raport.html) | samodzielny raport (figury osadzone). |
| [`RESULTS.md`](RESULTS.md) / [`METODY.md`](METODY.md) | wyniki+historia / katalog metod. |
| `results_cifar/`, `results_wiz/` | wyniki benchmarku / figury wizualizacji. |

---

## 5. KROK 1 — budowa grafu (szczegółowo)

### 5.1 Zasada krawędzi (formalnie)
Dla pary węzłów `(u,v)` krawędź istnieje wtedy i tylko wtedy, gdy **JEDNOCZEŚNIE**:
1. `(u,v)` są **sąsiadami przestrzennymi** (zależnie od konstrukcji: sąsiedztwo 8-spójne na
   siatce pikseli; wspólna granica superpikseli; okno przestrzenne dla patchy), ORAZ
2. **odległość cech** `d(u,v) ≤ τ` (podobny kolor/tekstura).

Krawędzie ponad próg są **przecinane** (nie tylko osłabiane wagą). To różni nasze podejście od
pierwotnego notatnika, gdzie warunek (2) był tylko miękką wagą i graf był prawie pełną siatką
dla każdego obrazu (Node2Vec kodował wtedy siatkę, nie treść).

### 5.2 Próg adaptacyjny `τ`
`τ` jest dobierany **per obraz** jako kwantyl rozkładu odległości krawędzi-kandydatów:
```
τ = quantile({d(u,v) : (u,v) sąsiedzi}, q),   q = edge_quantile (domyślnie 0.6)
```
Czyli zostaje ~60 % najbardziej „wewnątrzobiektowych" krawędzi. Alternatywnie stały `--tau`.
**Dlaczego kwantyl, nie stała:** obrazy mają różny kontrast; stały próg dałby gęsty graf dla
jednych i pusty dla innych. Kwantyl normalizuje gęstość między obrazami.

### 5.3 Graf pikselowy (`build_pixel_graph`)
- Węzeł = piksel w przestrzeni **LAB**; cechy `[L, a, b, |∇L|]` (kolor + magnituda gradientu jasności = tekstura).
- Sąsiedztwo 8-spójne; `d` = odległość euklidesowa w LAB.
- Krawędź gdy `d ≤ τ`; waga `exp(−d²/2σ_pixel²)`, `σ_pixel = 15` (skala dla L∈[0,100]).
- `|V| = 1024`. Koszt ~2.4 s/obraz (stąd do iteracji wolimy SLIC).

### 5.4 Graf patchy (`build_patch_graph`)
- Węzeł = nakładający się patch 4×4 (stride 2), cechy = `mean⊕std` koloru. `|V| ≈ 225`.
- Krawędzie: siatka 4-sąsiedzka **+** kNN (k=4) **ograniczone przestrzennie** oknem `knn_radius=3`
  — kNN bez ograniczenia łamałoby warunek bliskości. Obie rodziny progowane `τ`.

### 5.5 Graf SLIC / superpikselowy (`build_slic_graph`) — kluczowy
1. **Segmentacja** SLIC: `n_segments=60`, `compactness=10` → superpiksele (węzły).
2. **Cechy węzła** = `[mean_RGB(3), std_RGB(3)]` (+ mini-HOG przy `--rich-features`); `pos` = środek ciężkości.
3. **Sąsiedztwo (RAG)** z detekcji granic: skanujemy mapę etykiet; dla każdej pary
   sąsiadujących pikseli o różnych etykietach inkrementujemy licznik **długości wspólnej
   granicy** `blen(u,v)`. Para jest kandydatem na krawędź **tylko gdy graniczy** (`blen>0`).
4. **Odległość** `d(u,v) = ‖mean_RGB(u) − mean_RGB(v)‖`.
5. **Waga** krawędzi:
   ```
   color_w = exp(−d²/2σ_feat²),   σ_feat = 1.0
   edge_w  = (blen / max_b) · color_w        # człon przestrzenny × kolor
   ```
6. **Tryb deterministyczny (.py):** krawędź gdy `d ≤ τ`.
   **Tryb losowy (notatnik):** krawędź z prawdopodobieństwem `color_w` (`rand() < color_w`) — DropEdge.

### 5.6 Reperacja izolacji (`_ensure_connected`)
Każdy węzeł, który po progowaniu/losowaniu został izolowany, podłączamy do jego **najbardziej
podobnego sąsiada przestrzennego** (warunek bliskości zachowany). **Dlaczego:** Node2Vec
potrzebuje spójności lokalnej, żeby spacer miał gdzie iść; izolowany węzeł nie wniósłby nic.

### 5.7 Składniki losowości (finalny notatnik)
- **DropEdge** (probabilistyczne krawędzie): patrz §5.5 pkt 6. **Uzasadnienie:** uniezależnia
  graf od jednego twardego progu; uśredniony po zespole działa jak stochastyczna regularyzacja
  (analogicznie do DropEdge w sieciach grafowych).
- **Small-world** (`_apply_small_world`), pseudokod:
  ```
  N = floor(|V| · small_world_p)         # small_world_p = 0.05
  σ = σ_pixel if |V|>400 else σ_feat
  powtórz N razy:
      wylosuj parę (u,v); jeśli brak krawędzi:
          d = ‖feat(u)−feat(v)‖; jeśli rand() < exp(−d²/2σ²): dodaj krawędź
  ```
  **Dlaczego:** kilka dalekich skrótów skraca średnie ścieżki (efekt „małego świata"), dzięki
  czemu spacery Node2Vec sięgają kontekstu globalnego, a topologia nie jest czysto lokalna.
- **Ensemble** (`_build_ensemble_for_image`): budujemy `M = ensemble_m = 5` losowych grafów na
  obraz, liczymy z każdego cechy topo (§6.5) i zwracamy **średnią ⊕ odchylenie** → wektor 26-wym.
  **Dlaczego:** pojedyncze losowanie jest zaszumione; średnia po M grafach to stabilny deskryptor,
  a odchylenie koduje „jak bardzo struktura jest wrażliwa na losowanie".

### 5.8 Parametry i ich dobór

| parametr | wartość | rola / dlaczego tak |
|----------|---------|---------------------|
| `edge_quantile` | 0.6 | zostaw ~60 % wewnątrzobiektowych krawędzi; balans gęstość/fragmentacja |
| `n_segments` | 60 | kompromis: dość węzłów na bogaty graf, dość mało na szybkość/czytelność |
| `compactness` | 10 | regularność superpikseli (niższa → lepiej trzyma granice koloru) |
| `σ_feat` / `σ_pixel` | 1.0 / 15 | skala jądra wagi w przestrzeni cech / LAB |
| `small_world_p` | 0.05 | ~5 % węzłów jako dalekie skróty (mało, by nie zaszumić) |
| `ensemble_m` | 5 | uśrednianie cech losowych grafów |
| Node2Vec `dim/walk/num/p/q/window` | 32/30/20/1.0/0.5/5 | `q<1` faworyzuje eksplorację w głąb / oddalanie się (DFS-podobnie) |
| `wl_iterations` | 2 | głębokość wzorców WL (2-hop) |
| `g2v_dim` / `g2v_epochs` | 64 / 60 | wymiar i epoki graph2vec/WL-SVD |
| `n_color_bins` | 16 | liczba dyskretnych etykiet koloru (seed WL) |

---

## 6. KROK 2 — embedding (jeden wektor na obiekt)

### 6.1 Node2Vec (`node2vec_embeddings`)
Spacery losowe sterowane `p` (powrót) i `q` (eksploracja) generują „zdania" węzłów; model
**skip-gram (Word2Vec)** uczy osadzeń tak, by węzły bliskie w spacerach miały bliskie wektory.
Parametry: `dim=32, walk_length=30, num_walks=20, p=1.0, q=0.5, window=5, workers=1`
(równoległość `node2vec` zepsuta na Windows). Wynik: macierz `|V|×32` (embedding na węzeł).

### 6.2 Pooling węzeł→obiekt (`_pool`)
- **mean** (patch): zwykła średnia.
- **weighted_mean** (slic): średnia ważona **rozmiarem superpiksela** (większy region waży więcej).
- **spatial_quadrants** (pixel): obraz dzielony na 4 ćwiartki wg środka węzła, średnia w każdej,
  konkatenacja → zachowuje **zgrubny układ przestrzenny**, który zwykła średnia by zgubiła.

### 6.3 WL feature map (`embed_wl`, `_wl_relabel_docs`)
Algorytm (iteracje = `wl_iterations=2`):
```
etykieta_0(v) = label(v)            # dyskretny kolor (KMeans, §6.8) albo stopień
dla it = 1..L:
    wzorzec(v) = etykieta(v) + "|" + posortowane(etykiety sąsiadów)
    etykieta(v) = nowe_id(wzorzec(v))      # haszowanie wzorca na liczbę
    zlicz wystąpienia wzorca w grafie
```
Histogram wzorców (rzadki) → **TruncatedSVD** do `g2v_dim=64`. To jest jądro WL — rozkład
poddrzew sąsiedztwa, mocny deskryptor strukturalny.

### 6.4 graph2vec (`embed_graph2vec`)
Te same „dokumenty" wzorców WL, ale **uczone gęsto** przez **Doc2Vec (DBOW, `dm=0`)**,
`vector_size=64, epochs=60`. Uczy się ciągłej reprezentacji całego grafu z jego wzorców.

### 6.5 Cechy topologiczne (`graph_topo_features`) — 13 wymiarów
Dokładny wektor (graf-natywny, **bez koloru**):
```
[ |V|, |E|, |E|/|V|, gęstość,
  #komponentów, śr.rozmiar_komp, std_rozmiar_komp, max_rozmiar_komp, max_rozmiar/|V|,   # fragmentacja
  śr.stopień, std_stopień, śr.klasteryzacja, asortatywność ]
```
W notatniku losowym ensemble zwraca **średnią⊕odchylenie** tych 13 cech po `M=5` grafach = **26 wym.**
To bezpośrednio koduje skutek twardego warunku na krawędź (rozpad na regiony ≈ obiekty).

### 6.6 Sygnatura spektralna (`spectral_features`) — k=16
`k` najmniejszych wartości własnych **znormalizowanego Laplasjanu** `L = I − D^{-1/2} A D^{-1/2}`.
Interpretacja: liczba wartości ≈ 0 ≈ liczba komponentów; kolejne kodują „kształt/łączność".
Komplementarne do `topo` (próba #2 — w praktyce nie poprawiła wyniku).

### 6.7 Problem rozcieńczenia + fuzja (`fuse`) — kluczowe
**Problem:** płaska konkatenacja Node2Vec (`~128` wym.) ⊕ koloru (`~12` wym.) i **wspólna**
standaryzacja → wariancja zdominowana przez 128 wymiarów struktury, kolor „utopiony".
**Rozwiązanie:** każdy blok osobno standaryzowany i **L2-normalizowany**, potem ważony:
```
Z_s = normalize(standardize(X_struktura))
Z_a = normalize(standardize(X_atrybuty))
fuse = [ w · Z_s ,  (1−w) · Z_a ]          # w przemiatane ∈ {0, .25, .5, .75, 1}
```
**Dlaczego L2 na blok:** po normalizacji każdy blok wnosi **wektor jednostkowy** niezależnie od
liczby wymiarów → o wkładzie decyduje waga `w`, nie przypadkowa liczba wymiarów. `w=0` = tylko
atrybuty, `w=1` = tylko struktura.

### 6.8 Seed etykiet WL (`assign_color_labels`)
Globalny **KMeans** (`n_color_bins=16`) na cechach węzłów całego korpusu → dyskretna etykieta
`label` (wspólny słownik). Domyślnie po samym **kolorze** (3 cechy); `--label-rich` → po całym
(zestandaryzowanym) deskryptorze (kolor+tekstura+kształt). Próba #3: bogatszy seed **pomógł**
(g2v 0.62→0.66, wl 0.58→0.63) — wzmocnienie reprezentacji strukturalnej > zmiana topologii.

---

## 7. KROK 3 — metody w benchmarku

Nic nie usuwamy; każda metoda to osobny wiersz (widać postęp).

| metoda | skład | rola |
|--------|-------|------|
| `n2vflat` | Node2Vec ⊕ atrybuty, **płasko** (wspólna standaryzacja) | baseline „przed poprawką" (rozcieńczenie) |
| `n2v` | Node2Vec ⊕ atrybuty, **rozdzielone + waga w** | poprawiony Node2Vec |
| `wl` / `g2v` | WL (SVD) / graph2vec | całografowe |
| `topo` / `spec` | 13 cech strukturalnych / spektrum | czysto strukturalne (bez koloru) |
| `combo` | graph2vec ⊕ atrybuty, waga `w` | czysto grafowy |
| `gnat` | (graph2vec ⊕ topo) ⊕ atrybuty | **graf-natywny max (bez HOG)** |
| `gspec` | (graph2vec ⊕ topo ⊕ spec) ⊕ atrybuty | graf-natywny + spektrum |
| `hyb` | **HOG** ⊕ graph2vec, waga `w` | hybryda wygląd+struktura |
| `+r` (sufiks) | jw. z bogatym deskryptorem węzła (mini-HOG) | mocniejszy wariant |
| `baseline-rgb` / `baseline-hog` | średni RGB (3) / HOG | punkty odniesienia (bez grafu) |

Baseline HOG: `orientations=8, pixels_per_cell=8×8, cells_per_block=2×2`. Każda metoda występuje
w wariantach `-pixel/-patch/-slic` (i `*tex`).

---

## 8. KROK 4 — ewaluacja

### 8.1 Nadzorowana (sonda, `evaluate_classifiers`)
**StratifiedKFold** (`folds = min(5, najmniejsza klasa)`, shuffle, stałe ziarno); klasyfikatory:
**LogReg** (`max_iter=1000`), **RandomForest** (200 drzew), **SVM (RBF)**. Metryki: **dokładność**
i **F1-macro** (średnia z foldów). Rola: górne ograniczenie separowalności klas w danej reprezentacji.

### 8.2 Nienadzorowana (`evaluate_unsupervised`) — właściwe pytanie
**KMeans** z `k = liczba klas` (`n_init=10`). Metryki:
- **ARI** (Adjusted Rand Index): zgodność klastrów z klasami, **skorygowana o przypadek**;
  0 ≈ losowo, 1 = idealnie. Kluczowa, bo nie wymaga przypisania klaster→klasa.
- **NMI** (znormalizowana informacja wzajemna) ∈ [0,1].
- **silhouette** ∈ [−1,1]: spójność vs separacja klastrów (na samej geometrii, bez etykiet).

**Dlaczego `k = #klas`:** stawiamy klasteryzację w najuczciwszej sytuacji (tyle klastrów co klas),
żeby ARI mierzył pokrycie 1:1.

---

## 9. Wyniki (3 klasy: samolot/auto/statek, SLIC, 900 obrazów; próg losowy 0.333)

| metoda | dokładność (CV) | ARI | HOG? |
|--------|:---------------:|:---:|:----:|
| `hyb-slic` (w=0.5) | **0.786** | 0.025 | tak |
| `baseline-hog` | 0.759 | 0.023 | tak |
| `gnat-slic` | 0.613 | 0.110 | nie |
| `combo-slic` | 0.612 | **0.116** | nie |
| `topo-slic` | 0.543 | 0.085 | nie |
| `baseline-rgb` | 0.511 | 0.064 | — |
| `spec-slic` | 0.462 | 0.044 | nie |

- **Hybryda > HOG:** struktura niesie sygnał spoza wyglądu.
- **Nienadzorowanie graf ≫ baseline'y** (ARI ~0.11–0.14 vs ~0.02).
- **Sama `topo`** (bez koloru) > RGB.
- Pełne 10 klas: `hyb` ~0.475 > HOG 0.44; nikt blisko 0.7 (oczekiwane).

---

## 10. Wizualizacje (jak są zrobione)

Generuje [`wizualizacja_grafow.py`](wizualizacja_grafow.py); sercem jest `draw_overlay`, która
**nakłada graf na obraz**.

### 10.1 Współrzędne węzłów (`pos`)
`pos` = położenie na obrazie: pixel/patch `(wiersz, kolumna)`; slic = **środek ciężkości** superpiksela.

### 10.2 Nakładanie grafu na obraz — JAK i DLACZEGO
1. Tło: `ax.imshow(image)`.
2. **Zamiana `(wiersz, kolumna) → (x=kolumna, y=wiersz)`** (`_posxy`), bo `imshow` ma `x=kolumna`,
   `y=wiersz` rosnące **w dół**. Bez zamiany graf byłby obrócony o 90°; po zamianie **węzeł leży
   dokładnie na regionie, który reprezentuje**.
3. Krawędzie: odcinki między pozycjami końców (hurtowo `LineCollection`); **przezroczystość i
   grubość ∝ waga** krawędzi.
4. Węzły: `scatter` w tych samych współrzędnych; w SLIC **rozmiar markera ∝ rozmiar superpiksela**.
5. Zakres osi zablokowany do wymiarów obrazu (idealne pokrycie).

**Dlaczego nakładka:** pokazuje, że graf jest **zakotwiczony w obrazie** (krawędź = bliskość +
podobieństwo) i jak rozpada się na obiekt/tło. Białka nie mają obrazu → układ **Kamada–Kawai**.

### 10.3 Kolory i grubości
Węzły żółte (rozmiar ∝ superpiksel), krawędzie cyjan (jasność/grubość ∝ waga), mapy malowane
(komponenty / stopień) w fig_06/12/17.

### 10.4 CZERWONE krawędzie
- **Domyślnie:** **skróty small-world** (`_apply_small_world`) — losowe dalekie krawędzie rysowane
  na wierzchu kolorem `#ff2d2d`. Występują na figurach z `small_world=True`: fig_01 (ostatni panel),
  02, 04, 07, 08, 09, 10 (lewy), 12, 15, 16 oraz kolumna `slic+SW` w fig_23.
- **fig_05:** czerwień (kropkowana) = krawędzie-kandydaci **ODRZUCONE** przez losowanie (zielone = przyjęte).
- **fig_13:** czerwone **strzałki** = ścieżka spaceru losowego Node2Vec.

### 10.5 Indeks figur
Pełny: [`results_wiz/README.md`](results_wiz/README.md).

---

## 11. Złożoność i wydajność

- **Budowa grafu:** SLIC ~ms/obraz; **pixel ~2.4 s/obraz** (1024 węzły) — stąd do iteracji SLIC.
- **Node2Vec/graph2vec** liczone raz na konfigurację i **cache'owane** jako `.npy` (klucz = MD5
  hash konfiguracji) → przemiatanie wagi `w` jest natychmiastowe.
- **Równoległość** po obrazach (`joblib`); Node2Vec/Doc2Vec na `workers=1` (Windows).
- **Spektrum:** `eigvalsh` gęsto dla `|V|≤400`, inaczej `eigsh` (rzadkie, k najmniejszych).

---

## 12. Ograniczenia i uczciwe zastrzeżenia

- **Mała próbka → szum ±0.02:** drobnych różnic między wariantami graf-natywnymi nie
  nadinterpretowujemy; różne `--per-class` losują różne podzbiory, więc absolutne liczby się ruszają.
- **>0.7 tylko na kilku klasach;** pełne 10 klas jest trudne (nawet HOG ~0.45).
- **Czysto grafowa ścieżka nasyca się ~0.61–0.68;** przebicie HOG wymaga hybrydy z HOG.
- **W [0,1] (CIFAR) odległości koloru są małe** → w trybie probabilistycznym większość krawędzi
  sąsiadów przeżywa (widoczna losowość siedzi głównie w skrótach).
- **HOG to wygląd, nie struktura** — hybryda miesza dwa źródła; „czysto grafowy" wynik to `gnat`/`combo`.

---

## 13. Obrona — przewidywane pytania i odpowiedzi

**P: Jak gwarantujecie, że łączone węzły to bliskie fragmenty obrazu (wersja losowa)?**
Bramką sąsiedztwa (RAG): kandydaci to **tylko** superpiksele o wspólnej granicy (skan mapy etykiet
SLIC). Losowanie (`rand<color_w`) działa już tylko na sąsiadach i odrzuca pary o niepodobnym
kolorze; nigdy nie łączy odległych regionów. Długość granicy dodatkowo waży krawędź.

**P: Co dokładnie wchodzi do prawdopodobieństwa połączenia?**
Samo podobieństwo koloru `color_w = exp(−d²/2σ²)`. Człon przestrzenny realizują **bramka
sąsiedztwa** i **waga** `blen/max_b`, a nie samo losowanie.

**P: Dlaczego twardy próg, a nie miękka waga jak w pierwotnym notatniku?**
Miękka waga zostawiała prawie pełną siatkę dla każdego obrazu — Node2Vec kodował siatkę, nie treść.
Twardy próg/odcięcie sprawia, że topologia zależy od obrazu (fragmentacja na regiony).

**P: Czemu Node2Vec wypadał słabo i jak to naprawiliście?**
Uśrednianie embeddingów węzłów to słaba reprezentacja całego grafu; dodatkowo płaska konkatenacja
**rozcieńczała** kolor. Naprawa: rozdzielone strumienie + L2 na blok + waga `w` (§6.7) oraz
embeddingi **całografowe** (WL/graph2vec).

**P: Po co DropEdge / small-world / ensemble?**
DropEdge = stochastyczna regularyzacja (niezależność od jednego progu); small-world = skrócenie
ścieżek dla kontekstu globalnego; ensemble = stabilizacja cech (średnia+odchylenie po M grafach).

**P: Czy to nie HOG „robi robotę"?**
Najlepszy wynik (`hyb`) używa HOG, ale **`hyb` > sam HOG** → struktura dokłada sygnał. Czysto
grafowe `gnat`/`combo` ~0.61 i biją RGB; `topo` (bez koloru) też bije RGB. Nienadzorowanie graf ≫ HOG.

**P: Dlaczego ARI, nie sama dokładność klasteryzacji?**
ARI koryguje o przypadek i nie wymaga przypisania klaster→klasa; uczciwie mierzy pokrycie.

**P: Czy wyniki są powtarzalne?**
Stałe ziarna, cache embeddingów, stratyfikowana CV. Zastrzeżenie: przy małych próbkach ±0.02 szumu.

**P: Czym różni się notatnik (Jan) od skryptu `.py`?**
Notatnik: graf **losowy** + ensemble + cechy **topo** (26 wym.). `.py`: graf **deterministyczny** +
bogaty zestaw embeddingów (Node2Vec/WL/graph2vec/topo/spec) + hybryda z HOG. Wspólny pipeline,
różny krok (1) i (2). Oba trzymamy, by porównywać.

---

## 14. Jak odtworzyć

```bash
python cifar_graph_clustering.py --graph-type slic --classes 0 1 8 --per-class 200 --plots
python tu_graph_clustering.py --name PROTEINS_full --embedding graph2vec
python wizualizacja_grafow.py        # 26 figur -> results_wiz/
python raport_html.py                # raport.html
```

---

## 15. Słowniczek

- **RAG** — graf sąsiedztwa regionów (krawędź tylko między graniczącymi superpikselami).
- **DropEdge** — losowe zostawianie krawędzi z prawd. zależnym od koloru (regularyzacja).
- **Small-world** — kilka dalekich skrótów skracających ścieżki.
- **Ensemble** — uśrednianie cech po M losowych grafach jednego obrazu.
- **WL (Weisfeiler–Lehman)** — iteracyjne przeetykietowanie sąsiedztw; podstawa jąder grafowych i graph2vec.
- **Rozcieńczenie** — utopienie koloru przez ~128 wym. struktury przy płaskiej konkatenacji; naprawione przez `fuse`.
- **ARI / NMI / silhouette** — metryki klasteryzacji (zgodność z klasami / spójność klastrów).
- **Morfoprzestrzeń** — rzut 2D (PCA) embeddingów do oglądania separacji klas.
