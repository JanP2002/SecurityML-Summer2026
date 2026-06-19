# Lista 3 — katalog wszystkich użytych podejść

Pełny spis metod i technik zastosowanych w [`cifar_graph_clustering.py`](cifar_graph_clustering.py)
(wyniki i historia: [`RESULTS.md`](RESULTS.md)). Każdy obraz przechodzi ścieżkę:
**obraz → graf → embedding → pooling → fuzja → ewaluacja**. Poniżej każdy etap z
wariantami, które przetestowaliśmy.

---

## 1. Konstrukcja grafu (`GRAPH_BUILDERS`)

Wspólna zasada (wskazówka Krzysztofa): **krawędź powstaje tylko gdy spełnione są
JEDNOCZEŚNIE dwa warunki** — bliskość przestrzenna ORAZ podobieństwo koloru/cech
poniżej progu. Odległości powyżej progu są *przecinane* (nie tylko osłabiane wagą),
więc graf rozpada się na regiony ~obiekty.

| metoda | węzeł | krawędzie | wagi |
|--------|-------|-----------|------|
| **pixel** (`build_pixel_graph`) | piksel (przestrzeń LAB) | sąsiedztwo 4- lub 8-kierunkowe, **odcięte progiem** odległości koloru LAB | Gauss z odległości LAB |
| **patch** (`build_patch_graph`) | nakładający się patch (mean⊕std) | siatka 4-sąsiedzka **+** kNN **ograniczone przestrzennie** (okno `knn_radius`), obie rodziny progowane | Gauss z odległości cech |
| **slic** (`build_slic_graph`) | superpiksel SLIC (mean⊕std) | sąsiedztwo superpikseli z detekcji granic, odcięte progiem różnicy średniego koloru | (długość granicy / max) × podobieństwo koloru |

**Próg krawędzi** (`_threshold`): adaptacyjny **kwantyl** rozkładu odległości per obraz
(`--edge-quantile`, domyślnie 0.6 — zostaje ~60% najbardziej „wewnątrzobiektowych"
krawędzi) ALBO stały `--tau`.

**Reperacja izolacji** (`_ensure_connected`): każdy węzeł bez krawędzi po progowaniu
podłączamy do jego *najbardziej podobnego* sąsiada przestrzennego — żeby Node2Vec
zawsze miał gdzie chodzić (warunek bliskości zachowany).

## 2. Deskryptory węzła (cechy)

- **Pixel:** `[L, a, b, magnituda_gradientu]` — kolor LAB + tekstura (gradient kanału L).
- **Patch / SLIC (podstawowe):** `[średnia_koloru(3), odchylenie_std(3)]`.
- **SLIC bogaty** (`--rich-features`): do podstawowych dochodzi **mini-HOG na superpiksel** —
  histogram orientacji gradientu ważony magnitudą (`n_orient_bins`), średnia+std magnitudy,
  znormalizowany rozmiar i znormalizowany środek ciężkości. Graf-natywny odpowiednik HOG.
- **Kwantyzacja koloru na etykiety WL** (`assign_color_labels`): globalny KMeans na kolorach
  wszystkich węzłów → dyskretna etykieta `label` (wspólny słownik dla całego korpusu),
  używana jako inicjalizacja dla WL / graph2vec.

## 3. Embeddingi (jeden wektor na obraz)

| embedding | poziom | opis |
|-----------|--------|------|
| **Node2Vec + pooling** | węzeł→agregacja | spacery losowe (p, q) → Word2Vec na węzłach, potem pooling do wektora obrazu |
| **WL** (`embed_wl`) | całografowy | feature map jądra Weisfeiler–Lehman (zliczanie wzorców) → `TruncatedSVD` do gęstej postaci |
| **graph2vec** (`embed_graph2vec`) | całografowy | te same wzorce WL, ale **uczone** gęsto przez Doc2Vec |
| **topo** (`graph_topo_features`) | całografowy | czyste cechy STRUKTURALNE (bez koloru): liczba i rozkład rozmiarów komponentów po przecięciu krawędzi, stopnie, klasteryzacja, asortatywność |
| **pooling atrybutów** | węzeł→agregacja | pooling samych cech węzła (kolor/tekstura) — „strumień atrybutowy" |

`topo` bezpośrednio koduje skutek warunku Krzysztofa: po przecięciu krawędzi na granicach
obiektów obraz rozpada się na komponenty, a ich **liczba i rozmiary** są graf-natywnym
sygnałem o klasie (sama `topo`, bez koloru, bije już baseline RGB).

Całografowe (WL, graph2vec) dodano, bo Node2Vec+uśrednianie to znana **słaba**
reprezentacja całego grafu (potwierdza to ENZYMES w `tu_graph_clustering.py`).

## 4. Strategie poolingu (`_pool`)

- **mean** — zwykła średnia po węzłach (patch).
- **weighted_mean** — średnia ważona rozmiarem superpiksela (slic): większe regiony liczą się mocniej.
- **spatial_quadrants** — obraz dzielony na 4 ćwiartki, średnia w każdej, konkatenacja
  (pixel) — zachowuje zgrubny układ przestrzenny, który zwykła średnia by zgubiła.

## 5. Fuzja strumieni (`fuse`) — kluczowe odkrycie

Problem **rozcieńczenia**: płaska konkatenacja Node2Vec (~128 wym.) + koloru (~12 wym.)
i wspólna standaryzacja → struktura topi kolor.

Rozwiązanie: każdy blok osobno **standaryzowany + L2-normalizowany** (wnosi wektor
jednostkowy niezależnie od liczby wymiarów), potem mieszany wagą `w`:
`w=0` → tylko atrybuty, `w=1` → tylko struktura. Wagę **przemiatamy**
(`--weights`, domyślnie 0 / 0.25 / 0.5 / 0.75 / 1.0).

## 6. Metody w tabeli porównawczej (wiersze benchmarku)

Nic nie usuwamy — każde podejście to osobny wiersz, żeby było widać postęp.

| nazwa | skład | rola |
|-------|-------|------|
| `n2vflat-<gt>` | płaska konkatenacja Node2Vec+atrybuty (wspólna standaryzacja) | **baseline „przed poprawką"** (rozcieńczenie) |
| `n2v-<gt>` | Node2Vec ⊕ atrybuty, rozdzielone + waga `w` | poprawiony Node2Vec |
| `wl-<gt>` | WL (SVD) | całografowy, liczony |
| `g2v-<gt>` | graph2vec | całografowy, uczony |
| `topo-<gt>` | cechy strukturalne grafu | **czysto strukturalny** (bez koloru) |
| `combo-<gt>` | graph2vec ⊕ atrybuty, waga `w` | **czysto grafowy** (struktura+kolor) |
| `combo+r-<gt>` | jak combo, ale z bogatym deskryptorem (mini-HOG) | czysto grafowy, mocniejszy |
| `gnat-<gt>` / `gnat+r-<gt>` | (graph2vec ⊕ topo) ⊕ atrybuty, waga `w` | **graf-natywny max** (bez HOG) |
| `hyb-<gt>` | HOG ⊕ graph2vec, waga `w` | **hybryda** wygląd+struktura |
| `hyb+r-<gt>` | HOG ⊕ graph2vec(bogaty) | hybryda, najlepsza |
| `baseline-rgb` | średnie RGB (3 wym.) | baseline wyglądu |
| `baseline-hog` | HOG | mocny baseline wyglądu |

(`<gt>` = `pixel` / `patch` / `slic`.)

## 7. Ewaluacja

- **Nadzorowana** (`evaluate_classifiers`): 5-krotna walidacja krzyżowa (stratyfikowana),
  3 klasyfikatory — **Regresja logistyczna**, **Random Forest** (200 drzew),
  **SVM (RBF)**; metryki: dokładność i F1-macro.
- **Nienadzorowana** (`evaluate_unsupervised`): **KMeans** (k = liczba klas), metryki
  **ARI**, **NMI**, **silhouette** (to jest właściwe pytanie Listy 3 — wykrycie klasy bez etykiet).
- **Przemiatanie wagi** `w` dla metod z fuzją; skalowanie zależne od metody
  (fuzja: bez ponownej standaryzacji; baseline'y/WL/g2v: `StandardScaler`).
- **Dobór parametrów** (`--wl-iter`, `--g2v-dim`, `--n-orient-bins`, `--per-class`).

## 8. Inżynieria / odtwarzalność

- **Cache embeddingów** (`.npy` pod `<out-dir>/cache/`, klucz = hash konfiguracji):
  Node2Vec/graph2vec liczone RAZ, przemiatanie wag potem natychmiast.
- **Równoległość** przez `joblib` po obrazach; Node2Vec/Doc2Vec na `workers=1`
  (równoległość `node2vec` jest zepsuta na Windows).
- **UTF-8 na stdout** — żeby polskie znaki nie wywalały konsoli cp1252.
- **Wykresy** (`--plots`): `fig_postep` (historia usprawnień), `fig_porownanie`
  (ranking metod), `fig_waga` (wpływ wagi struktury), `fig_acc_vs_ari`
  (separowalność vs klasteryzowalność), `fig_morfo` (morfoprzestrzeń PCA 2D).
- **Dobór danych:** `--classes` (podzbiór klas), `--per-class` / `--num-samples`, ziarno losowości.

## 9. Baseline'y (bez grafu, punkt odniesienia)

- **RGB-mean** — średni kolor obrazu (3 wymiary).
- **HOG** — histogram zorientowanych gradientów (orientations=8, komórki 8×8, bloki 2×2);
  mocny klasyczny deskryptor wyglądu, główny rywal do pobicia.

## 10. Podejścia odziedziczone z notatnika (`cifar6_3graphs.ipynb`)

Stąd wyrósł skrypt; przeniesione i rozwinięte: trzy konstrukcje grafu (pixel/patch/slic),
przestrzeń LAB, nakładające się patche + kNN, pooling `spatial_quadrants` i `weighted_mean`,
fuzja cech (kolor doklejany do embeddingu) oraz baseline'y RGB/HOG. Notatnika nie
edytujemy — zmiany idą do `cifar_graph_clustering.py`.
