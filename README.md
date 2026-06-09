# Challenge 3 — Krok 1

**Autor:** Piotr Zapała
**Etap:** Krok 1 (fundament / „zajawka" — punkt wyjścia do dalszej rozbudowy przez zespół)

---

## Na czym polega Challenge 3

Challenge bada jedno pytanie:

> **Czy o przynależności obiektu do klasy decyduje sama struktura jego grafu — i czy da się to wykryć BEZ etykiet?**

Czyli: reprezentujemy każdy obiekt jako graf, zamieniamy graf na wektor (embedding),
grupujemy te wektory **nienadzorowanie** (nie pokazując algorytmowi klas), a etykiety
wykorzystujemy dopiero na końcu — jako klucz odpowiedzi do oceny, czy znalezione
skupiska pokrywają się z prawdziwymi klasami. To test, czy sygnał o klasie jest
**wewnętrzną** własnością struktury grafu, a nie czymś, czego model nauczył się,
bo dostał odpowiedzi.

**Cel praktyczny.** Zbudować **rozszerzalny benchmark reprezentacji grafów**:
jedno zadanie, wiele reprezentacji, te same metryki — tak, aby każdy kolejny
pomysł (nowy embedding, nowy zbiór) dało się „wpiąć" i od razu porównać.

---

## Pytanie badawcze

Która definicja „podobieństwa grafów" najlepiej pasuje do prawdziwych klas?
Porównujemy reprezentacje różniące się tym, _co_ uznają za istotne:
topologia całego grafu vs lokalne wzorce połączeń vs cechy węzłów; reprezentacja
zliczana vs uczona; poziom całego grafu vs poziom węzła.

---

## Co robi Krok 1 (ten etap)

1. **Loader TUDataset** — wczytuje ENZYMES (600 grafów, 6 klas po 100; węzły mają
   typ i 18-wym. atrybuty fizyko-chemiczne) do grafów `networkx`.
2. **Pięć reprezentacji** (jeden wektor na CAŁY graf):
   - `topo` — statystyki strukturalne (rozmiar, cykle, rozkład stopni),
   - `attr` — pooling atrybutów węzłów (mean⊕std),
   - `wl` — zliczanie podstruktur Weisfeilera–Lehmana (init etykietą węzła),
   - `graph2vec` — te same wzorce WL, ale uczone gęsto przez Doc2Vec,
   - `node2vec` — embedding węzłów + agregacja (metoda poziomu węzła; kontrast).
3. **Klasteryzacja** — KMeans (k = liczba klas) + HDBSCAN.
4. **Ocena** — zewnętrzna (ARI, NMI, V-measure) i wewnętrzna (silhouette,
   Davies-Bouldin, Calinski-Harabasz), **sonda nadzorowana kNN** (cross-val acc/F1)
   oraz **morfoprzestrzeń 2D** (PCA, punkty pokolorowane klasą).
5. **Skalowanie zależne od metody**: topo/attr/node2vec → StandardScaler;
   wl/graph2vec → (SVD) + L2-normalizacja.

---

## Wyniki na ENZYMES (poziom losowy sondy = 0.167)

    method      km_ARI  km_NMI  silhouette  probe_acc  probe_f1
    topo        0.019   0.046   0.226       0.357      0.350
    attr        0.026   0.072   0.207       0.583      0.586   <- najlepsza separowalnosc
    wl          0.037   0.075   0.125       0.513      0.518   <- najlepsze ARI klastrow
    graph2vec   0.027   0.053   0.081       0.483      0.486
    node2vec    0.019   0.044   0.135       0.223      0.220   <- najslabszy

**Wnioski:**

- **Separowalność ≠ klasteryzowalność.** ARI ~0 dla wszystkich, ale sonda kNN sięga
  0.58 (przy losowym 0.167) — sygnał o klasie JEST w embeddingach, tylko nie układa
  się w czyste, oddzielne klastry. ENZYMES słynie z tej trudności.
- **Atrybuty węzłów niosą najwięcej** (`attr` > `wl`/`graph2vec` > `topo`/`node2vec`).
- **Node2Vec najsłabszy** — potwierdza, że metoda poziomu węzła słabo nadaje się do
  zadań na poziomie całego grafu (osobna, niewyrównana przestrzeń na każdy graf).

---

## Pliki w repo

- `tu_graph_clustering.py` — główny skrypt (ENZYMES / dowolny zbiór TU przez `--name`).
- `README.md` — szczegóły techniczne tego skryptu.

---

## Uruchomienie

    # benchmark 5 reprezentacji (auto-pobiera ENZYMES):
    python tu_graph_clustering.py --all-embeddings

    # to samo na białkach — uwaga: w tym mirrorze zbiór nazywa się PROTEINS_full
    # (1113 grafów, 2 klasy: enzym / nie-enzym):
    python tu_graph_clustering.py --name PROTEINS_full --all-embeddings

Uwaga (Windows): równoległość w node2vec nie działa — ustaw w `Config`
`n2v_workers = 1` albo użyj WSL/Linux/macOS.

---

## Następne kroki (Krok 2+ — dla zespołu)

- **Krok 2: białka (`PROTEINS_full`)** — łatwiejszy problem 2-klasowy, dobry kontrast
  do „trudnego" ENZYMES; ten sam kod, tylko `--name PROTEINS_full`.
- `combo` = konkatenacja `attr` + `wl` (struktura + cechy) — zwykle najmocniejsze na ENZYMES.
- **GNN** (GIN/GraphSAGE) uczony z atrybutami węzłów jako kolejna reprezentacja w tabeli.
- **Jądra grafowe** (grakel: WL-OA, shortest-path) + klasteryzacja spektralna.
- Inne zbiory TU jednym przełącznikiem `--name` (MUTAG, DD, IMDB-BINARY…).
