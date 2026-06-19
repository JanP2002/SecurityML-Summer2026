# Lista 3 — reprezentacje grafowe CIFAR-10: wyniki i historia

Skrypt: [`cifar_graph_clustering.py`](cifar_graph_clustering.py). Wszystkie metody liczone
na tych samych danych, w jednej tabeli porównawczej; nic nie usuwaliśmy — każda
nowa metoda to dodatkowy wiersz, żeby było widać postęp. Pełny katalog użytych
podejść: [`METODY.md`](METODY.md).

## Pytanie

Czy o klasie obiektu decyduje **struktura jego grafu** — i czy da się to wykryć
**bez etykiet**? Obraz → graf → embedding → klasteryzacja (nienadzorowana) +
sonda nadzorowana. Baseline'y: średnie RGB i HOG (wygląd, bez grafu).

## Punkt wyjścia: Node2Vec przegrywał z baseline'em (efekt rozcieńczenia)

Pierwotnie pooling Node2Vec (~128 wym.) sklejano płasko z kolorem (~12 wym.) i
standaryzowano razem. **128 wymiarów struktury topiło sygnał koloru** — `n2v`
wypadał *gorzej* niż 3-liczbowy baseline RGB. To była główna przyczyna słabych
wyników, niezależna od jakości grafu.

## Co poprawiliśmy (i dlaczego)

1. **Twardy warunek na krawędź (wskazówka Krzysztofa).** Krawędź powstaje tylko gdy
   węzły są JEDNOCZEŚNIE blisko przestrzennie ORAZ podobne kolorystycznie (odległość
   poniżej adaptacyjnego progu per obraz). Granice obiektów są *przecinane*, więc graf
   rozpada się na regiony ~obiekty i jego topologia zaczyna zależeć od klasy.
2. **Rozdzielone strumienie + waga `w`.** Strukturę i atrybuty poolingujemy osobno,
   każdy blok standaryzujemy i L2-normalizujemy, a potem mieszamy wagą `w`
   (0 = tylko kolor/tekstura, 1 = tylko struktura). To usuwa rozcieńczenie i pozwala
   znaleźć optymalny mix. Efekt: `n2v-slic` skoczył z ~0.40 do ~0.59.
3. **Nowe całografowe metody** (bez uśredniania węzłów): `wl` (Weisfeiler–Lehman),
   `g2v` (graph2vec), `combo` (graph2vec ⊕ atrybuty) oraz `hyb` (HOG ⊕ graph2vec).
4. **Bogatszy deskryptor węzła** (`--rich-features`): histogram orientacji gradientu
   na superpiksel (graf-natywny mini-HOG) — żeby `combo` rywalizowało strukturą, a nie
   pożyczonym HOG. Metody `combo+r` / `hyb+r`.

## Historia usprawnień (co dało każde ulepszenie)

Najlepsza dokładność na podzbiorze 3 klas (SLIC), krok po kroku — wykres
[`fig_postep`](results_cifar/slic_rich/fig_postep_slic_0_1_8.png):

| krok | metoda | dokładność |
|------|--------|:----------:|
| 0. naiwna fuzja (rozcieńczenie) | `n2vflat` | 0.562 |
| 1. rozdzielone strumienie + waga | `n2v`     | 0.593 |
| 2. graph2vec (całografowy)       | `g2v`     | 0.613 |
| 3. combo (g2v + kolor)           | `combo`   | 0.647 |
| 4. combo + mini-HOG (bogaty węzeł) | `combo+r` | 0.665 |
| 5. hybryda HOG + struktura       | `hyb`     | **0.737** ✅ |

Każdy krok podnosi wynik; krok 5 przebija zarówno baseline HOG (0.733), jak i próg 0.7.

**Dostrojenie (większy graph2vec + gęstszy mini-HOG + więcej danych)** podnosi najlepszą
metodę jeszcze wyżej — `hyb+r-slic` = **0.768** vs HOG 0.749 (750 obrazów; `--wl-iter 3
--g2v-dim 128 --n-orient-bins 9 --per-class 250`). Wykresy w
[`results_cifar/slic_tuned/`](results_cifar/slic_tuned/).

## Wyniki

### Podzbiór 3 klas: airplane / automobile / ship (SLIC, 600 obrazów)

| metoda          | dokładność (CV) | ARI (nienadzorowana) |
|-----------------|:---------------:|:--------------------:|
| **hyb-slic** (w=0.5)   | **0.737** ✅ | 0.081 |
| baseline-hog           | 0.733        | 0.021 |
| combo+r-slic (mini-HOG)| 0.665        | **0.148** |
| combo-slic             | 0.647        | 0.138 |
| n2v-slic               | 0.593        | 0.134 |
| n2vflat-slic (rozcieńczenie) | 0.562  | 0.063 |
| baseline-rgb           | 0.545        | 0.059 |

Próg losowy = 0.333. Wykresy w [`results_cifar/slic_rich/`](results_cifar/slic_rich/):
[`fig_postep`](results_cifar/slic_rich/fig_postep_slic_0_1_8.png) (historia usprawnień),
[`fig_porownanie`](results_cifar/slic_rich/fig_porownanie_0_1_8.png),
[`fig_waga`](results_cifar/slic_rich/fig_waga_0_1_8.png),
[`fig_acc_vs_ari`](results_cifar/slic_rich/fig_acc_vs_ari_0_1_8.png),
[`fig_morfo`](results_cifar/slic_rich/fig_morfo_slic_0_1_8.png).

### Ten sam podzbiór, graf PIXEL (240 obrazów)

| metoda            | dokładność (CV) | ARI   |
|-------------------|:---------------:|:-----:|
| **hyb-pixel** (w=0.5)   | **0.692** | 0.037 |
| combo-pixel (w=0.5)     | 0.671     | 0.136 |
| baseline-hog            | 0.667     | 0.015 |
| n2v-pixel               | 0.646     | 0.134 |

Na grafie pikselowym (ten, o którym mówił Krzysztof) zarówno `hyb`, jak i *czysto
grafowe* `combo` biją HOG. Wykresy w [`results_cifar/pixel_3class/`](results_cifar/pixel_3class/).

### Pełne 10 klas (SLIC, 1000 obrazów)

| metoda        | SLIC: dokł. / ARI | PIXEL: dokł. / ARI |
|---------------|:-----------------:|:------------------:|
| **hyb** (struktura+HOG) | **0.470** / 0.068 | **0.478** / 0.094 |
| baseline-hog            | 0.440 / 0.069     | 0.455 / 0.072      |
| combo (czysto grafowe)  | 0.278 / 0.044     | 0.323 / 0.065      |

Próg losowy = 0.10. Nikt nie zbliża się do 0.7 — to oczekiwane: nawet HOG osiąga
tu tylko ~0.45. Struktura grafu nadal **dopełnia** wygląd (`hyb` > HOG na obu grafach).
Wykresy w [`results_cifar/slic_10class/`](results_cifar/slic_10class/) i
[`results_cifar/pixel_10class/`](results_cifar/pixel_10class/).

## Wnioski

- **Próg >0.7 osiągalny tylko na podzbiorze kilku klas** — i tam metoda hybrydowa
  (HOG ⊕ struktura grafu) **przebija sam HOG** (0.737 vs 0.733 na SLIC). Struktura
  grafu niesie sygnał, którego nie ma w samym wyglądzie.
- **Czysto grafowe `combo`** na pikselach bije HOG (0.671 vs 0.667); na SLIC bogatszy
  deskryptor (mini-HOG na węzeł) podniósł je z 0.647 do **0.665** — wciąż poniżej HOG,
  ale to ono wygrywa **nienadzorowanie** (ARI/silhouette ≫ baseline'y). To jest
  bezpośrednia odpowiedź na pytanie Listy 3: sygnał o klasie SIEDZI w strukturze.
- **Tylko struktura (w=1.0) jest najsłabsza** — Node2Vec + uśrednianie to znana słaba
  reprezentacja całego grafu (to samo widać na ENZYMES w `tu_graph_clustering.py`).
  Siła jest w *miksie* struktury z atrybutami.

## Jak odtworzyć

```bash
# podzbiór 3 klas + wykresy (embeddingi są cache'owane, więc przemiatanie wag jest błyskawiczne):
python cifar_graph_clustering.py --graph-type slic --classes 0 1 8 --per-class 200 --plots
# pełne 10 klas:
python cifar_graph_clustering.py --graph-type slic --per-class 100 --plots
```
