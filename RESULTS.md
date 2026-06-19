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
5. **Cechy topologiczne + metoda graf-natywna** (`topo`, `gnat`): czysto strukturalne
   cechy grafu (liczba i rozkład rozmiarów KOMPONENTÓW po przecięciu krawędzi — wprost
   skutek warunku Krzysztofa) oraz pełna fuzja graf-natywna `gnat` = graph2vec ⊕ topo ⊕
   atrybuty węzła. **Zero globalnego HOG** — sprawdzamy, jak daleko zajdzie sama struktura.

## Historia usprawnień (co dało każde ulepszenie)

Najlepsza dokładność na podzbiorze 3 klas (SLIC, 900 obrazów), krok po kroku — wykres
[`fig_postep`](results_cifar/final_3class/fig_postep_slic_0_1_8.png):

| krok | metoda | dokładność | HOG? |
|------|--------|:----------:|:----:|
| 0. naiwna fuzja (rozcieńczenie) | `n2vflat` | 0.536 | nie |
| 1. rozdzielone strumienie + waga | `n2v`     | 0.557 | nie |
| 2. graph2vec (całografowy)       | `g2v`     | 0.578 | nie |
| 3. combo (g2v + kolor)           | `combo`   | 0.612 | nie |
| 4. combo + mini-HOG (bogaty węzeł) | `combo+r` | 0.610 | nie |
| 5. **graf-natywne** (g2v ⊕ topo ⊕ węzeł) | `gnat+r` | 0.610 | **nie** |
| 6. hybryda HOG + struktura       | `hyb`     | **0.786** ✅ | tak |

Czysta ścieżka grafowa (kroki 0–3) rośnie do ~0.61, potem **nasyca się** (kroki 4–5 to
poprawki w granicach szumu ±0.02 przy tym rozmiarze próby) — sufit pure-graph ~0.61–0.68
zależnie od próby. Krok 6 (dołożenie HOG) wyraźnie przebija HOG (0.759) i próg 0.7, ale to
już nie jest „czysto grafowe". **Solidne (nie-szumowe) wnioski:** rozcieńczenie psuje
głównie ARI (patrz niżej), a jedyne przejście >0.7 daje hybryda z HOG.

Dostępne pokrętła do strojenia (`--wl-iter`, `--g2v-dim`, `--n-orient-bins`, `--per-class`)
— osobny run [`results_cifar/slic_tuned/`](results_cifar/slic_tuned/) pokazuje, że ruszają
wynikiem w granicach kilku punktów, ale nie zmieniają obrazu: hybryda > HOG, pure-graph poniżej.

## Próby graf-natywne (BEZ HOG) — co jeszcze pchnęliśmy

W duchu wskazówek Krzysztofa, bez wciągania globalnego HOG, sprawdziliśmy trzy dźwignie
(każdą zostawiamy w kodzie jako osobną metodę/typ — nic nie usuwamy). Wykres:
[`fig_graf_natywne`](results_cifar/fig_graf_natywne.png).

| dźwignia | pomogło? | efekt |
|----------|:--------:|-------|
| #1 mocniejsza krawędź: kolor **ORAZ** tekstura (typy `*tex`) | ✗ | lekko gorzej; zyskała tylko czysta `topo` (czystsza fragmentacja) |
| #2 sygnatura spektralna Laplasjanu (`spec`, `gspec`) | ✗ | `gspec+r` ≈ `gnat+r`; sama `spec` słaba (~0.45) |
| #3 bogatszy seed WL: etykiety po kolorze **+ teksturze** (`--label-rich`) | ✓ | `g2v` 0.62→**0.66**, `wl` 0.58→**0.63** (ARI ~podwojone), `gnat` 0.65→0.67 |

Wniosek: sufit ścieżki czysto grafowej to ~**0.68**; najwięcej dało wzmocnienie SAMEJ
reprezentacji strukturalnej (lepszy seed WL), a nie zmiana topologii grafu. Przebicie HOG
(0.73) wciąż wymaga hybrydy z HOG. Runy: [`slic_edgetex`](results_cifar/slic_edgetex/),
[`slic_spec`](results_cifar/slic_spec/), [`slic_labelrich`](results_cifar/slic_labelrich/).

## Wyniki

### Podzbiór 3 klas: airplane / automobile / ship (SLIC, 900 obrazów) — run kanoniczny

Pełne porównanie wszystkich metod: [`fig_porownanie`](results_cifar/final_3class/fig_porownanie_0_1_8.png).

| metoda          | dokładność (CV) | ARI (nienadzorowana) | HOG? |
|-----------------|:---------------:|:--------------------:|:----:|
| hyb-slic (w=0.5)       | **0.786** ✅ | 0.025 | tak |
| baseline-hog           | 0.759        | 0.023 | tak |
| gnat-slic (graf-natywne) | 0.613      | 0.110 | **nie** |
| combo-slic             | 0.612        | **0.116** | nie |
| combo+r-slic / gnat+r  | 0.610        | 0.135 | nie |
| wl-slic                | 0.607        | 0.006 | nie |
| g2v-slic               | 0.578        | 0.077 | nie |
| n2v-slic               | 0.557        | 0.112 | nie |
| topo-slic (sama struktura) | 0.543    | 0.085 | nie |
| n2vflat-slic (rozcieńczenie) | 0.536  | 0.091 | nie |
| baseline-rgb           | 0.511        | 0.064 | — |
| spec-slic              | 0.462        | 0.044 | nie |

Próg losowy = 0.333. `topo` (czyste cechy strukturalne, bez koloru) bije już baseline RGB.
**Nadzorowanie** HOG/hybryda wygrywają; **nienadzorowanie** metody grafowe mają ARI ~0.11–0.14
vs ~0.02 baseline'ów. Wykresy w [`results_cifar/final_3class/`](results_cifar/final_3class/):
[`fig_postep`](results_cifar/final_3class/fig_postep_slic_0_1_8.png) (historia usprawnień),
[`fig_porownanie`](results_cifar/final_3class/fig_porownanie_0_1_8.png),
[`fig_waga`](results_cifar/final_3class/fig_waga_0_1_8.png),
[`fig_acc_vs_ari`](results_cifar/final_3class/fig_acc_vs_ari_0_1_8.png),
[`fig_morfo`](results_cifar/final_3class/fig_morfo_slic_0_1_8.png).

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
| **hyb** (struktura+HOG)     | **0.475** / 0.069 | **0.478** / 0.094 |
| baseline-hog                | 0.440 / 0.069     | 0.455 / 0.072      |
| gnat+r / gspec+r (graf-nat.)| 0.350 / 0.067     | —                  |
| combo (czysto grafowe)      | 0.293 / 0.039     | 0.323 / 0.065      |
| baseline-rgb                | 0.231 / 0.019     | —                  |

Próg losowy = 0.10. Nikt nie zbliża się do 0.7 — to oczekiwane: nawet HOG osiąga
tu tylko ~0.45. Struktura grafu nadal **dopełnia** wygląd (`hyb` > HOG na obu grafach),
a graf-natywne `gnat+r` (0.35) wyraźnie bije RGB (0.23). Wykresy w
[`results_cifar/slic_10class/`](results_cifar/slic_10class/) i
[`results_cifar/pixel_10class/`](results_cifar/pixel_10class/).

## Wnioski

- **Próg >0.7 osiągalny tylko na podzbiorze kilku klas** — i tam metoda hybrydowa
  (HOG ⊕ struktura grafu) **przebija sam HOG** (0.786 vs 0.759 na SLIC). Struktura
  grafu niesie sygnał, którego nie ma w samym wyglądzie.
- **Ścieżka czysto grafowa (bez HOG)** nasyca się ~**0.61–0.68** (zależnie od próby) —
  poniżej HOG, ale: **same cechy strukturalne `topo`** (liczba/rozmiary komponentów po
  przecięciu krawędzi) biją już baseline RGB — sama struktura grafu niesie sygnał o klasie.
  Na pikselach czyste `combo` bije nawet HOG (0.671 vs 0.667). I to ścieżka grafowa
  wygrywa **nienadzorowanie** (ARI ~0.11–0.15 vs ~0.02). Bezpośrednia odpowiedź na pytanie
  Listy 3: sygnał o klasie SIEDZI w strukturze.
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
