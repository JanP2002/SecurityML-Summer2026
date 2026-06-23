# graphrep — reprezentacje grafowe, klastrowanie i prywatność

Framework do badania **reprezentacji grafowych** jednocześnie pod kątem **użyteczności**
(klastrowanie bez etykiet, separowalność klas) i **prywatności** (wyciek informacji o klasie
oraz odwracalność reprezentacji). Jedno narzędzie obejmuje dwa typy danych pod wspólnym
interfejsem:

- **grafy gotowe** — zbiory TU (`ENZYMES`, `PROTEINS_full`),
- **grafy budowane z obrazu** — CIFAR-10 (reprezentacje `pixel` / `patch` / `slic`).

Wspólna oś badawcza: **graf jako reprezentacja obfuskująca dane** — surowy obiekt (obraz,
cząsteczka) kodowany jest jako abstrakcyjny graf, który „ukrywa" oryginał. Pytanie brzmi,
ile z oryginału da się z takiej reprezentacji odzyskać i czy klasa obiektu pozostaje
wykrywalna bez etykiet.

---

## Rama: użyteczność a prywatność

| pytanie | metryka | interpretacja |
|---|---|---|
| ile klasy **wycieka** bez etykiet? | ARI / NMI / silhouette (klastrowanie) | wysokie = duży wyciek |
| jaka jest **separowalność** klas? | sonda nadzorowana (acc / F1) | górna granica wycieku |
| czy reprezentacja jest **odwracalna**? | atak rekonstrukcyjny (R² odtworzenia treści) | wysokie R² = słaba prywatność |

**Obfuskacja jako pokrętło siły.** Reprezentację można dodatkowo zaburzać na poziomie
krawędzi (`rewire` / `dropedge` / `shortcuts` / `er`) — to perturbacja struktury w duchu
edge-differential-privacy. Przemiatając siłę obfuskacji, otrzymujemy **krzywą
prywatność–użyteczność**: jeden kraniec to graf wiernie oddający obiekt (duża użyteczność,
większy wyciek), drugi to graf losowy (silna obfuskacja, mała użyteczność).

---

## Konstrukcja grafu z obrazu

Krawędź powstaje tylko gdy spełnione są **jednocześnie** dwa warunki: węzły leżą blisko
siebie **przestrzennie** ORAZ są **podobne kolorystycznie/cechowo** (próg adaptacyjny —
kwantyl rozkładu odległości per obraz, albo stały `--tau`). Odległości powyżej progu są
*przecinane* (nie tylko osłabiane wagą), więc graf rozpada się na regiony ~obiekty, a jego
topologia zaczyna zależeć od klasy. Dostępne typy grafu:

- `pixel` — węzeł = piksel (przestrzeń LAB), sąsiedztwo 4/8-kierunkowe odcięte progiem koloru,
- `patch` — węzeł = nakładający się patch, siatka + kNN ograniczone przestrzennie, progowane,
- `slic` — węzeł = superpiksel SLIC, sąsiedztwo z detekcji granic odcięte progiem różnicy koloru.

Opcje: `--edge-texture` (mocniejszy warunek: kolor ORAZ tekstura), `--rich-features`
(mini-HOG na superpiksel), `--label-rich` (seed WL po pełnym deskryptorze zamiast po kolorze).

---

## Mapa modułów (biblioteka `graphrep/`)

```
graphrep/
  config.py      Config — jedna konfiguracja całości
  data.py        źródła grafów: tu (ENZYMES/PROTEINS_full, auto-pobieranie),
                 cifar (pixel/patch/slic), synth (obrazy syntetyczne do testów)
                 + obfuskacja (rewire/dropedge/shortcuts/er) na dowolnej liście grafów
  features.py    etykiety węzłów (seed WL: typ węzła w TU lub KMeans kolorów w obrazie)
                 + pooling atrybutów (strumień atrybutowy)
  embeddings.py  reprezentacje: topo, attr, wl, graph2vec, node2vec, spectral
                 + fuzja strumieni z wagą w (rozdzielone, L2, mieszane — bez rozcieńczenia)
  evaluate.py    klastrowanie (kmeans/spectral/agglo/hdbscan) + metryki + sonda nadzorowana
                 + atak rekonstrukcyjny (metryka prywatności)
  plots.py       benchmark, krzywa prywatność–użyteczność, sweep wagi, morfoprzestrzeń
run_experiments.py   skrypt EKSPERYMENTÓW (źródło × metoda × waga × obfuskacja) → CSV
make_plots.py        skrypt WYKRESÓW (czyta CSV → figury)
```

---

## Reprezentacje (jeden wektor na graf)

Do wyboru w `--methods`:

| nazwa | poziom | opis |
|---|---|---|
| `topo` | całografowy | cechy strukturalne (liczba/rozmiary komponentów, stopnie, cykle, asortatywność, klasteryzacja) |
| `wl` | całografowy | feature map jądra Weisfeilera–Lehmana → TruncatedSVD |
| `g2v` | całografowy | graph2vec (wzorce WL uczone gęsto przez Doc2Vec) |
| `spec` | całografowy | sygnatura spektralna (k najmniejszych wartości własnych Laplasjanu) |
| `attr` | agregacja węzłów | pooling atrybutów węzła (mean ⊕ std) — strumień atrybutowy |
| `n2v` | poziom węzła | node2vec + pooling — **kontrast** (osobna, niewyrównana przestrzeń na graf → słaby do klastrowania) |
| `combo` | fuzja | `g2v ⊕ atrybuty`, waga `w` |
| `gnat` | fuzja | `g2v ⊕ topo ⊕ atrybuty`, waga `w` |

Do klastrowania preferowane są reprezentacje **całografowe, permutacyjnie niezmiennicze**
(`topo`, `wl`, `g2v`, `spec`). Fuzja miesza strumień strukturalny i atrybutowy wagą `w`
(`w=1` → tylko struktura, `w=0` → tylko atrybuty); każdy strumień jest osobno
normalizowany do wektora jednostkowego, więc liczba wymiarów jednego nie „topi" drugiego.

---

## Instalacja

```bash
pip install -r requirements_framework.txt
# albo:
conda create -n graphrep -c conda-forge python=3.11 numpy scipy scikit-learn \
      networkx gensim scikit-image joblib matplotlib
conda activate graphrep && pip install node2vec
```

`source=cifar` dodatkowo wymaga `torchvision` (i dostępu do sieci, by pobrać CIFAR-10).
W środowiskach bez sieci do testu ścieżki obrazowej służy `--source synth`.

---

## Uruchomienie (z katalogu, w którym leży `graphrep/`)

```bash
# benchmark reprezentacji na ENZYMES:
python run_experiments.py --source tu --dataset ENZYMES --per-class 50

# białka (2 klasy, łatwiejszy kontrast):
python run_experiments.py --source tu --dataset PROTEINS_full --per-class 100

# krzywa PRYWATNOŚĆ–UŻYTECZNOŚĆ (obfuskacja przez rewiring):
python run_experiments.py --source tu --dataset ENZYMES --per-class 50 --privacy-curve

# ścieżka obrazowa na CIFAR (graf z warunkiem bliskość + podobieństwo):
python run_experiments.py --source cifar --graph-type slic --classes 0 1 8 --per-class 200 --rich-features
#   (bez sieci / do testu: --source synth)

# wykresy z dowolnego przebiegu:
python make_plots.py --out-dir results
```

Przydatne flagi: `--methods topo wl g2v combo gnat`, `--cluster-algo spectral`,
`--weights 0 0.5 1`, `--obf-method rewire|dropedge|shortcuts|er`,
`--obf-strengths 0 0.25 0.5 0.75 1`, `--label-rich`, `--edge-quantile 0.6`.

---

## Wyjście (w `--out-dir`)

- `wyniki.csv` — jeden wiersz na (metoda × waga × siła obfuskacji): `km_ARI`, `km_NMI`,
  `silhouette`, sonda `probe_acc`/`probe_f1`, `recon_r2` (prywatność), `dim`.
- `fig_benchmark.png` — ranking reprezentacji (sonda + ARI).
- `fig_privacy_utility.png` — krzywa prywatność–użyteczność (przy `--privacy-curve`).
- `fig_weight_sweep.png` — wpływ wagi fuzji (struktura vs atrybuty).
- `fig_morphospace.png` — rzut 2D reprezentacji (kolor = klasa).
- `cache/` — surowe embeddingi (drogie `graph2vec`/`node2vec` liczone raz; sweep wag i
  obfuskacji jest dzięki temu szybki).

---

## Przykładowy wynik (ENZYMES)

Atak rekonstrukcyjny (R² odtworzenia treści obiektu z embeddingu):

- `attr` (pooling atrybutów) → R² ≈ 0.7: reprezentacja **wycieka** treść (bo nią jest),
- `topo` / `g2v` / `node2vec` (strukturalne) → R² ujemne: ze struktury **nie da się**
  odtworzyć treści → wysoka prywatność.

Obfuskacja (rewiring) stopniowo **niszczy użyteczność strukturalną**, podczas gdy strumień
atrybutów jest na perturbację krawędzi **niewrażliwy** — `combo` przy silnej obfuskacji
zwija się do atrybutów. Wniosek: reprezentacja grafowa potrafi ukryć obiekt (niska
odwracalność) przy zachowaniu części użyteczności, a poziom tego kompromisu można świadomie
regulować obfuskacją.

---

## Rozszerzenia

- **Mocniejsze reprezentacje do klastrowania**: NetLSD, liczności graphletów, klastrowanie
  spektralne na jądrze WL (haki w `embeddings.STRUCT_EMBEDDERS`).
- **Samonadzorowany GNN** (np. InfoGraph / GraphCL) jako kolejna reprezentacja w tabeli.
- **Formalny mechanizm prywatności**: randomized-response na krawędziach z parametrem ε
  (edge-DP) zamiast heurystycznej obfuskacji.
- **Silniejszy atak**: rekonstrukcja pełnego obrazu (nie tylko średniej treści) lub
  re-identyfikacja pojedynczych obiektów.