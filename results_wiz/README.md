# Wizualizacje grafów — Lista 3

Wygenerowane przez [`wizualizacja_grafow.py`](../wizualizacja_grafow.py).
Nacisk na NOWY notatnik Jana `cifar_rand_graphs2_description.ipynb` (graf LOSOWY na superpikselach:
probabilistyczne wstawianie krawędzi + skróty small-world + ensemble M grafów na obraz).

## CIFAR-10 — graf losowy (Jan)
- **fig_01_pipeline** — pełny potok: obraz -> SLIC -> RAG (wszystkie sąsiedztwa) -> graf wylosowany.
- **fig_02_ensemble_losowosc** — M niezależnych losowań jednego obrazu (dlaczego potrzebny ensemble).
- **fig_03_det_vs_prob** — twardy próg d<=tau vs DropEdge (rand < color_w).
- **fig_04_small_world** — dalekie skróty small-world (czerwone) dokładane do grafu.
- **fig_05_prawdopodobienstwo_krawedzi** — ODPOWIEDŹ na pytanie o bliskość przestrzenną: gwarantuje ją bramka RAG, nie losowanie.
- **fig_06_komponenty** — rozpad na komponenty ~obiekty po przecięciu krawędzi.
- **fig_07_galeria_klasy** — graf losowy dla różnych klas (topologia zależna od obiektu).
- **fig_08_n_segments** — wpływ ziarnistości SLIC (30/60/120).
- **fig_09_sigma** — sigma_feat steruje gęstością grafu probabilistycznego.
- **fig_10_ER_baseline** — graf oparty na obrazie vs czysto losowy Erdős–Rényi.
- **fig_11_wagi_krawedzi** — wagi = (granica/max) × podobieństwo koloru.
- **fig_12_stopnie** — mapa stopni węzłów (struktura jako 'ciepło').
- **fig_13_random_walk** — spacer losowy Node2Vec ('zdanie' węzłów) zaznaczony na grafie.
- **fig_14_adjacency** — macierze sąsiedztwa: rzednięcie RAG -> próg -> losowanie.
- **fig_15_galeria_10klas** — graf losowy SLIC dla wszystkich 10 klas (różnorodność kształtów).
- **fig_16_ciekawe_ksztalty** — ranking obrazów po liczbie komponentów (rozdrobnione vs zwarte).
- **fig_17_pixel_obiekty** — graf pikselowy: obiekt oddziela się od tła w osobne komponenty.

## Przegląd wszystkich metod budowy grafu
- **fig_23_wszystkie_metody** — WSZYSTKIE metody obok siebie na różnych klasach: nasz `.py`
  (pixel / pixel+tex / patch / slic / slic+tex), graf losowy Jana (slic-prob / slic+small-world)
  oraz baseline Erdős–Rényi. Kolor podpisu kolumny = rodzina metody.

## Białka — PROTEINS_full
- **fig_20_bialka_galeria** — galeria grafów (układ kamada-kawai, kolor = typ SSE, rozmiar ~ stopień, legenda).
- **fig_21_bialka_klasy** — enzym vs nie-enzym obok siebie, ze statystykami (⟨deg⟩, klasteryzacja).
- **fig_22_bialka_rozklady** — rozkłady 4 cech wg klasy: rozmiar, stopień, klasteryzacja, komponenty.
- **fig_24_bialko_hero** — pojedynczy duży graf z ramką statystyk.
- **fig_25_bialko_centralnosc** — węzły kolorowane centralnością pośrednictwa (huby strukturalne).
- **fig_26_bialko_macierze** — macierze sąsiedztwa enzym vs nie-enzym (łańcuch + kontakty 3D).
- **fig_27_bialko_sklad_spektrum** — skład typów węzłów + uśredniona sygnatura spektralna wg klasy.
- **fig_28_bialko_rozmiary** — galeria posortowana po rozmiarze (od małych do dużych).
