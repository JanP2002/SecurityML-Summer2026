#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generuje SAMODZIELNY raport HTML (obrazy osadzone w base64) — Lista 3.
Showcase wizualizacji grafów (CIFAR-10 + Białka) + ściąga do rozmowy z profesorem.

Uruchomienie:
  python wizualizacja_grafow.py      # najpierw wygeneruj figury do results_wiz/
  python raport_html.py              # potem zbuduj raport.html
"""
import base64, os, sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

WIZ = "results_wiz"
OUT = "raport.html"


def embed(fname):
    p = os.path.join(WIZ, fname)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


# ---- spis figur (plik, podpis) -------------------------------------------------
CIFAR = [
    ("fig_01_pipeline.png",
     "Pełny potok grafu losowego Jana: obraz → superpiksele SLIC → graf sąsiedztwa regionów (RAG, wszystkie krawędzie) → graf wylosowany. Szare = wszystkie sąsiedztwa, kolor = krawędzie wylosowane (DropEdge), czerwone = skróty small-world."),
    ("fig_02_ensemble_losowosc.png",
     "Sześć NIEZALEŻNYCH losowań tego samego obrazu. Graf jest losowy, więc na obraz budujemy ZESPÓŁ (ensemble) M grafów i uśredniamy cechy topologiczne — to stabilizuje deskryptor."),
    ("fig_03_det_vs_prob.png",
     "Twardy próg (krawędź gdy odległość koloru d ≤ tau) vs DropEdge (krawędź sąsiada zostaje z prawdopodobieństwem równym podobieństwu koloru)."),
    ("fig_04_small_world.png",
     "Skróty small-world: kilka dalekich krawędzi (czerwone) dokładanych losowo. Skracają ścieżki w grafie, dzięki czemu spacery Node2Vec sięgają kontekstu globalnego."),
    ("fig_05_prawdopodobienstwo_krawedzi.png",
     "ODPOWIEDŹ na pytanie o bliskość przestrzenną. Lewy: mechanizm losowania P = exp(−d²/2σ²) (zależny od koloru). Prawy: kandydaci na krawędź to WYŁĄCZNIE sąsiedzi (RAG); losowanie tylko odrzuca pary o niepodobnym kolorze, nigdy nie łączy odległych regionów."),
    ("fig_06_komponenty.png",
     "Po przecięciu krawędzi obraz rozpada się na komponenty spójne ~ obiekty (każdy kolor = osobny region). Liczba i rozmiar komponentów to graf-natywny sygnał o klasie."),
    ("fig_07_galeria_klasy.png",
     "Graf losowy (prob + small-world) dla różnych klas — topologia grafu zależy od obiektu na obrazie."),
    ("fig_08_n_segments.png",
     "Wpływ ziarnistości SLIC (30 / 60 / 120 superpikseli): więcej węzłów = bogatszy, drobniejszy graf."),
    ("fig_09_sigma.png",
     "Parametr sigma_feat steruje gęstością grafu probabilistycznego (przez color_w): większa sigma → wyższe prawdopodobieństwa → gęstszy graf."),
    ("fig_10_ER_baseline.png",
     "Graf oparty na obrazie (struktura) vs czysto losowy graf Erdősa–Rényiego na tych samych węzłach — baseline 'szumu topologicznego' używany w fuzji w notatniku."),
    ("fig_11_wagi_krawedzi.png",
     "Wagi krawędzi = (długość wspólnej granicy / max) × podobieństwo koloru. Łączą człon przestrzenny (granica) i wyglądowy (kolor)."),
    ("fig_12_stopnie.png",
     "Superpiksele pomalowane stopniem węzła — strukturalna 'mapa ciepła' grafu."),
    ("fig_13_random_walk.png",
     "Spacer losowy Node2Vec = 'zdanie' węzłów (kolejność odwiedzin). Takie sekwencje trafiają do Word2Vec i uczą osadzeń węzłów."),
    ("fig_14_adjacency.png",
     "Macierze sąsiedztwa tego samego obrazu: rzednięcie krawędzi przy przejściu RAG → twardy próg → losowanie (DropEdge)."),
    ("fig_15_galeria_10klas.png",
     "Graf losowy SLIC dla WSZYSTKICH 10 klas CIFAR-10 — przegląd różnorodności kształtów grafów między klasami."),
    ("fig_16_ciekawe_ksztalty.png",
     "Ranking obrazów po liczbie komponentów (twardy próg q=0.5): od najbardziej rozdrobnionych (góra) do najbardziej zwartych (dół). Rozdrobnienie to wprost skutek twardego warunku na krawędź."),
    ("fig_17_pixel_obiekty.png",
     "Graf PIKSELOWY: twardy próg w przestrzeni LAB tnie krawędzie na granicach, więc obiekt oddziela się od tła w osobne komponenty (np. samolot vs niebo)."),
]

METODY = ("fig_23_wszystkie_metody.png",
          "Wszystkie metody budowy grafu obok siebie na różnych klasach. Kolumny = metoda (kolor podpisu = rodzina: niebieski — nasz skrypt .py, czerwony — graf losowy z notatnika Jana, szary — baseline ER), wiersze = klasa obrazu.")

PROT = [
    ("fig_20_bialka_galeria.png",
     "Galeria grafów białek (układ Kamada–Kawai). Kolor węzła = typ elementu struktury wtórnej (SSE), rozmiar ~ stopień. Widać charakterystyczny łańcuch (backbone) z kontaktami."),
    ("fig_21_bialka_klasy.png",
     "Porównanie klas enzym vs nie-enzym obok siebie, ze statystykami strukturalnymi (średni stopień, współczynnik klasteryzacji C)."),
    ("fig_22_bialka_rozklady.png",
     "Rozkłady czterech cech strukturalnych w rozbiciu na klasy: rozmiar grafu, stopień, klasteryzacja, liczba komponentów."),
    ("fig_24_bialko_hero.png",
     "Pojedynczy, czytelny graf białka z ramką statystyk. Wydłużony kształt = łańcuch aminokwasów, poprzeczne krawędzie = kontakty przestrzenne."),
    ("fig_25_bialko_centralnosc.png",
     "Centralność pośrednictwa węzłów — strukturalne 'huby' (jasne) leżące na wielu najkrótszych ścieżkach, łączące regiony białka."),
    ("fig_26_bialko_macierze.png",
     "Macierze sąsiedztwa (enzym vs nie-enzym): przekątna = łańcuch aminokwasów, plamy poza przekątną = kontakty przestrzenne (fałdowanie)."),
    ("fig_27_bialko_sklad_spektrum.png",
     "'Odcisk' strukturalny klas: skład typów węzłów (lewy) oraz uśredniona sygnatura spektralna znormalizowanego Laplasjanu (prawy)."),
    ("fig_28_bialko_rozmiary.png",
     "Zróżnicowanie rozmiaru grafów białek — od najmniejszych do największych (kolory jak w galerii)."),
]

# ---- bloki tekstowe (HTML) -----------------------------------------------------
INTRO = """
<p><b>Pytanie Listy 3:</b> czy o przynależności obiektu do klasy decyduje sama
<b>struktura jego grafu</b> — i czy da się to wykryć <b>bez etykiet</b>? Każdy obiekt
(obraz CIFAR-10 albo białko) zamieniamy w graf, graf w wektor, wektory klastrujemy
<b>nienadzorowanie</b>; etykiety służą tylko do oceny, jak dobrze klastry pokrywają się
z prawdziwymi klasami.</p>
<p>Badamy dwa zbiory: <b>CIFAR-10</b> (obrazy → grafy pikselowe / patch / superpikselowe)
oraz <b>PROTEINS_full</b> (białka jako gotowe grafy: węzeł = aminokwas/SSE, krawędź = kontakt).
Ten dokument jest przede wszystkim <b>pokazem wizualizacji</b>, a na końcu zawiera ściągę
do rozmowy z profesorem.</p>
"""

KONSTRUKCJA = """
<h3>Warunek na krawędź (kluczowa zasada)</h3>
<p>Krawędź w grafie obrazu powstaje tylko gdy spełnione są <b>JEDNOCZEŚNIE</b> dwa warunki:
(1) węzły są <b>blisko przestrzennie</b> (sąsiadują na obrazie), ORAZ (2) są <b>podobne</b>
(odległość koloru/cech poniżej progu). Krawędzie na granicach obiektów są <b>przecinane</b>,
więc graf rozpada się na regiony ~ obiekty, a jego topologia zaczyna zależeć od klasy.</p>

<h3>Wersja losowa — notatnik Jana (<code>cifar_rand_graphs2_description.ipynb</code>)</h3>
<ul>
  <li><b>RAG</b> (graf sąsiedztwa regionów): węzły = superpiksele SLIC; kandydaci na krawędź
      to tylko superpiksele fizycznie graniczące (wykrywane skanem mapy etykiet).</li>
  <li><b>DropEdge (probabilistyczność)</b>: krawędź sąsiada zostaje z prawdopodobieństwem
      <code>color_w = exp(−d²/2σ²)</code> (im podobniejszy kolor, tym pewniej).</li>
  <li><b>Small-world</b>: dokładamy kilka losowych dalekich skrótów (skracają ścieżki).</li>
  <li><b>Ensemble</b>: bo graf jest losowy — budujemy M grafów na obraz i uśredniamy cechy.</li>
  <li><b>Waga krawędzi</b> = (długość wspólnej granicy / max) × podobieństwo koloru.</li>
</ul>
<p>Nasz skrypt <code>cifar_graph_clustering.py</code> trzyma <b>wariant deterministyczny</b>
(twardy próg) i bogaty zestaw metod (Node2Vec, WL, graph2vec, cechy topologiczne, combo, hybryda
z HOG). Oba podejścia trzymamy obok siebie, żeby móc je porównywać.</p>
"""

PYTANIE = """
<p><b>Pytanie o bliskość przestrzenną:</b> w (probabilistycznej) wersji grafu losowego opartego na
superpikselach — <b>w jaki sposób sprawdzamy, że łączone węzły odpowiadają BLISKIM
fragmentom obrazu?</b></p>
<p><b>Odpowiedź:</b> bliskość przestrzenną gwarantuje sam <b>zbiór kandydatów</b>, a nie
losowanie. Krawędź rozważamy <b>wyłącznie</b> między superpikselami, które się fizycznie
stykają — wykrywamy to skanując mapę etykiet SLIC (sąsiadujące piksele o różnych etykietach
tworzą wspólną granicę). To jest <b>graf sąsiedztwa regionów (RAG)</b>. Odległe superpiksele
nigdy nie wchodzą do zbioru kandydatów, więc nie mogą zostać połączone — niezależnie od koloru.
Długość wspólnej granicy dodatkowo <b>waży</b> krawędź. Losowanie (<code>rand &lt; color_w</code>)
działa już <b>tylko</b> na sąsiadach i odrzuca pary o niepodobnym kolorze. Patrz
<a href="#fig_05">fig_05</a>.</p>
<p class="small">Niuans do uczciwego powiedzenia: w kodzie <i>prawdopodobieństwo</i> połączenia
zależy od <b>koloru</b> (color_w); człon <b>przestrzenny</b> wchodzi przez warunek sąsiedztwa
(bramka RAG) oraz przez wagę (granica/max), a nie przez samo losowanie.</p>
"""

WYNIKI = """
<p>Skrót wyników (pełny opis: <code>RESULTS.md</code>). Podzbiór 3 klas (samolot/auto/statek,
SLIC, 900 obrazów), próg losowy = 0.333:</p>
<table>
  <tr><th>metoda</th><th>dokładność (CV)</th><th>ARI (nienadzor.)</th><th>HOG?</th></tr>
  <tr><td>hyb-slic (HOG ⊕ struktura)</td><td><b>0.786</b></td><td>0.025</td><td>tak</td></tr>
  <tr><td>baseline-hog</td><td>0.759</td><td>0.023</td><td>tak</td></tr>
  <tr><td>gnat-slic (graf-natywne)</td><td>0.613</td><td>0.110</td><td>nie</td></tr>
  <tr><td>combo-slic</td><td>0.612</td><td><b>0.116</b></td><td>nie</td></tr>
  <tr><td>topo-slic (sama struktura)</td><td>0.543</td><td>0.085</td><td>nie</td></tr>
  <tr><td>baseline-rgb</td><td>0.511</td><td>0.064</td><td>—</td></tr>
</table>
<ul>
  <li><b>Nadzorowanie:</b> hybryda HOG ⊕ struktura <b>przebija sam HOG</b> — struktura grafu
      niesie sygnał, którego nie ma w samym wyglądzie.</li>
  <li><b>Nienadzorowanie (właściwe pytanie Listy 3):</b> metody grafowe wygrywają zdecydowanie
      (ARI ~0.11–0.14 vs ~0.02 dla baseline'ów).</li>
  <li><b>Sama struktura</b> (<code>topo</code>, bez koloru) bije już baseline RGB.</li>
  <li>Czysto grafowa ścieżka nasyca się ~0.61–0.68; próg >0.7 daje dopiero hybryda z HOG.
      Pełne 10 klas jest trudne (nawet HOG ~0.45).</li>
</ul>
"""

QA = [
    ("Jak sprawdzacie, że łączone węzły to bliskie fragmenty obrazu?",
     "Bramką sąsiedztwa (RAG): krawędź rozważamy tylko między superpikselami mającymi wspólną granicę, wykrytą skanem mapy etykiet SLIC. Długość granicy waży krawędź. Odległe regiony nie są kandydatami."),
    ("Co dokładnie wchodzi do wzoru na prawdopodobieństwo połączenia?",
     "Samo podobieństwo koloru: color_w = exp(−d²/2σ²). Człon przestrzenny realizujemy przez warunek sąsiedztwa i przez wagę (granica/max), a nie przez samo losowanie."),
    ("Po co losowość (DropEdge), skoro można dać twardy próg?",
     "Uniezależnia wynik od jednego sztywnego progu i — uśredniona po zespole — działa jak regularyzacja (analogicznie do DropEdge w sieciach grafowych). Daje też miarę niepewności (odchylenie cech)."),
    ("Po co skróty small-world?",
     "Skracają ścieżki w grafie, dzięki czemu spacery losowe Node2Vec docierają do kontekstu globalnego, a nie tylko najbliższego sąsiedztwa."),
    ("Po co ensemble M grafów na obraz?",
     "Skoro graf jest losowy, cechy z jednego losowania są zaszumione. Średnia i odchylenie cech topologicznych po M grafach to znacznie stabilniejszy deskryptor obrazu."),
    ("Czym różni się wasz skrypt .py od notatnika?",
     ".py: wariant deterministyczny (twardy próg) + bogaty zestaw metod (WL, graph2vec, topo, combo, hybryda z HOG). Notatnik: wariant losowy (probabilistyczny) + ensemble. Trzymamy oba, żeby porównywać postęp."),
    ("Czy struktura grafu naprawdę niesie sygnał o klasie?",
     "Tak. Same cechy strukturalne (topo: liczba i rozmiary komponentów po przecięciu) biją baseline RGB, a nienadzorowanie metody grafowe mają wyraźnie wyższe ARI niż baseline'y."),
    ("Dlaczego nie zawsze przekraczacie 0.7?",
     ">0.7 jest realne tylko na podzbiorze kilku klas. Pełne 10 klas jest z natury trudne (nawet HOG osiąga ~0.45). Najmocniejszy wynik grafów jest nienadzorowany."),
]

CAVEATS = """
<ul>
  <li>Małe próbki → szum rzędu ±0.02; drobnych różnic między wariantami graf-natywnymi
      nie należy nadinterpretować.</li>
  <li>Każde <code>--per-class</code> losuje inny podzbiór obrazów, więc absolutne liczby
      zmieniają się między uruchomieniami; stabilne są tendencje, nie pojedyncze wartości.</li>
  <li>W grafie pikselowym obrazy są w [0,1], więc odległości koloru są małe — w trybie
      probabilistycznym większość krawędzi sąsiadów przeżywa (widoczna losowość siedzi
      głównie w skrótach small-world).</li>
</ul>
"""

CSS = """
:root{--ink:#1b1f24;--mut:#5b6570;--line:#e3e7ec;--accent:#1f6feb;--warn:#b3541e;--bg:#f7f8fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{background:linear-gradient(120deg,#12233b,#1f6feb);color:#fff;padding:38px 24px}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px}
header h1{margin:0 0 6px;font-size:30px}
header p{margin:2px 0;opacity:.92}
nav{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);
  padding:10px 0}
nav a{color:var(--accent);text-decoration:none;margin-right:16px;font-size:14px;white-space:nowrap}
nav a:hover{text-decoration:underline}
section{padding:26px 0;border-bottom:1px solid var(--line)}
h2{font-size:23px;margin:0 0 12px;border-left:5px solid var(--accent);padding-left:12px}
h3{font-size:18px;margin:18px 0 6px}
figure{background:#fff;border:1px solid var(--line);border-radius:10px;margin:18px 0;
  padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.05)}
figure img{width:100%;height:auto;border-radius:6px;display:block}
figcaption{color:var(--mut);font-size:14px;margin-top:10px}
figcaption b{color:var(--ink)}
.callout{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid var(--accent);
  border-radius:8px;padding:14px 16px;margin:14px 0}
.callout.warn{background:#fdf3ec;border-color:#f0d3bf;border-left-color:var(--warn)}
table{border-collapse:collapse;width:100%;margin:12px 0;background:#fff;font-size:15px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#f0f3f7}
.qa{margin:10px 0}
.qa .q{font-weight:600;margin-top:12px}
.qa .a{color:#37424d;margin:2px 0 0 0}
.small{color:var(--mut);font-size:13px}
footer{padding:24px;text-align:center;color:var(--mut);font-size:13px}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def figure_html(item):
    fname, cap = item
    data = embed(fname)
    if data is None:
        return f'<p class="small">[brak pliku {fname}]</p>'
    fid = fname.replace(".png", "")
    return (f'<figure id="{fid}"><img src="{data}" alt="{fname}" loading="lazy">'
            f'<figcaption><b>{fname}</b> — {cap}</figcaption></figure>')


def build():
    miss = [f for f, _ in (CIFAR + [METODY] + PROT) if embed(f) is None]
    if miss:
        print("  UWAGA, brak figur (uruchom najpierw wizualizacja_grafow.py):", ", ".join(miss))

    qa_html = "".join(f'<div class="q">P: {q}</div><div class="a">O: {a}</div>' for q, a in QA)

    p = []
    p.append(f"<!doctype html><html lang='pl'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Lista 3 — wizualizacje grafów (CIFAR + Białka)</title>"
             f"<style>{CSS}</style></head><body>")
    p.append("<header><div class='wrap'><h1>Lista 3 — reprezentacje grafowe: pokaz wizualizacji</h1>"
             "<p>CIFAR-10 (graf losowy oparty na superpikselach, wg notatnika Jana) oraz PROTEINS_full</p>"
             "<p>Pokaz wizualizacji + ściąga do rozmowy z profesorem</p></div></header>")
    p.append("<nav><div class='wrap'>"
             "<a href='#intro'>Pytanie</a><a href='#konstrukcja'>Budowa grafu</a>"
             "<a href='#bliskosc'>Bliskość przestrzenna</a><a href='#cifar'>CIFAR</a>"
             "<a href='#metody'>Metody</a><a href='#bialka'>Białka</a>"
             "<a href='#wyniki'>Wyniki</a><a href='#profesor'>Do rozmowy</a></div></nav>")

    p.append(f"<section id='intro'><div class='wrap'><h2>Pytanie badawcze</h2>{INTRO}</div></section>")
    p.append(f"<section id='konstrukcja'><div class='wrap'><h2>Jak budujemy graf</h2>{KONSTRUKCJA}</div></section>")
    p.append(f"<section id='bliskosc'><div class='wrap'><h2>Pytanie o bliskość przestrzenną</h2>"
             f"<div class='callout'>{PYTANIE}</div></div></section>")

    cifar_figs = "".join(figure_html(it) for it in CIFAR)
    p.append(f"<section id='cifar'><div class='wrap'><h2>CIFAR-10 — graf losowy (showcase)</h2>"
             f"<p>Wizualizacje wiernie odtwarzają konstrukcję z notatnika Jana "
             f"(<code>cifar_rand_graphs2_description.ipynb</code>).</p>{cifar_figs}</div></section>")

    p.append(f"<section id='metody'><div class='wrap'><h2>Przegląd wszystkich metod budowy grafu</h2>"
             f"{figure_html(METODY)}</div></section>")

    prot_figs = "".join(figure_html(it) for it in PROT)
    p.append(f"<section id='bialka'><div class='wrap'><h2>Białka — PROTEINS_full (showcase)</h2>"
             f"<p>Węzeł = element struktury (SSE), krawędź = kontakt; klasy: enzym vs nie-enzym.</p>"
             f"{prot_figs}</div></section>")

    p.append(f"<section id='wyniki'><div class='wrap'><h2>Wyniki w skrócie</h2>{WYNIKI}</div></section>")

    p.append("<section id='profesor'><div class='wrap'><h2>Przygotowanie do rozmowy z profesorem</h2>")
    p.append("<h3>Co podkreślić (główne tezy)</h3><ul>"
             "<li>Graf budujemy wg kluczowej zasady: krawędź = bliskość przestrzenna ORAZ podobieństwo.</li>"
             "<li>Wersja losowa: RAG + DropEdge + small-world + ensemble.</li>"
             "<li>Bliskość przestrzenną gwarantuje bramka sąsiedztwa (RAG), nie losowanie.</li>"
             "<li>Struktura niesie sygnał o klasie; hybryda z HOG przebija sam HOG; metody grafowe wygrywają nienadzorowanie.</li></ul>")
    p.append(f"<div class='callout'><b>Kluczowa odpowiedź (bliskość przestrzenna):</b>{PYTANIE}</div>")
    p.append(f"<h3>Przewidywane pytania i odpowiedzi</h3><div class='qa'>{qa_html}</div>")
    p.append(f"<h3>Uczciwe zastrzeżenia</h3><div class='callout warn'>{CAVEATS}</div>")
    p.append("</div></section>")

    p.append("<footer>Wygenerowano skryptem <code>raport_html.py</code> · figury: "
             "<code>wizualizacja_grafow.py</code> → <code>results_wiz/</code></footer>")
    p.append("</body></html>")

    html = "".join(p)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    kb = os.path.getsize(OUT) / 1024
    n = len(CIFAR) + 1 + len(PROT)
    print(f"Zapisano {OUT} ({kb:.0f} KB, {n} figur osadzonych).")


if __name__ == "__main__":
    build()
