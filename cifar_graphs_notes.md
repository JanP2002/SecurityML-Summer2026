[Obraz CIFAR] -> [Tworzenie Grafu (PIXEL)] -> [Node2Vec: Spacery Losowe (Zdania)] 
                                                      |
                                                      v
[Klasyfikacja / Wizualizacja] <- [Wektor Obrazu] <- [Word2Vec (Gensim)]



 Trzy rodzaje badanych grafów (Metody modelowania)

Choć zależą one od dokładnej implementacji w ukrytym kodzie, standardowo w takich eksperymentach bada się następujące 3 topologie:

Metoda PIXEL (Graf Siatkowy / Grid Graph):

Węzły (Nodes): Każdy pojedynczy piksel obrazu to jeden węzeł.

Krawędzie (Edges): Połączenia tworzone są tylko między pikselami sąsiadującymi ze sobą w przestrzeni (najczęściej 4- lub 8-kierunkowe sąsiedztwo).

Wagi krawędzi: Zależą od różnicy intensywności (jasności lub koloru) między sąsiadującymi pikselami. Im bardziej podobne piksele, tym "silniejsza" (cięższa) krawędź.

Metoda SLIC / Superpiksele (Region Adjacency Graph):

Węzły (Nodes): Obraz jest wstępnie segmentowany (np. algorytmem SLIC) na tzw. superpiksele (małe, spójne grupy pikseli o podobnym kolorze). Każdy superpiksel staje się pojedynczym węzłem. Znacząco redukuje to wielkość grafu.

Krawędzie (Edges): Połączenia istnieją między superpikselami, które fizycznie ze sobą graniczą na obrazie.

Wagi krawędzi: Obliczane na podstawie różnicy średnich kolorów/intensywności połączonych superpikseli.

Metoda KNN / Patch-based (Graf K-Najbliższych Sąsiadów):

Węzły (Nodes): Obraz dzielony jest na małe łatki (patches), np. 3x3 lub 5x5 pikseli. Każda łatka to węzeł.

Krawędzie (Edges): Krawędzie nie zależą od położenia przestrzennego. Każda łatka łączona jest z $K$ najbardziej do niej podobnymi łatkami w całym obrazie (np. na podstawie odległości euklidesowej ich wektorów cech).

Wagi krawędzi: Określane przez stopień podobieństwa (odległość w przestrzeni cech) między łatkami.

2. Działanie algorytmu Node2Vec na utworzonym grafie

Gdy pojedynczy obraz jest już reprezentowany jako graf (niezależnie od wybranej metody z powyższych), do akcji wkracza Node2Vec:

Błądzenie losowe (Random Walks): Z każdego węzła w grafie wypuszczani są "wirtualni spacerowicze". Wędrują oni po krawędziach do sąsiednich węzłów. Algorytm Node2Vec steruje tymi spacerami za pomocą parametrów $p$ (powrót do poprzedniego węzła) i $q$ (eksploracja nowych, dalszych węzłów). Prawdopodobieństwo przejścia zależy również od wagi krawędzi.

Generowanie sekwencji: Wynikiem błądzenia są "zdania" (sekwencje węzłów), np.: [Węzeł_1, Węzeł_4, Węzeł_9, ...]. Zapisują one strukturę lokalną grafu obrazu.

Model Skip-Gram (Word2Vec): Te sekwencje trafiają do płytkiej sieci neuronowej. Sieć ta uczy się przypisywać każdemu węzłowi wektor liczbowy (tzw. osadzenie / embedding) w taki sposób, aby węzły często występujące blisko siebie w sekwencjach miały podobne wektory w przestrzeni wielowymiarowej.


