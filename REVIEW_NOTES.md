# Review notes — Omnisolver bruteforce (Version 2)

Stan checklisty po weryfikacji manuskryptu `bruteforce.tex` i plików pomocniczych.

## Status ogólny

PDF składa się poprawnie przez `latexmk`. Bibliografia jest obecnie w dobrym stanie
technicznym: wszystkie cytowane klucze istnieją, Ray jest cytowany jako formalna
publikacja OSDI 2018, a wpisy z ponad sześcioma autorami renderują się jako
`et al.`.

Nadal nie są domknięte wszystkie punkty merytoryczne z perspektywy recenzenta
SoftwareX. Najważniejsze braki to: reprodukowalność benchmarku, parametry SBM,
jawne ograniczenia metody oraz krótkie zakończenie.

## Wykonane

1. **Bibliografia Ray**
   - Wpis Ray został zmieniony na `@inproceedings` OSDI 2018.
   - Autorzy są skróceni przez `and others`, więc w bibliografii pojawia się
     `et al.`.
   - USENIX URL można zostawić; jest właściwy dla formalnej publikacji
     konferencyjnej.

2. **Cytowania w abstrakcie**
   - Abstrakt nie używa już numerycznych `\cite{...}`.
   - Krótkie cytowania Omnisolver i oryginalnego solvera CUDA są podane inline
     i są klikalne przez DOI.

3. **Acknowledgements**
   - Sekcja jest uzupełniona grantami NCN/PARP.

## Częściowo wykonane

4. **Impact**
   - Treściowo funkcję impact pełni paragraf `Purpose of an exhaustive solver in
     Omnisolver`.
   - Nadal nie ma osobnej sekcji `Impact`. Jeśli SoftwareX tego oczekuje w
     formularzu lub checklistach redakcyjnych, warto dodać krótki paragraf pod
     tym tytułem.

5. **Topologia i rozkład instancji**
   - Tekst mówi już o `all-to-all random Ising` oraz `uniformly distributed
     couplings and biases`.
   - Nadal brakuje precyzyjnego zapisu rozkładu, np. `J_{ij}, h_i \sim
     U[-1,1]`, oraz informacji, czy diagonalne wpisy z plików COO są traktowane
     jako biasy/pola.
   - Warto zsynchronizować opis z `code/bf.py`, gdzie instancje są generowane
     przez `2*(np.random.rand(d, d) - 0.5)` i zapisywane dla `i <= j`.

6. **Crossover single-GPU vs distributed**
   - Tekst wspomina, że dla mniejszych `N` dominuje overhead Ray/kerneli.
   - Nadal brakuje praktycznej wskazówki, kiedy wybrać `BruteforceGPUSampler`,
     a kiedy `DistributedBruteforceGPUSampler`.

7. **Wyjaśnienie `float32`**
   - Jest opis błędu roundoff i rekalkulacji energii w `float64`.
   - Nadal brakuje jednego zdania o trade-offie: `float32` jest używany na
     fast path ze względu na przepustowość/szybkość GPU, a stabilizacja i
     rekalkulacja ograniczają ryzyko numeryczne.

8. **Komenda benchmarkowa**
   - `code/README.md` zawiera komendy do uruchamiania `code/bf.py`.
   - W samym artykule nadal nie ma konkretnej komendy ani pełnej konfiguracji
     generującej `distributed.pdf`.
   - `code/plot_distributed.py` czyta wyniki z `../omni-bench/results`, więc
     ścieżka nie jest samowystarczalna względem repo artykułu.

## Niewykonane / otwarte

9. **Conclusions**
   - Brak sekcji `Conclusions`.
   - Warto dodać krótki akapit: co wnosi update, gdzie są obecne limity, co
     dalej planowane.

10. **Porównanie z oryginalnym single-GPU brute-force**
    - Brak konkretnej liczby lub zdania pokazującego relację do
      `jalowiecki2021brute`.
    - Minimum: jedno zdanie, czy nowa wersja zachowuje porównywalną wydajność
      single-GPU przy tych samych parametrach, albo jasne stwierdzenie, że
      artykuł nie benchmarkuje tej osi.

11. **Parametry SBM**
    - Sekcja `Precision of the reported energies` nadal podaje wynik porównania
      z SBM bez informacji o implementacji i hyperparametrach.
    - Do uzupełnienia: użyty solver/implementacja, liczba prób, liczba kroków,
      najważniejsze parametry dynamiki oraz czy raportowany wynik to single-run
      czy best-of-N.

12. **Limitations / known issues**
    - Brak osobnej sekcji lub akapitu o ograniczeniach.
    - Do uzupełnienia: maksymalne wspierane `N`, ograniczenia pamięci/buforów,
      ograniczenia `float32`, ograniczenia Ray/controller overhead oraz zakres,
      w którym projekcje skalowania są tylko back-of-the-envelope.

13. **Seedy i odtwarzalność instancji**
    - `code/bf.py` generuje instancje przez globalne `np.random.rand(...)` bez
      jawnego seeda.
    - Do uzupełnienia: argument `--seed`, zapis seeda w wynikach JSON oraz
      informacja w artykule, jakie seedy wygenerowały dane z Fig. 1.

14. **Dane benchmarkowe**
    - W repo artykułu nie ma widocznych `results/*.json` ani `instances/*.txt`
      użytych do wykresu.
    - Dla SoftwareX warto dołączyć dane lub wskazać trwałe archiwum/commit,
      z którego można je odtworzyć.

15. **Formalności software/release**
    - Manuskrypt deklaruje `v0.0.5` i instalację z PyPI. Przed submission trzeba
      upewnić się, że PyPI faktycznie zawiera tę wersję i zgodne wymagania
      Pythona.
    - C2 powinno najlepiej wskazywać trwały release/tag/Zenodo DOI, a nie tylko
      główny URL repozytorium.
    - Trzeba potwierdzić, że repo ma plik `LICENSE` zgodny z deklarowanym
      Apache License 2.0.

16. **Competing interests**
    - Obecna deklaracja mówi o braku competing interests.
    - Ze względu na afiliację Quantumz.io warto sprawdzić, czy zatrudnienie,
      własność udziałów lub finansowanie firmowe nie powinny być jawnie
      zadeklarowane według reguł Elseviera.

## Priorytet przed submission

**Wysoki**
- Parametry SBM.
- Seedy i dane benchmarkowe.
- Precyzyjny rozkład/topologia instancji.
- Limitations / known issues.
- Zgodność wersji PyPI/release/licencji z tabelą metadanych.

**Średni**
- Conclusions.
- Praktyczny próg single-GPU vs distributed.
- Jednozdaniowe porównanie z oryginalnym solverem CUDA.
- Doprecyzowanie trade-offu `float32`.

**Niski**
- Osobna sekcja `Impact`, jeśli redakcja jej wymaga wprost.
- Uporządkowanie ścieżek w `plot_distributed.py`, żeby repo artykułu było
  samowystarczalne.
