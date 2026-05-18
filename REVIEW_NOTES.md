# Review notes — Omnisolver bruteforce (Version 2)

Aktualna checklista po weryfikacji `bruteforce.tex`, `bruteforce.bib`,
`code/bf.py`, `code/README.md` i `code/plot_distributed.py`.

## Status ogólny

PDF składa się poprawnie przez `latexmk`. Bibliografia jest technicznie w
dobrym stanie: wszystkie cytowane klucze istnieją, Ray jest cytowany jako
formalna publikacja OSDI 2018, a wpisy z ponad sześcioma autorami renderują się
jako `et al.`.

Manuskrypt został istotnie rozwinięty od poprzedniej wersji notatek: ma już
klikane cytowania inline w abstrakcie, sekcję ograniczeń oraz sekcję
`Conclusions and outlook`. Główne otwarte ryzyka recenzenckie to teraz:
reprodukowalność benchmarku, parametry SBM, precyzyjny opis generowania
instancji oraz formalności release/PyPI/licencja.

## Wykonane

1. **Bibliografia Ray**
   - Wpis Ray jest `@inproceedings` dla OSDI 2018.
   - Autorzy są skróceni przez `and others`, więc w bibliografii pojawia się
     `et al.`.
   - USENIX URL jest akceptowalny dla formalnej publikacji konferencyjnej.

2. **Cytowania w abstrakcie**
   - Abstrakt nie używa numerycznych `\cite{...}`.
   - Krótkie cytowania Omnisolver i oryginalnego solvera CUDA są podane inline
     i są klikalne przez DOI.

3. **Acknowledgements**
   - Sekcja jest uzupełniona grantami NCN/PARP.

4. **Conclusions**
   - Dodano sekcję `Conclusions and outlook`.
   - Sekcja podsumowuje wkład update'u, wynik do `N=60` i naturalne kierunki
     dalszych prac.

5. **Limitations / intended scope**
   - Dodano paragraf `Limitations and intended scope`.
   - Paragraf jasno mówi, że exhaustive enumeration pozostaje `O(2^N)`, że
     solver jest backendem certyfikacyjnym, oraz że czasy są zależne od
     konkretnego hardware/software stack.

## Częściowo wykonane

6. **Impact**
   - Treściowo funkcję impact pełni paragraf `Purpose of an exhaustive solver in
     Omnisolver`.
   - Nadal nie ma osobnej sekcji `Impact`. Jeśli SoftwareX tego oczekuje wprost
     w formularzu lub checklistach redakcyjnych, warto dodać krótki paragraf
     pod tym tytułem.

7. **Topologia i rozkład instancji**
   - Tekst mówi o `all-to-all random Ising` oraz `uniformly distributed
     couplings and biases`.
   - Nadal brakuje precyzyjnego zapisu rozkładu, np. `J_{ij}, h_i \sim
     U[-1,1]`, oraz wyjaśnienia, że diagonalne wpisy z plików COO odpowiadają
     biasom/polom, jeśli tak właśnie są interpretowane.
   - Opis warto zsynchronizować z `code/bf.py`, gdzie instancje są generowane
     przez `2*(np.random.rand(d, d) - 0.5)` i zapisywane dla `i <= j`.

8. **Crossover single-GPU vs distributed**
   - Tekst wspomina, że dla mniejszych `N` dominuje overhead Ray/kerneli.
   - Nadal brakuje praktycznej wskazówki, kiedy wybrać `BruteforceGPUSampler`,
     a kiedy `DistributedBruteforceGPUSampler`.

9. **Wyjaśnienie `float32`**
   - Jest opis błędu roundoff, stabilizacji i rekalkulacji energii w `float64`.
   - Nadal warto dodać jedno zdanie o trade-offie: `float32` jest używany na
     fast path ze względu na przepustowość/szybkość GPU, a stabilizacja i
     rekalkulacja ograniczają ryzyko numeryczne.

10. **Komenda benchmarkowa**
    - `code/README.md` zawiera komendy do uruchamiania `code/bf.py`.
    - W samym artykule nadal nie ma konkretnej komendy ani pełnej konfiguracji
      generującej `distributed.pdf`.
    - `code/plot_distributed.py` nadal czyta wyniki z `../omni-bench/results`,
      więc repo artykułu nie jest samowystarczalne.

11. **Limitations — doprecyzowanie techniczne**
    - Istnieje już ogólny paragraf ograniczeń.
    - Nadal brakuje twardych limitów: maksymalne wspierane `N`, ograniczenia
      pamięci/buforów, ograniczenia `num_fixed_vars`, oraz dokładniejszy zakres,
      w którym projekcje skalowania są tylko back-of-the-envelope.

## Niewykonane / otwarte

12. **Porównanie z oryginalnym single-GPU brute-force**
    - Brak konkretnej liczby lub zdania pokazującego relację do
      `jalowiecki2021brute`.
    - Minimum: jedno zdanie, czy nowa wersja zachowuje porównywalną wydajność
      single-GPU przy tych samych parametrach, albo jasne stwierdzenie, że
      artykuł nie benchmarkuje tej osi.

13. **Parametry SBM**
    - Sekcja `Precision of the reported energies` nadal podaje wynik porównania
      z SBM bez informacji o implementacji i hyperparametrach.
    - Do uzupełnienia: użyty solver/implementacja, liczba prób, liczba kroków,
      najważniejsze parametry dynamiki oraz czy raportowany wynik to single-run
      czy best-of-N.

14. **Seedy i odtwarzalność instancji**
    - `code/bf.py` generuje instancje przez globalne `np.random.rand(...)` bez
      jawnego seeda.
    - Do uzupełnienia: argument `--seed`, zapis seeda w wynikach JSON oraz
      informacja w artykule, jakie seedy wygenerowały dane z Fig. 1.

15. **Dane benchmarkowe**
    - W repo artykułu nie ma widocznych `results/*.json` ani `instances/*.txt`
      użytych do wykresu.
    - Dla SoftwareX warto dołączyć dane albo wskazać trwałe archiwum/commit,
      z którego można je odtworzyć.

16. **Formalności software/release**
    - Manuskrypt deklaruje `v0.0.5` i instalację z PyPI. Przed submission trzeba
      upewnić się, że PyPI faktycznie zawiera tę wersję i zgodne wymagania
      Pythona.
    - C2 powinno najlepiej wskazywać trwały release/tag/Zenodo DOI, a nie tylko
      główny URL repozytorium.
    - Trzeba potwierdzić, że repo ma plik `LICENSE` zgodny z deklarowanym
      Apache License 2.0.

17. **Competing interests**
    - Obecna deklaracja mówi o braku competing interests.
    - Ze względu na afiliację Quantumz.io warto sprawdzić, czy zatrudnienie,
      własność udziałów lub finansowanie firmowe nie powinny być jawnie
      zadeklarowane według reguł Elseviera.

## Priorytet przed submission

**Wysoki**
- Parametry SBM.
- Seedy i dane benchmarkowe.
- Precyzyjny rozkład/topologia instancji.
- Zgodność wersji PyPI/release/licencji z tabelą metadanych.

**Średni**
- Twarde limity metody w sekcji ograniczeń.
- Praktyczny próg single-GPU vs distributed.
- Jednozdaniowe porównanie z oryginalnym solverem CUDA.
- Doprecyzowanie trade-offu `float32`.

**Niski**
- Osobna sekcja `Impact`, jeśli redakcja jej wymaga wprost.
- Uporządkowanie ścieżek w `plot_distributed.py`, żeby repo artykułu było
  samowystarczalne.
