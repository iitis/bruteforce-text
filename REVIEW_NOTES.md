# Review notes — Omnisolver bruteforce (Version 3)

Aktualna checklista po ponownej weryfikacji `bruteforce.tex`,
`bruteforce.bib`, `code/bf.py`, `code/README.md`,
`code/plot_distributed.py` oraz metadanych PyPI/GitHub.

## Status ogólny

PDF składa się poprawnie przez `latexmk`. Bibliografia jest technicznie w
dobrym stanie: wszystkie cytowane klucze istnieją, Ray jest cytowany jako
formalna publikacja OSDI 2018, a wpisy z ponad sześcioma autorami renderują się
jako `et al.`. Abstrakt nie używa numerycznych cytowań i zawiera klikalne
odnośniki DOI.

Większość wcześniejszych uwag merytorycznych została wykonana: artykuł ma
opis progu single-GPU/distributed, relację do poprzedniego solvera, dokładny
rozkład instancji, seed benchmarku, komendę reprodukcyjną, parametry SBM,
twarde limity metody, sekcję `Conclusions and outlook`, pełne
`Acknowledgements` oraz poprawione metadane C6 z zależnością `ray`.

Główne pozostałe ryzyka przed submission: deklaracja competing interests jest
znów zbyt ogólna, PyPI nadal nie zgadza się z deklarowaną wersją `0.0.5`, w
repo kodu nie widać rozpoznawalnego pliku `LICENSE`, a dane benchmarkowe i
reproducibility capsule nadal powinny dostać trwały release/archiwum.

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

3. **Conclusions**
   - Dodano sekcję `Conclusions and outlook`.
   - Sekcja podsumowuje wkład update'u, wynik do `N=60` i naturalne kierunki
     dalszych prac.

4. **Limitations / intended scope**
   - Dodano paragraf `Limitations and intended scope`.
   - Paragraf wskazuje, że exhaustive enumeration pozostaje `O(2^N)`, że solver
     jest backendem certyfikacyjnym, oraz że czasy są zależne od hardware i
     software stack.
   - Dodano twarde limity: `N <= 64`, ograniczenie `suffix_size` przez
     working set/L2 oraz zastrzeżenie, że projekcje dla `N > 60` są
     back-of-the-envelope.

5. **Topologia i rozkład instancji**
   - Tekst precyzuje `all-to-all random Ising` z `J_{ij}, h_i \sim
     U[-1,1]` i.i.d.
   - W opisie figury podano, że każdy punkt to pojedynczy solver run na jednej
     ustalonej instancji dla danego `N`.

6. **Seedy i odtwarzalność instancji**
   - `code/bf.py` ma argument `--seed` z domyślną wartością `42`.
   - Instancje są generowane przez `np.random.default_rng(seed)`.
   - Seed jest zapisywany w JSON jako `instance_seed`.
   - Artykuł podaje komendę benchmarkową z `--seed 42`.

7. **Komenda benchmarkowa**
   - Artykuł zawiera komendy startu Ray oraz komendę benchmark sweep używaną do
     Fig. 1.
   - Komenda wskazuje `python code/bf.py --start 40 --stop 60 --step 2
     --sampler-mode distributed --seed 42 --skip-existing`.

8. **Crossover single-GPU vs distributed**
   - Dodano praktyczną wskazówkę: single GPU jest właściwe, dopóki jedna karta
     mieści problem i czas jest rozsądny; distributed sampler amortyzuje pracę
     przez liczbę workerów.
   - Tekst wskazuje break-even region około `N ≈ 46` na opisywanym sprzęcie.

9. **Wyjaśnienie `float32`**
   - Dodano zdanie, że fast path `num_states=1` działa w `float32`, aby
     maksymalizować GPU throughput, a stabilizacja kompensuje wynikający
     roundoff.

10. **Relacja do poprzedniego single-GPU solvera**
    - Dodano osobny paragraf `Relation to the predecessor single-GPU solver`.
    - Tekst mówi, że update nie re-benchmarkuje single-device axis, ale kernel
      jest API-compatible i dziedziczy performance envelope poprzednika.

11. **Parametry SBM**
    - Sekcja precyzji podaje: chaotic-variant simulated bifurcation solver,
      single H100 GPU, `2^12 = 4096` równoległych replik, `3000` integration
      steps i raportowanie najniższej energii repliki.
    - Tekst wskazuje companion repo `euro-hpc-pl/omni-bench` z Julia driver,
      parametrami SBM oraz tabelami `bf_velox_verification_*.csv`.

12. **Acknowledgements**
    - Sekcja jest kompletna i zawiera granty NCN oraz PARP.
    - Poprawiono drobną interpunkcję w nazwie `Quantumz.io Sp. z o.o.`.

13. **Permanent link / release tag**
    - C2 wskazuje konkretny release:
      `https://github.com/euro-hpc-pl/omnisolver-bruteforce/releases/tag/0.0.5`.
    - Tag `0.0.5` istnieje na GitHub.

14. **C6 dependencies**
    - C6 zawiera teraz `ray`, zgodnie z opisem distributed samplera i komendami
      uruchomieniowymi.

## Częściowo wykonane

15. **Impact**
    - Treściowo funkcję impact pełni paragraf `Purpose of an exhaustive solver
      in Omnisolver`.
    - Nadal nie ma osobnej sekcji `Impact`. Jeśli SoftwareX wymaga jej wprost
      w formularzu lub checklistach redakcyjnych, warto dodać krótki paragraf
      pod tym tytułem.

16. **Dane benchmarkowe**
    - Artykuł wskazuje companion repo `euro-hpc-pl/omni-bench`, które ma
      zawierać `instances/*.txt`, `results/*.json` i tabele weryfikacyjne.
    - Nadal warto upewnić się, że te dane są dostępne w publicznym, trwałym
      release/tagu lub archiwum DOI, a nie tylko w ruchomym branchu repo.
    - `code/plot_distributed.py` w repo artykułu nadal czyta z
      `../omni-bench/results`, więc samo repo artykułu nie jest w pełni
      samowystarczalne.

17. **Formalności software/release**
    - Release tag `0.0.5` jest wskazany w C2 i istnieje na GitHub.
    - PyPI nadal pokazuje `omnisolver-bruteforce` w wersji `0.0.3` oraz
      `requires_python <3.10, >=3.7`, co jest niespójne z manuskryptem
      deklarującym `0.0.5`, `pip install omnisolver-bruteforce` i
      `Python >= 3.10`.
    - GitHub API dla tagu `0.0.5` nie znajduje rozpoznawalnego pliku `LICENSE`
      (`license` endpoint zwraca `404`), mimo że tabela deklaruje Apache
      License 2.0.

18. **Dokumentacja C7**
    - C7 wskazuje ogólne dokumenty Omnisolver:
      `https://euro-hpc-pl.github.io/omnisolver/`.
    - To może być akceptowalne, ale jeśli plugin ma własną dokumentację w
      release `0.0.5`, bezpieczniej podać link bezpośrednio do niej.

## Niewykonane / otwarte

19. **Competing interests**
    - Aktualny tekst brzmi: `The Authors declare that they have no known
      competing financial interests.`
    - To jest słabsze niż wcześniejsza jawna deklaracja i może być ryzykowne,
      bo część autorów ma afiliację `Quantumz.io Sp. z o.o.`, a w pracy
      pojawia się SBM jako metoda porównawcza/cross-checking.
    - Do decyzji autorów: albo przywrócić rozszerzoną deklarację, albo
      potwierdzić, że obecna krótka deklaracja jest zgodna z polityką
      czasopisma i stanem faktycznym.

20. **Formalny reproducibility capsule / archival DOI**
    - Conclusions zapowiadają publikację wersjonowanej reproducibility capsule
      jako przyszły krok.
    - Przed submission warto zdecydować, czy obecne companion repo wystarcza,
      czy potrzebny jest Zenodo/Software Heritage DOI dla danych i benchmarków.

## Priorytet przed submission

**Wysoki**
- Rozstrzygnąć i ewentualnie rozszerzyć `Declaration of competing interest`.
- Doprowadzić PyPI do wersji `0.0.5` albo złagodzić claim o instalacji z PyPI.
- Dodać/zweryfikować plik `LICENSE` w repo kodu.
- Zapewnić trwały release/archiwum danych benchmarkowych i companion repo.

**Średni**
- Osobna sekcja `Impact`, jeśli redakcja jej wymaga wprost.
- Uporządkowanie ścieżek w `plot_distributed.py`, żeby repo artykułu było
  samowystarczalne albo jawnie zależne od companion repo.
- Rozważyć bezpośredni link C7 do dokumentacji pluginu, jeśli taka istnieje.

**Niski**
- Rozważyć dodanie krótkiego zdania w metadanych lub Usage, że `0.0.5` jest
  instalowane z GitHub release, jeśli PyPI nie zostanie zaktualizowane przed
  submission.
