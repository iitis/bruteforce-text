# Review notes — Omnisolver bruteforce (Version 2)

Uwagi merytoryczne do publikacji `bruteforce.tex` przed submission do SoftwareX.

## Strukturalne braki względem formatu SoftwareX

1. **Brak sekcji "Impact"** — SoftwareX zwykle wymaga sekcji opisującej, *jak* to oprogramowanie zmienia praktykę badawczą. Można dodać jeden krótki paragraf po §Performance: kto skorzysta, co teraz jest możliwe a wcześniej nie było (np. certyfikowane benchmarki dla claims o quantum advantage).

2. **Brak sekcji "Conclusions"** — krótkie podsumowanie wkładu i wskazanie kierunku przyszłych prac (np. integracja z innymi solverami, więcej topologii grafów).

## Brakujące dane techniczne (recenzent na pewno zapyta)

3. **Topologia instancji** — "dense random Ising", ale jaki rozkład \(J_{ij}\)? Gaussowski \(\mathcal{N}(0,1)\)? ±1? Uniform \([-1,1]\)? Pola \(h_i\) zerowe czy losowe? Bez tego rysunku nie da się odtworzyć.

4. **Porównanie z oryginalnym single-GPU brute-force** (`jalowiecki2021brute`) — claim "promotes ... to first-class component" wymaga pokazania, że nowa wersja jest co najmniej tak szybka jak oryginał na równym N. Wystarczy jedno zdanie z liczbą.

5. **Parametry uruchomienia SBM** w §Precision — który solver (Toshiba SBM API? własna implementacja? OpenJij?), jakie hyperparams (liczba kroków, dt, pump rate), single-run czy best-of-N. Bez tego porównanie "SBM osiąga ground state" jest niepowtarzalne.

6. **Crossover single-GPU vs distributed** — §Architecture wspomina dwa samplery, ale nie mówi, kiedy używać którego. Dla jakiego N opłaca się Ray? Praktyczna wskazówka dla użytkownika.

## Brakuje "Limitations" / "Known issues"

7. **Górne ograniczenia rozmiaru** — kernel 64-bit-safe ⇒ N ≤ 64? Czy mniej (ograniczenia pamięci buffer'a, liczba spinów per GPU)? Warto wyznaczyć granice eksplicytnie.

8. **Dlaczego tylko `float32` na fast path** — jedno zdanie wyjaśniające trade-off (przepustowość vs precyzja); użytkownik może się dziwić.

## Reprodukowalność

9. **§Performance odnosi się do "benchmark script in C2"**, ale nie podaje konkretnego pliku/komendy. Recenzent SoftwareX prawdopodobnie spróbuje uruchomić. Warto wskazać dokładnie, np. `scripts/benchmark_distributed.py --N 58 --seed 42`.

10. **Seedy losowe** — czy konkretny zestaw N=46..58 pochodzi z jednego seeda? Wielu? Bez tej informacji ktoś replikujący dostanie inne czasy.

## Uwaga do bibliografii

Wpis Ray został zmieniony na `@inproceedings` OSDI 2018 z polem `url=` (USENIX), bez DOI. Ponieważ inne wpisy mają DOI, ten jeden ponownie pokaże osobny URL — wraca dawna niespójność. Opcje:

- dodać DOI arXiv `10.48550/arXiv.1712.05889` i usunąć `url` (zachowując OSDI inproceedings);
- lub zostawić jak jest (OSDI 2018 jest właściwym cytowaniem konferencji, USENIX URL to standard).

## Priorytet

**Wysoki (blokery dla recenzji):**
- (1) Impact
- (3) Topologia instancji
- (5) Parametry SBM
- (9–10) Reprodukowalność (skrypt, seedy)

**Średni (podnosi jakość, nie jest blokerem):**
- (2) Conclusions
- (4) Porównanie z oryginalnym brute-force
- (6) Crossover single-GPU vs distributed
- (7) Górne ograniczenia rozmiaru
- (8) Wyjaśnienie wyboru float32
