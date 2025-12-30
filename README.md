# Plan eksperymentów do pracy inżynierskiej

**Tytuł pracy:** Zastosowanie sieci neuronowych i metod uczenia maszynowego do predykcji przeżywalności pacjentów z niewydolnością serca – reprodukcja i rozszerzenie badań

---

## 1. Wprowadzenie

Niewydolność serca stanowi jedno z najpoważniejszych wyzwań współczesnej medycyny, będąc przyczyną około 17,9 miliona zgonów rocznie na świecie. Choroba ta charakteryzuje się osłabieniem funkcji pompowania krwi przez serce, co prowadzi do niedostatecznego zaopatrzenia tkanek w tlen i substancje odżywcze. Wczesna i dokładna predykcja ryzyka zgonu u pacjentów z niewydolnością serca ma kluczowe znaczenie dla optymalizacji strategii terapeutycznych, alokacji zasobów medycznych oraz poprawy jakości życia pacjentów.

Niniejsza praca stanowi próbę reprodukcji oraz rozszerzenia badań przedstawionych w publikacji Mishry (2022), która przeprowadziła kompleksową analizę przeżycia i predykcję zgonu dla 299 pacjentów z zaawansowaną niewydolnością serca (klasa III/IV według klasyfikacji NYHA). Oryginalnie badanie wykorzystało metody analizy przeżycia (Kaplan-Meier, model Coksa) oraz klasyczne algorytmy uczenia maszynowego (SVM, Random Forest, XGBoost, LightGBM). Autorzy zidentyfikowali frakcję wyrzutową, poziom kreatyniny w surowicy oraz wiek jako najistotniejsze czynniki prognostyczne.

**Głównym celem pracy jest sprawdzenie czy sieć neuronowa jest w stanie przewyższyć stworzone w eksperymencie modele uczenia maszynowego**

## 2. Cel i zakres pracy

### 2.1. Cel główny

Celem głównym pracy jest kompleksowa ocena skuteczności różnych metod uczenia maszynowego, ze szczególnym uwzględnieniem sieci neuronowych, w predykcji zgonu pacjentów z niewydolnością serca, oraz porównanie uzyskanych wyników z metodami opisanymi w literaturze.

### 2.2. Cele szczegółowe

Realizacja celu głównego obejmuje następujące zadania:

1. Reprodukcję kluczowych eksperymentów z oryginalnej publikacji, w tym eksploracyjnej analizy danych, analizy przeżycia metodami klasycznych algorytmów ML.
2. Przeprowadzenie rozszerzonej inżynierii cech w celu potencjalnej poprawy jakości predykcji.
3. Zaprojektowanie, implementację i optymalizację modeli opartych na sieciach neuronowych (MLP oraz DeepSurv).
4. Systematyczne porównanie wyników uzyskanych przez nowe modele z wynikami modeli w literaturze.
5. Sformułowanie wniosków dotyczących wartości dodanej sieci neuronowych w analizowanym problemie klinicznym.

## 3. Zbiór danych

Analiza zostanie przeprowadzona na publicznie dostępnym zbiorze danych "Heart Failure Clinical Records", który zawiera 299 rekordów pacjentów z zaawansowaną niewydolnością serca. Zbiór składa się z 12 cech klinicznych oraz zmiennej celu (`DEATH_EVENT`), która wskazuje, czy pacjent zmarł w okresie obserwacji.

### 3.1. Charakterystyka zbioru danych

Kohortę pacjentów stanowiło 105 kobiet i 194 mężczyzn w wieku od 40 do 95 lat. Wszyscy pacjenci byli zdiagnozowani z dysfunkcją skurczową lewej komory i mieli wcześniejszą historię niewydolności serca, co skutkowało ich klasyfikacją do klasy III lub IV według klasyfikacji NYHA (New York Heart Association).

| Cecha | Opis | Typ | Zakres wartości |
| :--- | :--- | :--- | :--- |
| `age` | Wiek pacjenta | numeryczny | [40, ..., 95] lat |
| `anaemia` | Występowanie anemii | binarny | 0 (nie), 1 (tak) |
| `creatinine_phosphokinase` | Poziom kinazy kreatynowej (CPK) w krwi | numeryczny | [23, ..., 7861] mcg/L |
| `diabetes` | Występowanie cukrzycy | binarny | 0 (nie), 1 (tak) |
| `ejection_fraction` | Frakcja wyrzutowa serca | numeryczny | [14, ..., 80] % |
| `high_blood_pressure` | Występowanie nadciśnienia | binarny | 0 (nie), 1 (tak) |
| `platelets` | Liczba płytek krwi | numeryczny | [25.01, ..., 850.00] kiloplatelets/mL |
| `serum_creatinine` | Poziom kreatyniny w surowicy | numeryczny | [0.50, ..., 9.40] mg/dL |
| `serum_sodium` | Poziom sodu w surowicy | numeryczny | [114, ..., 148] mEq/L |
| `sex` | Płeć pacjenta | binarny | 0 (kobieta), 1 (mężczyzna) |
| `smoking` | Palenie papierosów | binarny | 0 (nie), 1 (tak) |
| `time` | Okres obserwacji pacjenta | numeryczny | [4, ..., 285] dni |
| `DEATH_EVENT` | Zgon pacjenta w okresie obserwacji | binarny | 0 (przeżył), 1 (zmarł) |

### 3.2. Rozkład zmiennej celu

W zbiorze danych 96 pacjentów (32,11%) zmarło w okresie obserwacji, podczas gdy 203 pacjentów (67,89%) przeżyło. Ten niezbalansowany rozkład klas będzie wymagał odpowiedniego uwzględnienia w procesie modelowania.

## 4. Metodologia badawcza

### 4.1. Etap 1: Reprodukcja badań bazowych

Pierwszym etapem pracy będzie odtworzenie środowiska i eksperymentów z repozytorium GitHub w celu weryfikacji wyników z publikacji.

#### 4.1.1. Reprodukcja eksploracyjnej analizy danych (EDA)

Zostanie odtworzona analiza z notebooka `Exploratory_Data_Analysis.ipynb`, obejmująca:

*   **Analizę jednowymiarową:** Wizualizacja rozkładów poszczególnych cech numerycznych (histogramy, wykresy gęstości) oraz analiza częstości dla cech binarnych (wykresy słupkowe). Pozwoli to na identyfikację potencjalnych wartości odstających oraz zrozumienie charakterystyki danych.
*   **Analizę dwuwymiarową:** Badanie korelacji pomiędzy cechami numerycznymi za pomocą macierzy korelacji i heatmapy w celu identyfikacji współliniowości. Analiza zależności między poszczególnymi cechami a zmienną celu `DEATH_EVENT` za pomocą wykresów pudełkowych i testów statystycznych (t-test dla cech ciągłych, chi-kwadrat dla cech kategorycznych).
*   **Weryfikację kluczowych obserwacji:** Potwierdzenie, że pacjenci, którzy zmarli, charakteryzują się wyższym poziomem kreatyniny w surowicy, niższą frakcją wyrzutową oraz niższym poziomem sodu w surowicy w porównaniu do pacjentów, którzy przeżyli.


#### 4.1.2. Reprodukcja modeli uczenia maszynowego

Zostanie odtworzony potok uczenia maszynowego z notebooka `Heart_Failure_Prediction.ipynb`:

*   **Wybór cech:** Zgodnie z publikacją, do modelowania zostaną wykorzystane cechy: `age`, `ejection_fraction`, `serum_creatinine` (cechy o najwyższej istotności statystycznej w modelu Coksa, z wyłączeniem `time`).
*   **Podział danych:** Zastosowanie randomized cross-validation w celu optymalizacji hiperparametrów i oceny stabilności modeli.
*   **Modele bazowe:** Implementacja i trening modeli: SVM, Decision Tree, Random Forest, XGBoost, LightGBM.
*   **Weryfikacja wyników:** Próba uzyskania zbliżonych wyników do tych przedstawionych w Tabeli 2 publikacji (SVM: F1=88.37, Accuracy=83.33%; LightGBM: F1=85.71, Accuracy=80.00%).

### 4.2. Etap 2: Rozszerzenie i nowe eksperymenty

Po pomyślnej reprodukcji wyników bazowych, praca skupi się na nowych eksperymentach, stanowiących oryginalny wkład badawczy.

#### 4.2.1. Inżynieria cech (Feature Engineering)

Oryginalna praca wykorzystywała głównie surowe cechy. W niniejszej pracy zostaną zaproponowane nowe cechy, które mogą poprawić jakość predykcji.

**a) Dyskretyzacja cech ciągłych**

Podział cech ciągłych na kategorie może pomóc modelom w uchwyceniu nieliniowych zależności, które nie są oczywiste w reprezentacji ciągłej.

*   **Age:** Podział na grupy wiekowe, np. [40-60], [60-80], [80-95], co odpowiada naturalnym przedziałom ryzyka klinicznego.
*   **Serum Creatinine:** Podział na kategorie w oparciu o progi kliniczne (np. norma: 0.5-1.2 mg/dL, podwyższony: 1.2-3.0 mg/dL, wysoki: >3.0 mg/dL).
*   **Ejection Fraction:** Podział na kategorie odpowiadające stopniom dysfunkcji (np. ciężka: <30%, umiarkowana: 30-45%, lekka/norma: >45%).

**Uzasadnienie:** Dyskretyzacja może być szczególnie korzystna dla modeli drzewiastych, które naturalnie operują na progach decyzyjnych. Może również pomóc w interpretacji wyników przez klinicystów, którzy często myślą w kategoriach przedziałów referencyjnych.

**b) Tworzenie cech interakcyjnych**

Interakcje między cechami mogą reprezentować złożone zjawiska fizjologiczne, których pojedyncze cechy nie oddają.

*   **Age × Serum Creatinine:** Interakcja wieku i poziomu kreatyniny może odzwierciedlać skumulowane ryzyko związane z wiekiem i funkcją nerek.
*   **Ejection Fraction × Serum Sodium:** Interakcja frakcji wyrzutowej i poziomu sodu może wskazywać na zaawansowanie niewydolności serca i zaburzenia elektrolitowe.
*   **Age × Ejection Fraction:** Interakcja wieku i frakcji wyrzutowej może reprezentować różnicę w rokowaniu między młodszymi i starszymi pacjentami o podobnej funkcji serca.

**Uzasadnienie:** Cechy interakcyjne mogą ujawnić synergiczne efekty, które nie są widoczne przy analizie pojedynczych zmiennych. Są szczególnie istotne w kontekście medycznym, gdzie wiele czynników działa łącznie.

**c) Normalizacja i standaryzacja**

Przeskalowanie cech numerycznych jest kluczowe dla prawidłowego działania wielu algorytmów.

*   **StandardScaler:** Standaryzacja do średniej 0 i odchylenia standardowego 1, zalecana dla SVM i sieci neuronowych.
*   **MinMaxScaler:** Normalizacja do zakresu [0, 1], alternatywna metoda, która może być korzystna dla niektórych architektur neuronowych.

**Uzasadnienie:** Sieci neuronowe są szczególnie wrażliwe na skalę cech wejściowych. Brak normalizacji może prowadzić do problemów ze zbieżnością podczas treningu oraz dominacji cech o większych wartościach bezwzględnych.

#### 4.2.2. Główne podejście badawcze: Sieci neuronowe

Jako alternatywa dla klasycznych modeli ML, zbadane zostaną głębokie sieci neuronowe. Sieci te mają potencjał do modelowania bardzo złożonych, nieliniowych zależności w danych.

**a) Wielowarstwowy perceptron (MLP)**

MLP jest podstawową architekturą sieci neuronowej typu feed-forward, składającą się z warstw wejściowej, ukrytych i wyjściowej. Każdy neuron w warstwie ukrytej oblicza ważoną sumę wejść, a następnie przekształca ją za pomocą nieliniowej funkcji aktywacji.

**Uzasadnienie wyboru MLP:** MLP jest uniwersalnym aproksymatorem funkcji, co oznacza, że przy odpowiedniej liczbie neuronów może aproksymować dowolną funkcję ciągłą. W kontekście predykcji medycznej, MLP może odkryć ukryte, nieliniowe wzorce w danych klinicznych, które są niedostępne dla modeli liniowych czy prostych drzew decyzyjnych.

**b) DeepSurv – sieć neuronowa do analizy przeżycia**

DeepSurv to model głębokiego uczenia zaprojektowany specjalnie do analizy przeżycia. Jest to wariant sieci neuronowej, który bezpośrednio modeluje funkcję hazardu (ryzyko zdarzenia w czasie), stanowiąc nieliniowe rozszerzenie modelu proporcjonalnych hazardów Coksa.

**Uzasadnienie wyboru DeepSurv:** Podczas gdy MLP przewiduje binarny wynik (zgon/przeżycie), DeepSurv uwzględnia wymiar czasowy i modeluje ryzyko w funkcji czasu. Pozwala to na bardziej precyzyjną predykcję, uwzględniającą moment wystąpienia zdarzenia oraz dane cenzurowane (pacjenci, którzy opuścili badanie lub nie doświadczyli zdarzenia w okresie obserwacji). DeepSurv może lepiej oddać złożoność zależności czasowych w niewydolności serca.

#### 4.2.3. Eksperymenty z optymalizacją sieci neuronowych

Skuteczność sieci neuronowych zależy w dużym stopniu od doboru architektury i hiperparametrów. Przeprowadzona zostanie seria systematycznych eksperymentów w celu znalezienia optymalnej konfiguracji.

**a) Architektura sieci**

Testowanie różnej liczby warstw ukrytych oraz liczby neuronów w każdej warstwie.

*   **Warianty do przetestowania:**
    *   Płytka sieć: 1 warstwa ukryta (np. 32, 64, 128 neuronów)
    *   Średnia sieć: 2 warstwy ukryte (np. [64, 32], [128, 64], [256, 128])
    *   Głęboka sieć: 3 warstwy ukryte (np. [128, 64, 32], [256, 128, 64])

**Uzasadnienie:** Liczba warstw i neuronów determinuje zdolność sieci do modelowania złożonych zależności. Zbyt mała sieć może nie uchwycić wszystkich wzorców (underfitting), podczas gdy zbyt duża może przeuczyć się do danych treningowych (overfitting). Systematyczne testowanie pozwoli znaleźć optymalny kompromis.

**b) Funkcje aktywacji**

Porównanie różnych funkcji aktywacji w warstwach ukrytych.

*   **ReLU (Rectified Linear Unit):** f(x) = max(0, x). Najpopularniejsza funkcja, szybka w obliczeniach, ale może cierpieć na problem "umierających neuronów".
*   **LeakyReLU:** f(x) = max(αx, x), gdzie α jest małą stałą (np. 0.01). Rozwiązuje problem umierających neuronów ReLU.
*   **ELU (Exponential Linear Unit):** f(x) = x jeśli x > 0, α(e^x - 1) w przeciwnym razie. Może przyspieszyć zbieżność i poprawić wyniki.

**Uzasadnienie:** Funkcja aktywacji wprowadza nieliniowość do sieci, co jest kluczowe dla jej zdolności do modelowania złożonych zależności. Różne funkcje mają różne właściwości w kontekście szybkości uczenia i stabilności gradientów.

**c) Techniki regularyzacji**

Zastosowanie metod zapobiegających przeuczeniu.

*   **Dropout:** Losowe wyłączanie neuronów podczas treningu z określonym prawdopodobieństwem (np. 0.2, 0.3, 0.5). Zmusza sieć do uczenia się bardziej rozproszonych reprezentacji.
*   **Regularyzacja L1/L2:** Dodawanie kary do funkcji kosztu za duże wartości wag. L1 promuje rzadkość (zerowanie niektórych wag), L2 promuje małe wartości wszystkich wag.
*   **Early Stopping:** Zatrzymanie treningu, gdy wydajność na zbiorze walidacyjnym przestaje się poprawiać, aby uniknąć przeuczenia.

**Uzasadnienie:** Regularyzacja jest kluczowa dla małych zbiorów danych (jak w tym przypadku, 299 próbek), gdzie ryzyko przeuczenia jest wysokie. Dropout i regularyzacja L1/L2 zmuszają model do generalizacji, a early stopping zapobiega nadmiernemu dopasowaniu do danych treningowych.

**d) Optymalizatory**

Porównanie różnych algorytmów optymalizacji.

*   **Adam (Adaptive Moment Estimation):** Adaptacyjny algorytm, który łączy zalety RMSprop i momentum. Często wybierany jako domyślny optymalizator.
*   **SGD z momentum:** Klasyczny algorytm gradientu prostego z dodanym momentum, który przyspiesza zbieżność.
*   **RMSprop:** Adaptacyjny algorytm, który dostosowuje współczynnik uczenia dla każdego parametru.

**Uzasadnienie:** Wybór optymalizatora wpływa na szybkość i stabilność procesu uczenia. Adam jest często skuteczny w praktyce, ale dla niektórych problemów SGD z momentum może dawać lepsze wyniki końcowe.

**e) Strojenie hiperparametrów**

Wykorzystanie zautomatyzowanych technik do znalezienia optymalnej kombinacji hiperparametrów.

*   **Grid Search:** Systematyczne przeszukiwanie zdefiniowanej siatki wartości hiperparametrów.
*   **Random Search:** Losowe próbkowanie z przestrzeni hiperparametrów, często bardziej efektywne niż Grid Search.
*   **Bayesian Optimization:** Zaawansowana metoda, która modeluje funkcję celu i inteligentnie wybiera kolejne punkty do przetestowania.

**Uzasadnienie:** Ręczne strojenie hiperparametrów jest czasochłonne i nieefektywne. Zautomatyzowane metody pozwalają na systematyczne eksplorowanie przestrzeni hiperparametrów i znalezienie optymalnej konfiguracji.

### 4.3. Etap 3: Porównanie i ewaluacja

Wszystkie modele zostaną ocenione przy użyciu tych samych metryk i procedur walidacyjnych, aby zapewnić porównywalność wyników.

#### 4.3.1. Metryki oceny

W celu zapewnienia porównywalności z pracą bazową [1], zostaną użyte następujące metryki:

*   **Accuracy (Trafność):** Ogólny procent poprawnych klasyfikacji. Metryka podstawowa, ale może być myląca przy niezbalansowanych klasach.
*   **Precision (Precyzja):** Odsetek pozytywnych predykcji, które były prawidłowe. Istotna w kontekście klinicznym, gdy chcemy minimalizować fałszywe alarmy.
*   **Recall (Czułość):** Odsetek faktycznych przypadków pozytywnych (zgonów), które zostały poprawnie zidentyfikowane. Kluczowa w medycynie, gdy chcemy wykryć wszystkie przypadki wysokiego ryzyka.
*   **F1-Score:** Średnia harmoniczna precyzji i czułości. Dobrze balansuje oba aspekty, szczególnie przy niezbalansowanych klasach.
*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Pole pod krzywą ROC, metryka oceniająca zdolność modelu do rozróżniania między klasami niezależnie od progu decyzyjnego.
*   **AUC-PR (Area Under the Precision-Recall Curve):** Pole pod krzywą precyzja-czułość, szczególnie przydatne przy niezbalansowanych klasach.

Dodatkowo, dla modeli analizy przeżycia (model Coksa, DeepSurv):

*   **C-index (Concordance Index):** Wskaźnik konkordancji, który mierzy, jak dobrze model porządkuje czasy przeżycia pacjentów. Wartość 0.5 oznacza losowe przewidywania, 1.0 oznacza idealne przewidywania.

#### 4.3.2. Procedura walidacji

*   **Stratified K-Fold Cross-Validation:** Zastosowanie k-krotnej walidacji krzyżowej ze stratyfikacją (zachowanie proporcji klas w każdym foldzie) w celu zapewnienia stabilnej oceny modeli.
*   **Test set holdout:** Opcjonalnie, wydzielenie części danych jako zbioru testowego, który nie będzie używany podczas treningu ani strojenia hiperparametrów, aby uzyskać bezstronną ocenę końcową.

#### 4.3.3. Analiza wyników

*   **Porównanie z baseline:** Zestawienie wyników nowych modeli (MLP, DeepSurv) z wynikami modeli bazowych (SVM, Random Forest, XGBoost, LightGBM) z etapu reprodukcji.
*   **Analiza istotności statystycznej:** Przeprowadzenie testów statystycznych (np. test t-Studenta dla par) w celu oceny, czy różnice w wynikach są istotne statystycznie.
*   **Analiza interpretowalności:** Wykorzystanie technik takich jak SHAP (SHapley Additive exPlanations) do wyjaśnienia predykcji modeli neuronowych i identyfikacji najważniejszych cech.

## 5. Oczekiwane rezultaty i wkład pracy

### 5.1. Rezultaty naukowe

Oczekuje się, że praca dostarczy:

1. **Weryfikację wyników publikacji bazowej:** Potwierdzenie (lub zakwestionowanie) wyników przedstawionych w [1] poprzez niezależną reprodukcję eksperymentów.
2. **Kompleksową ocenę sieci neuronowych:** Szczegółową analizę skuteczności MLP i DeepSurv w porównaniu do klasycznych metod ML w kontekście predykcji niewydolności serca.
3. **Rekomendacje metodologiczne:** Wnioski dotyczące optymalnej architektury, hiperparametrów i technik regularyzacji dla sieci neuronowych w tym problemie.
4. **Analizę interpretowalności:** Identyfikację najważniejszych cech klinicznych z perspektywy modeli neuronowych, co może dostarczyć nowych insights klinicznych.

### 5.2. Wkład praktyczny

Wyniki pracy mogą mieć praktyczne zastosowanie w systemach wspomagania decyzji klinicznych, pomagając lekarzom w:

*   Identyfikacji pacjentów wysokiego ryzyka wymagających intensywniejszego monitorowania.
*   Optymalizacji strategii terapeutycznych w oparciu o indywidualne profile ryzyka.
*   Lepszej alokacji zasobów medycznych poprzez priorytetyzację pacjentów.

### 5.3. Ograniczenia i kierunki przyszłych badań

Praca będzie miała pewne ograniczenia, które należy uwzględnić:

*   **Mały zbiór danych:** 299 próbek to relatywnie mała liczba dla treningu głębokich sieci neuronowych. Wyniki mogą być mniej stabilne niż w przypadku większych zbiorów.
*   **Brak walidacji zewnętrznej:** Modele będą trenowane i testowane na tym samym zbiorze danych. Idealna walidacja wymagałaby testowania na niezależnym zbiorze z innej populacji.
*   **Specyficzna populacja:** Pacjenci w zbiorze to wyłącznie osoby z zaawansowaną niewydolnością serca (klasa III/IV NYHA). Wyniki mogą nie generalizować się na pacjentów z lżejszymi postaciami choroby.
