# Plant-Care-Assistant-App

## Test Coverage
| | |
| --- | --- |
| AI | ![Coverage](./ai/assets/unit-coverage.svg)
| Backend | ![Coverage](./backend/assets/unit-coverage.svg)

## Opis projektu grupowego – Plant Care Assistant

Projekt **„Plant Care Assistant”** to aplikacja webowa wspierająca użytkowników w rozpoznawaniu gatunków roślin oraz udzielaniu porad dotyczących ich pielęgnacji. Celem projektu jest stworzenie intuicyjnego narzędzia, które umożliwi identyfikację roślin domowych na podstawie zdjęcia oraz zaprezentuje najważniejsze informacje o ich wymaganiach i warunkach wzrostu.

**Osoby wchodzące w skład zespołu:**

- Oliwia Pawelec,
- Polina Nesterova,
- Julian Stefan,
- Mateusz Gołębiewski,
- Jakub Żebrowski,
- Jakub Wilczek.

### **Cele i założenia projektu**

Podstawowe cele projektu obejmują:

1. **Rozpoznawanie roślin na podstawie zdjęcia** –
   użytkownik będzie mógł przesłać zdjęcie rośliny, a aplikacja rozpozna jej gatunek.
   * Wymagane działania:
     * przygotowanie mechanizmu przesyłania i obsługi zdjęć,
     * opracowanie sposobu analizy obrazu i uzyskania wyniku rozpoznania,
     * przekazanie wyniku do dalszego przetwarzania w aplikacji.
2. **Wyświetlanie informacji o roślinie** –
   po rozpoznaniu gatunku użytkownik otrzyma opis rośliny oraz podstawowe porady pielęgnacyjne.
   * Wymagane działania:
     * opracowanie struktury danych z opisami roślin,
     * stworzenie mechanizmu wyszukiwania i prezentacji danych,
     * przygotowanie przejrzystego układu informacji w interfejsie.
3. **Prezentowanie optymalnych warunków wzrostu** –
   aplikacja ma umożliwiać sprawdzenie zalecanych parametrów środowiskowych, takich jak poziom nasłonecznienia, temperatura i wilgotność.
   * Wymagane działania:
     * zdefiniowanie danych dotyczących warunków wzrostu,
     * opracowanie sposobu ich prezentacji w aplikacji,
     * przygotowanie możliwości aktualizacji i rozszerzania informacji.

### **Główne obszary pracy**

Aplikacja składa się z kilku powiązanych elementów, które będą rozwijane równolegle i połączone w spójny system:

1. **Interfejs użytkownika**
   * Zaprojektowanie układu strony i widoków,
   * Stworzenie mechanizmu wczytywania zdjęcia i prezentowania wyników,
   * Zapewnienie przejrzystej prezentacji informacji o roślinie.
2. **Warstwa obsługi logiki aplikacji**
   * Przyjmowanie danych od użytkownika (zdjęcia),
   * Przekazywanie ich do odpowiednich modułów przetwarzania,
   * Odbieranie wyników i przygotowywanie ich do wyświetlenia.
3. **Zarządzanie danymi o roślinach**
   * Stworzenie uporządkowanego zbioru informacji o roślinach,
   * Umożliwienie wyszukiwania danych po nazwie lub gatunku.
4. **Moduł rozpoznawania roślin**
   * Przygotowanie sposobu analizy zdjęcia i identyfikacji gatunku,
   * Opracowanie mechanizmu przekazania wyniku do aplikacji.
5. **Integracja elementów aplikacji**
   * Połączenie interfejsu użytkownika z logiką obsługi danych i modułem rozpoznawania,
   * Ujednolicenie sposobu przekazywania i przetwarzania informacji.

### **Czynności niezbędne do połączenia elementów**

Aby aplikacja działała jako spójny system, konieczne będzie:

* zaprojektowanie sposobu wymiany danych między poszczególnymi częściami aplikacji,
* ustalenie wspólnych formatów przesyłanych informacji,
* wykonanie testów połączeń między modułami,
* uporządkowanie struktury projektu i kodu, aby umożliwić równoległą pracę w zespole.

### **Plan rozwoju projektu (rozszerzenia funkcjonalne)**

Po zrealizowaniu podstawowej wersji aplikacji planowane jest stopniowe rozszerzenie jej możliwości o dodatkowe funkcje zwiększające użyteczność i atrakcyjność systemu.

#### **1\. Rozpoznawanie chorób roślin**

* Rozbudowanie modułu analizy zdjęć o możliwość identyfikacji chorób na podstawie objawów widocznych na liściach lub łodygach.
* Dodanie opisów chorób, ich przyczyn i sposobów leczenia.
* Integracja wyników z modułem informacyjnym, aby użytkownik otrzymywał nie tylko nazwę choroby, ale też zalecenia dotyczące pielęgnacji.

#### **2\. Moduł monitorowania warunków środowiskowych**

* Stworzenie modułu monitorowania warunków jakie mają rośliny,
* Stworzenie połączenia z urządzeniem mierzącymi wilgotność gleby, temperaturę i natężenie światła.
* Zbieranie danych z czujników i prezentowanie ich w aplikacji w czasie rzeczywistym.
* Wprowadzenie alertów lub powiadomień o nieprawidłowych warunkach (np. zbyt suche podłoże).

#### **3\. Biblioteka roślin użytkownika**

* Wprowadzenie możliwości dodawania przez użytkownika własnych roślin do indywidualnej biblioteki.
* Przechowywanie zdjęć, nazw i krótkich notatek o każdej z roślin.
* Możliwość przeglądania historii i obserwacji postępów wzrostu.
* Zapisywanie danych z czujników – użytkownik może przeglądać dane historyczne oraz wykresy mierzonych parametrów.

#### **4\. Portal społecznościowy**

* Możliwość umieszczania zdjęć swoich roślin na tablicy wraz z opisem tekstowym.
* Komentowanie, reagowanie na zdjęcia innych użytkowników.
* Integracja z innymi mediami społecznościowymi (Facebook, Instagram).
* Moderacja i zarządzanie treściami użytkowników w celu zapewnienia bezpieczeństwa.

| Filary aplikacji | Zakres działań | Osoby | Typ pracy |
| ----- | ----- | ----- | ----- |
| Interfejs użytkownika | Projekt graficzny, formularze, prezentacja danych | Polina Nesterova | Frontend |
| Logika aplikacji | Obsługa danych, przepływ informacji, integracja | Julian Stefan, Jakub Żebrowski | Backend |
| Baza danych | Struktura danych, treści o roślinach | Jakub Wilczek, Jakub Żebrowski | Backend / Data |
| Rozpoznawanie roślin | Analiza zdjęć, przygotowanie wyników | Oliwia Pawelec Mateusz Gołębiewski | AI / Data Processing |
| Koordynacja i planowanie pracy | — | Mateusz Gołębiewski | — |
