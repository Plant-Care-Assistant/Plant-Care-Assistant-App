# Zasady repozytorium


## Struktura projektu

Repozytorium jest monorepo i składa się z głównych modułów:

- frontend/   – UI aplikacji
- backend/    – logika aplikacji i API
- ai/         – moduł rozpoznawania roślin
- data/       – dane o roślinach
- infra/      – Docker, deployment, środowisko uruchomieniowe
- docs/       – dokumentacja techniczna
- scripts/    – narzędzia i skrypty pomocnicze


## Branching strategy

Pracujemy w systemie feature branches:

### Gałąź główna
- main – stabilna, nie commitujemy bezpośrednio
- Każda zmiana do main przechodzi przez Pull Request

### Gałęzie robocze

Twórz je według schematu:
- feature/[ID-zadania]-krotki-opis

Przykład:
- feature/PLNT-12-upload-photo

(ID-zadania pochodzi z Linear, można skopiować nazwę gałęzi w Linear, klikając "copy git branch name")


## Pull Requests

Każda zmiana do main wymaga PR.

Zasady PR:
- PR dotyczy jednego zadania / jednej funkcji

Wymagania przed merge:
- projekt buduje się lokalnie,
- linter/formatter przeszedł,
- testy przeszły,
- PR ma co najmniej 1 approve po review,


## Testy

Każdy moduł powinien mieć własny katalog testów:

frontend/tests/
backend/tests/
ai/tests/

Wymagania:
- PR zmieniający logikę musi dodać lub zaktualizować testy
- testy powinny przechodzić przed merge


## Styl commitów

Używamy spójnych typów commitów:

- FEAT:	nowa funkcjonalność
- FIX:	naprawa błędu
- ENH:	ulepszenie działania (nie feature)
- REF:	refaktoryzacja kodu
- DOC:	zmiana dokumentacji
- TEST:	dodanie lub zmiana testów
- CHORE:	zmiany techniczne, konfiguracja, CI, struktura repo


## Pre-commit hooks

W projekcie korzystamy z hooków pre-commit.

Każdy po sklonowaniu repo musi wykonać:
- pip install pre-commit

Ewentualnie (w przypadku problemów):
- python3 -m pip install pre-commit
- python3 -m pip show pre-commit
- echo 'export PATH="{lokalizacja-instalacji-pre-commit}" >> ~/.zshrc LUB ~/.bashrc
- source ~/.zshrc LUB ~/.bashrc

---

I na końcu w katalogu głównym repo:
- pre-commit install

Od tej pory:
- pre-commit uruchamia się automatycznie przy każdym commitcie
- Użyj pre-commit --all-files (jeśli chcesz sprawdzic pliki ręcznie)

Hooki:
- formatują kod,
- wykrywają błędy zanim trafią do PR,


## Dokumentacja

Wszelkie zmiany w logice aplikacji lub API muszą być odzwierciedlone w:
- docs/
