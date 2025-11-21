# Informacje

| | |
| --- | --- |
| Wersja | ![Version](https://img.shields.io/badge/version-0.0.1-blue)
| Pokrycie testów | ![Coverage](./assets/unit-coverage.svg)


# Konfiguracja

## Budowanie obrazów Dockera

Wszystkie poniższe polecenia zakładają, że jesteś w katalogu `ai/`.

### Budowanie i uruchomienie (CPU)

Budowa obrazu CPU:
```
docker build -f docker/cpu.Dockerfile -t plant-care-ai:cpu .
```
Uruchom kontener CPU interaktywnie:
```
docker run --rm -it plant-care-ai:cpu bash
```


### Budowanie i uruchomienie (GPU)
Budowa obrazu GPU (wymaga NVIDIA Container Toolkit i sterowników na hoście):
```
docker build -f docker/gpu.Dockerfile -t plant-care-ai:gpu .
```
Uruchom kontener GPU interaktywnie (dołącz GPU poprzez `--gpus all`):
```
docker run --rm -it --gpus all plant-care-ai:gpu bash
```
