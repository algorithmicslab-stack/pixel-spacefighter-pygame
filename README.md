# Pixel Spacefighter (versión Pygame)

Copia en Python + Pygame de [Pixel Spacefighter](https://github.com/algorithmicslab-stack/pixel-spacefighter): juego de plataformas y peleas en 3 sectores espaciales.

## Jugar en el navegador

https://algorithmicslab-stack.github.io/pixel-spacefighter-pygame/ (haz clic en la pantalla para empezar)

## Cómo jugar en tu computadora

```bash
pip install pygame numpy
python juego.py
```

El piloto se lee de `assets/piloto.json` (exportado del [creador web](https://algorithmicslab-stack.github.io/pixel-spacefighter/)). La música y los efectos se generan con numpy.

## Controles

| Acción | Teclas |
|---|---|
| Moverse | ← → / A D |
| Saltar | ↑ / W / Espacio / Z |
| Disparar | J / X / K |
| Empezar | Enter |
| Pausa | P / Esc |
| Silenciar | M |

## Compilar la versión web

```bash
pip install pygbag
python -m pygbag --build --title "Pixel Spacefighter" --ume_block 1 .
```

El resultado queda en `build/web` y se publica en la rama `gh-pages`.
