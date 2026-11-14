# Punto de entrada para la versión web (pygbag). En escritorio usa: python juego.py
import asyncio
import numpy  # pygbag necesita ver este import para incluir numpy
import pygame
import juego

asyncio.run(juego.main())
