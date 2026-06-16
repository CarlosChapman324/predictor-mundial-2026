"""Modulo de valor: valor esperado (EV), mejor apuesta y backtest de la estrategia.

El angulo financiero del proyecto. Compara la probabilidad del modelo con las
cuotas del mercado para detectar EV positivo, SOLO en los mercados con cuotas
(1X2, doble oportunidad, totales). Marco honesto: el mercado es muy eficiente, asi
que un EV positivo grande casi siempre delata un dato que le falta al modelo, no
una oportunidad real. Matematica pura, sin red.
"""
