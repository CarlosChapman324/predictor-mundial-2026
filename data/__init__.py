"""Ingesta y normalizacion de datos: historico de partidos, fixture 2026 y Elo.

Es la unica capa que toca la red (descargas). El modelo nunca importa de aqui
en tiempo de ejecucion: consume los Parquet ya guardados en disco.
"""
