"""
data_manager.py
Módulo de obtención y caché de datos históricos de criptomonedas.
Fuente: CoinGecko API (gratuita, sin clave API requerida)
Almacenamiento local: SQLite
"""

import sqlite3
import requests
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# Mapeo nombre visible -> ID de CoinGecko
CRYPTO_IDS = {
    "Bitcoin (BTC)":   "bitcoin",
    "Ethereum (ETH)":  "ethereum",
    "Solana (SOL)":    "solana",
    "BNB (BNB)":       "binancecoin",
    "XRP (XRP)":       "ripple",
    "Cardano (ADA)":   "cardano",
    "Avalanche (AVAX)":"avalanche-2",
    "Polkadot (DOT)":  "polkadot",
    "Chainlink (LINK)":"chainlink",
    "Litecoin (LTC)":  "litecoin",
}

DB_PATH = "crypto_cache.db"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


class DataManager:
    """
    Maneja la obtención y caché de precios históricos de criptomonedas.

    Flujo:
    1. Intenta cargar datos desde SQLite (caché local)
    2. Si no hay datos o están desactualizados, consulta CoinGecko
    3. Guarda en SQLite para evitar llamadas repetidas
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Inicializa la base de datos SQLite con el esquema necesario."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                crypto_id   TEXT NOT NULL,
                date        TEXT NOT NULL,
                price_usd   REAL NOT NULL,
                fetched_at  TEXT NOT NULL,
                PRIMARY KEY (crypto_id, date)
            )
        """)
        conn.commit()
        conn.close()

    def _fetch_from_api(self, crypto_id: str, days: int = 180) -> Optional[List[Tuple]]:
        """
        Obtiene precios históricos de CoinGecko.
        Retorna lista de (fecha_str, precio) o None si falla.
        """
        url = f"{COINGECKO_BASE}/coins/{crypto_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 429:
                print(f"  Rate limit alcanzado, esperando 60s...")
                time.sleep(60)
                response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                print(f"  Error API para {crypto_id}: HTTP {response.status_code}")
                return None

            data = response.json()
            prices = data.get("prices", [])

            result = []
            for timestamp_ms, price in prices:
                date_str = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")
                result.append((date_str, price))

            return result

        except Exception as e:
            print(f"  Error de conexión para {crypto_id}: {e}")
            return None

    def _is_cache_valid(self, crypto_id: str, min_rows: int = 30) -> bool:
        """Verifica si los datos en caché son suficientes y recientes (< 24h)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*), MAX(fetched_at) FROM price_data
            WHERE crypto_id = ?
        """, (crypto_id,))
        count, last_fetch = cursor.fetchone()
        conn.close()

        if count < min_rows or last_fetch is None:
            return False

        last_dt = datetime.fromisoformat(last_fetch)
        return (datetime.utcnow() - last_dt).total_seconds() < 86400  # 24 horas

    def _save_to_db(self, crypto_id: str, price_data: List[Tuple]):
        """Guarda los datos en SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.executemany("""
            INSERT OR REPLACE INTO price_data (crypto_id, date, price_usd, fetched_at)
            VALUES (?, ?, ?, ?)
        """, [(crypto_id, date, price, now) for date, price in price_data])

        conn.commit()
        conn.close()

    def _load_from_db(self, crypto_id: str, days: int = 180) -> List[Tuple]:
        """Carga datos desde SQLite, limitado a los últimos N días."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT date, price_usd FROM price_data
            WHERE crypto_id = ? AND date >= ?
            ORDER BY date ASC
        """, (crypto_id, since))

        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_prices(self, crypto_id: str, days: int = 180, force_refresh: bool = False) -> Optional[np.ndarray]:
        """
        Obtiene array de precios para una cripto. Usa caché si está disponible.

        Retorna np.ndarray con precios diarios ordenados cronológicamente.
        """
        if force_refresh or not self._is_cache_valid(crypto_id, min_rows=30):
            print(f"  Descargando datos de {crypto_id}...")
            raw = self._fetch_from_api(crypto_id, days=days)
            if raw:
                self._save_to_db(crypto_id, raw)
                time.sleep(1.2)  # Respetar rate limit de CoinGecko (50 req/min)
            else:
                print(f"  Usando caché existente para {crypto_id}")
        else:
            print(f"  Cargando {crypto_id} desde caché local")

        rows = self._load_from_db(crypto_id, days=days)
        if not rows:
            return None

        return np.array([price for _, price in rows])

    def get_returns_matrix(
        self,
        selected_cryptos: List[str],
        days: int = 180,
        progress_callback=None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Construye la matriz de retornos logarítmicos diarios.

        Parámetros:
        -----------
        selected_cryptos : List[str]
            Lista de nombres visibles (claves de CRYPTO_IDS)
        days : int
            Número de días históricos a usar
        progress_callback : callable
            Función llamada con (i, total, nombre) para actualizar progreso

        Retorna:
        --------
        (returns_matrix, valid_names)
            - returns_matrix: shape (n_dias, n_activos)
            - valid_names: criptos para las que se obtuvieron datos
        """
        prices_dict = {}
        valid_names = []
        total = len(selected_cryptos)

        for i, name in enumerate(selected_cryptos):
            if progress_callback:
                progress_callback(i, total, name)

            crypto_id = CRYPTO_IDS.get(name)
            if not crypto_id:
                print(f"  AVISO: {name} no encontrado en CRYPTO_IDS")
                continue

            prices = self.get_prices(crypto_id, days=days)
            if prices is not None and len(prices) > 10:
                prices_dict[name] = prices
                valid_names.append(name)

        if not prices_dict:
            raise ValueError("No se pudieron obtener datos de ninguna criptomoneda.")

        # Alinear longitudes (tomar el mínimo común)
        min_len = min(len(p) for p in prices_dict.values())
        aligned = {name: prices[-min_len:] for name, prices in prices_dict.items()}

        # Calcular retornos logarítmicos diarios: ln(P_t / P_{t-1})
        returns_list = []
        for name in valid_names:
            prices = aligned[name]
            log_returns = np.diff(np.log(prices))
            returns_list.append(log_returns)

        # Transponer para obtener shape (n_dias, n_activos)
        returns_matrix = np.column_stack(returns_list)

        return returns_matrix, valid_names


def generate_simulated_data(n_assets: int = 5, n_days: int = 180) -> Tuple[np.ndarray, List[str]]:
    """
    Genera datos simulados para pruebas offline.
    Útil cuando no hay conexión a internet.
    """
    np.random.seed(42)
    names = list(CRYPTO_IDS.keys())[:n_assets]

    returns_list = []
    for i in range(n_assets):
        # Parámetros realistas para criptomonedas
        mu = np.random.uniform(-0.001, 0.003)
        sigma = np.random.uniform(0.02, 0.06)
        returns = np.random.normal(mu, sigma, n_days - 1)
        returns_list.append(returns)

    returns_matrix = np.column_stack(returns_list)
    return returns_matrix, names
