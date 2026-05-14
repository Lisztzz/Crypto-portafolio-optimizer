"""
genetic_algorithm.py
Algoritmo Genético para Optimización de Portafolio de Criptomonedas
Soporta modos: Solo LONG | LONG + SHORT (venta en corto)
Curso: Inteligencia Artificial (SI404) - TB1 2026
"""

import numpy as np
import random
from typing import List, Tuple


class Individual:
    def __init__(self, n_assets: int, weights: np.ndarray = None):
        self.n_assets = n_assets
        self.weights  = weights if weights is not None else self._random_weights()
        self.fitness  = 0.0

    def _random_weights(self) -> np.ndarray:
        w = np.random.dirichlet(np.ones(self.n_assets))
        return w / w.sum()

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f})"


class GeneticAlgorithm:
    """
    Algoritmo Genético para optimización de portafolio.

    allow_short=False  →  Solo posiciones LONG (compra).
    allow_short=True   →  Permite LONG y SHORT (venta en corto).
                          Un peso negativo = apostamos a que ESA cripto BAJE.
                          Útil en mercados bajistas.

    min_weight : mínimo % de capital por activo activo (ej. 0.05 = 5%)
    max_weight : máximo % de capital por activo (ej. 0.60 = 60%)
    short_cap  : máximo % total del portafolio en posiciones cortas (ej. 0.40 = 40%)
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        population_size: int = 100,
        n_generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.12,
        elite_size: int = 10,
        risk_free_rate: float = 0.0001,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
        allow_short: bool = False,
        short_cap: float = 0.40,
    ):
        self.returns_matrix = returns_matrix
        self.n_assets       = returns_matrix.shape[1]
        self.population_size = population_size
        self.n_generations  = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.elite_size     = elite_size
        self.risk_free_rate = risk_free_rate
        self.allow_short    = allow_short
        self.short_cap      = short_cap
        self.min_weight     = min(min_weight, 1.0 / self.n_assets)
        self.max_weight     = min(max_weight, 1.0 - (self.n_assets - 1) * self.min_weight)

        self.population:          List[Individual] = []
        self.best_individual:     Individual       = None
        self.fitness_history:     List[float]      = []
        self.avg_fitness_history: List[float]      = []

    # ── RESTRICCIONES ─────────────────────────────────────────────────────────
    def _enforce_constraints(self, w: np.ndarray) -> np.ndarray:
        """
        Normaliza el vector de pesos garantizando:

          1. sum(abs(w)) == 1  exactamente (exposición bruta = 100% del capital).
             Los shorts consumen capital igual que los longs; no hay apalancamiento.

          2. abs(w[i]) >= mn  para todo activo activo (mínimo con abs()).

          3. abs(w[i]) <= mx  para todo activo (máximo con abs()), nunca ignorado.
             El exceso se redistribuye proporcionalmente entre los demás activos
             respetando también su límite máximo.

          4. Los signos se preservan intactos durante toda la operación.
             Un short que entra negativo sale negativo.

          5. En modo long+short: sum(abs(w[shorts])) <= short_cap.

        Algoritmo (único para ambos modos):
          a) Trabajar siempre sobre magnitudes = abs(w), guardar signos aparte.
          b) Aplicar límites [mn, mx] sobre magnitudes con redistribución iterativa.
          c) En long+short: si la exposición short supera short_cap, escalar shorts
             proporcionalmente y redistribuir el capital liberado a longs.
          d) Normalizar magnitudes a suma = 1 (exposición bruta).
          e) Restaurar signos: result = magnitudes * signs.
        """
        w   = w.copy()
        mn  = self.min_weight
        mx  = self.max_weight

        # ── a) Separar magnitudes y signos ────────────────────────────────────
        signs = np.sign(w)
        signs[signs == 0] = 1          # activos neutros → long por defecto
        if not self.allow_short:
            signs = np.ones(self.n_assets)   # long-only: todos positivos
        mags = np.abs(w)

        long_mask  = signs > 0
        short_mask = signs < 0

        # ── b) Aplicar [mn, mx] sobre magnitudes con redistribución ───────────
        # Iterar hasta que todas las magnitudes activas estén en [mn, mx].
        # Cada iteración:
        #   - Recortar los que superan mx y repartir el exceso entre los que
        #     tienen margen disponible (proporcional a su magnitud actual).
        #   - Elevar los que están bajo mn y descontar el déficit de los que
        #     están sobre mn (proporcional).
        # Se aplica sobre TODAS las magnitudes sin distinción long/short todavía.
        mags = np.clip(mags, 0.0, mx)
        for _ in range(300):
            over  = mags > mx                    # ← usa abs implícito (mags >= 0)
            under = mags < mn                    # ← usa abs implícito (mags >= 0)
            if not over.any() and not under.any():
                break
            if over.any():
                excess = (mags[over] - mx).sum()
                mags[over] = mx
                free = (~over) & (mags < mx)
                if free.any():
                    mags[free] += excess * (mags[free] / mags[free].sum())
            if under.any():
                deficit = (mn - mags[under]).sum()
                mags[under] = mn
                above = mags > mn
                if above.any():
                    mags[above] -= deficit * (mags[above] / mags[above].sum())

        mags = np.clip(mags, 0.0, mx)   # clip final por seguridad numérica

        # ── c) Respetar short_cap (solo en modo long+short) ───────────────────
        if self.allow_short and short_mask.any():
            short_exp = mags[short_mask].sum()
            if short_exp > self.short_cap:
                # Escalar shorts proporcionalmente al cap
                scale = self.short_cap / short_exp
                freed = mags[short_mask].sum() * (1 - scale)   # capital liberado
                mags[short_mask] *= scale
                # Redistribuir capital liberado a longs (proporcional, respetando mx)
                if long_mask.any() and freed > 1e-12:
                    room  = mx - mags[long_mask]                # margen disponible por long
                    room  = np.clip(room, 0.0, None)
                    total_room = room.sum()
                    if total_room > 1e-12:
                        mags[long_mask] += freed * (room / total_room)
                    mags[long_mask] = np.clip(mags[long_mask], 0.0, mx)

        # ── d) Normalizar a exposición bruta = 1  (sum(abs) == 1) ─────────────
        # y luego volver a aplicar [mn, mx] porque la división escala las
        # magnitudes y puede violar los límites. Iterar hasta convergencia.
        for _pass in range(10):
            gross = mags.sum()
            if gross < 1e-9:
                # Fallback: distribuir uniformemente preservando los signos actuales
                mags = np.full(self.n_assets, 1.0 / self.n_assets)
                break
            mags = mags / gross          # sum(mags) == 1 aquí

            # Re-aplicar [mn, mx] tras la normalización
            converged = True
            for _ in range(300):
                over  = mags > mx
                under = mags < mn
                if not over.any() and not under.any():
                    break
                converged = False
                if over.any():
                    excess = (mags[over] - mx).sum()
                    mags[over] = mx
                    free = (~over) & (mags < mx)
                    if free.any():
                        mags[free] += excess * (mags[free] / mags[free].sum())
                if under.any():
                    deficit = (mn - mags[under]).sum()
                    mags[under] = mn
                    above = mags > mn
                    if above.any():
                        mags[above] -= deficit * (mags[above] / mags[above].sum())
            mags = np.clip(mags, 0.0, mx)

            # Si ya convergió (ningún límite violado), salir
            if converged:
                break
            # Si no convergió, volver a normalizar y repetir

        # Garantía final: suma == 1 y todos en [mn, mx]
        s = mags.sum()
        mags = mags / s if s > 1e-9 else np.full(self.n_assets, 1.0 / self.n_assets)
        mags = np.clip(mags, 0.0, mx)

        # ── e) Restaurar signos: result[i] = mags[i] * signs[i] ──────────────
        return mags * signs

    # ── FITNESS: Sharpe Ratio ─────────────────────────────────────────────────
    def calculate_fitness(self, individual: Individual) -> float:
        """
        Sharpe Ratio anualizado.

        Como _enforce_constraints garantiza sum(abs(w)) == 1, el retorno
        del portafolio ya está expresado por unidad de capital real: no se
        necesita dividir por exposición bruta dentro del fitness.

        r_port[t] = sum_i( w[i] * r_i[t] )
          - w[i] > 0 → contribución positiva cuando r_i sube
          - w[i] < 0 → contribución positiva cuando r_i baja (short)

        Sharpe anualizado = (mean(r_port) - rf) / std(r_port) * sqrt(252)
        """
        w                 = individual.weights
        portfolio_returns = self.returns_matrix @ w
        mean_r            = np.mean(portfolio_returns)
        std_r             = np.std(portfolio_returns)
        if std_r < 1e-10:
            return 0.0
        return float((mean_r - self.risk_free_rate) / std_r * np.sqrt(252))

    # ── INICIALIZACIÓN ────────────────────────────────────────────────────────
    def _random_individual(self) -> Individual:
        if not self.allow_short:
            w = np.random.dirichlet(np.ones(self.n_assets))
        else:
            w = np.random.dirichlet(np.ones(self.n_assets))
            for i in range(self.n_assets):
                if random.random() < 0.3:
                    w[i] *= -1
        return Individual(self.n_assets, self._enforce_constraints(w))

    def initialize_population(self):
        self.population = [self._random_individual() for _ in range(self.population_size)]
        self._evaluate_population()

    # ── SELECCIÓN ────────────────────────────────────────────────────────────
    def selection_tournament(self, k: int = 3) -> Individual:
        return max(random.sample(self.population, k), key=lambda x: x.fitness)

    # ── CRUCE BLX-alpha ───────────────────────────────────────────────────────
    def crossover_blend(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.crossover_rate:
            return (Individual(self.n_assets, p1.weights.copy()),
                    Individual(self.n_assets, p2.weights.copy()))
        alpha = 0.5
        c1_w  = np.zeros(self.n_assets)
        c2_w  = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            lo, hi = min(p1.weights[i], p2.weights[i]), max(p1.weights[i], p2.weights[i])
            rng    = hi - lo
            c1_w[i] = lo - alpha * rng + random.random() * rng * (1 + 2 * alpha)
            c2_w[i] = lo - alpha * rng + random.random() * rng * (1 + 2 * alpha)
        return (Individual(self.n_assets, self._enforce_constraints(c1_w)),
                Individual(self.n_assets, self._enforce_constraints(c2_w)))

    # ── MUTACIÓN ──────────────────────────────────────────────────────────────
    def mutate(self, individual: Individual) -> Individual:
        """
        Mutacion suave tipo creep (gaussiano gen a gen).

        Diferencia clave frente al reemplazo aleatorio:
          El reemplazo asigna un valor completamente nuevo al gen, ignorando
          el valor actual y pudiendo invertir el signo de activos con pesos
          altos (salto destructivo: un long de 0.55 pasa a -0.55 de golpe).

        Creep mutation:
          - Cada gen se perturba de forma independiente (prob gene_mut_prob).
          - El ruido gaussiano es pequeno (sigma ~3%%), preservando la
            magnitud y el signo del gen original.
          - El flip de signo (long->short o viceversa) es un operador
            separado, de baja probabilidad y solo para activos con magnitud
            pequena (flip_max_mag), donde el salto es menos destructivo.
        """
        if random.random() > self.mutation_rate:
            return individual

        w             = individual.weights.copy()
        gene_mut_prob = 0.30          # prob de mutar cada gen individualmente
        sigma         = 0.03          # desv. estandar del creep (~3%% del capital)
        flip_prob     = 0.10          # prob de flip de signo por activo
        flip_max_mag  = self.min_weight * 3   # solo flippear activos pequenos

        # ── Creep: sumar ruido gaussiano gen a gen ──────────────────────
        for i in range(self.n_assets):
            if random.random() < gene_mut_prob:
                w[i] += random.gauss(0, sigma)   # perturbacion suave

        # ── Flip de signo controlado (solo en modo long+short) ──────────
        # Solo para activos de magnitud pequena para evitar saltos grandes.
        if self.allow_short:
            for i in range(self.n_assets):
                if abs(w[i]) <= flip_max_mag and random.random() < flip_prob:
                    w[i] *= -1

        return Individual(self.n_assets, self._enforce_constraints(w))

    # ── EVALUACIÓN ────────────────────────────────────────────────────────────
    def _evaluate_population(self):
        for ind in self.population:
            ind.fitness = self.calculate_fitness(ind)

    # ── EVOLUCIÓN PRINCIPAL ───────────────────────────────────────────────────
    def evolve(self, callback=None) -> Individual:
        self.initialize_population()
        self.fitness_history, self.avg_fitness_history = [], []

        for gen in range(self.n_generations):
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best_f = self.population[0].fitness
            avg_f  = float(np.mean([x.fitness for x in self.population]))
            self.fitness_history.append(best_f)
            self.avg_fitness_history.append(avg_f)
            if callback:
                callback(gen, best_f, avg_f)

            new_pop = [Individual(self.n_assets, self.population[i].weights.copy())
                       for i in range(self.elite_size)]
            while len(new_pop) < self.population_size:
                c1, c2 = self.crossover_blend(self.selection_tournament(),
                                               self.selection_tournament())
                new_pop.extend([self.mutate(c1), self.mutate(c2)])

            self.population = new_pop[:self.population_size]
            self._evaluate_population()

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0]
        return self.best_individual

    # ── MÉTRICAS ──────────────────────────────────────────────────────────────
    def get_portfolio_metrics(self, weights: np.ndarray) -> dict:
        pr         = self.returns_matrix @ weights
        mean_daily = float(np.mean(pr))
        std_daily  = float(np.std(pr))
        ann_ret    = mean_daily * 252
        ann_vol    = std_daily  * np.sqrt(252)
        sharpe     = (ann_ret - self.risk_free_rate * 252) / ann_vol if ann_vol > 0 else 0
        cum        = np.cumprod(1 + pr)
        max_dd     = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))
        return {
            "retorno_anual":           round(ann_ret    * 100, 2),
            "volatilidad_anual":       round(ann_vol    * 100, 2),
            "sharpe_ratio":            round(sharpe,          4),
            "max_drawdown":            round(max_dd     * 100, 2),
            "retorno_diario_promedio": round(mean_daily * 100, 4),
            "n_long":  int(np.sum(weights > 0.001)),
            "n_short": int(np.sum(weights < -0.001)),
        }


def generate_simulated_data(n_assets: int = 5, n_days: int = 180):
    from data_manager import CRYPTO_IDS
    np.random.seed(42)
    names = list(CRYPTO_IDS.keys())[:n_assets]
    rets  = [np.random.normal(np.random.uniform(-0.001, 0.003),
                               np.random.uniform(0.02, 0.06), n_days - 1)
             for _ in range(n_assets)]
    return np.column_stack(rets), names