"""
main_gui.py
Interfaz Gráfica Principal - Optimizador de Portafolio Cripto con Algoritmo Genético
Curso: Inteligencia Artificial (SI404) - TB1 2026

Ejecutar con: python main_gui.py
Requiere: pip install numpy matplotlib requests
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from genetic_algorithm import GeneticAlgorithm
from data_manager import DataManager, CRYPTO_IDS, generate_simulated_data


# ───────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ───────────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark":    "#0d1117",
    "bg_panel":   "#161b22",
    "bg_widget":  "#21262d",
    "accent":     "#58a6ff",
    "green":      "#3fb950",
    "red":        "#f85149",
    "yellow":     "#d29922",
    "text":       "#e6edf3",
    "text_dim":   "#8b949e",
    "border":     "#30363d",
}

CRYPTO_COLORS = [
    "#f7931a", "#627eea", "#9945ff", "#f3ba2f",
    "#00aae4", "#0033ad", "#e84142", "#e6007a",
    "#2a5ada", "#bfbbbb",
]


# ───────────────────────────────────────────────────────────────────────────────
# APLICACIÓN PRINCIPAL
# ───────────────────────────────────────────────────────────────────────────────
class CryptoPortfolioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimizador de Portafolio Cripto — Algoritmo Genético | IA SI404")
        self.geometry("1280x820")
        self.configure(bg=COLORS["bg_dark"])
        self.resizable(True, True)

        self.data_manager = DataManager()
        self._running = False
        self._result_weights = None
        self._result_names = None
        self._returns_matrix = None

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DE LA INTERFAZ
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS["bg_panel"], height=55)
        header.pack(fill="x", padx=0, pady=0)
        tk.Label(
            header,
            text="🧬  Optimizador de Portafolio Cripto  |  Algoritmo Genético",
            font=("Segoe UI", 15, "bold"),
            fg=COLORS["accent"], bg=COLORS["bg_panel"]
        ).pack(side="left", padx=20, pady=12)

        tk.Label(
            header,
            text="IA SI404 · TB1 2026",
            font=("Segoe UI", 10),
            fg=COLORS["text_dim"], bg=COLORS["bg_panel"]
        ).pack(side="right", padx=20, pady=12)

        # Contenedor principal (dos columnas)
        main = tk.Frame(self, bg=COLORS["bg_dark"])
        main.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(main, bg=COLORS["bg_dark"], width=310)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)

        right = tk.Frame(main, bg=COLORS["bg_dark"])
        right.pack(side="left", fill="both", expand=True)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        """Panel izquierdo: selección de criptos + parámetros del AG."""

        def section(label):
            f = tk.Frame(parent, bg=COLORS["bg_panel"], bd=0,
                         highlightbackground=COLORS["border"], highlightthickness=1)
            f.pack(fill="x", pady=(0, 8))
            tk.Label(f, text=label, font=("Segoe UI", 10, "bold"),
                     fg=COLORS["accent"], bg=COLORS["bg_panel"]).pack(anchor="w", padx=10, pady=(8, 4))
            return f

        # ── Selección de criptomonedas ──
        sec1 = section("📊  Seleccionar Criptomonedas (2-8)")
        self._crypto_vars = {}
        defaults = ["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)", "BNB (BNB)", "XRP (XRP)"]
        for name in CRYPTO_IDS:
            var = tk.BooleanVar(value=(name in defaults))
            self._crypto_vars[name] = var
            tk.Checkbutton(
                sec1, text=name, variable=var,
                font=("Segoe UI", 9), fg=COLORS["text"],
                bg=COLORS["bg_panel"], selectcolor=COLORS["bg_widget"],
                activebackground=COLORS["bg_panel"], activeforeground=COLORS["text"],
                cursor="hand2"
            ).pack(anchor="w", padx=14)
        tk.Frame(sec1, height=6, bg=COLORS["bg_panel"]).pack()

        # ── Parámetros del AG ──
        sec2 = section("⚙️  Parámetros del Algoritmo Genético")

        params = [
            ("Tamaño de Población:", "pop_size",    "80",   "Individuos por generación"),
            ("Generaciones:",         "n_gen",       "150",  "Iteraciones del AG"),
            ("Tasa de Cruce:",        "crossover",   "0.80", "Probabilidad de recombinación"),
            ("Tasa de Mutación:",     "mutation",    "0.12", "Probabilidad de mutación"),
            ("Élite (top N):",        "elite",       "8",    "Pasan directos a siguiente gen"),
            ("Días históricos:",      "days",        "180",  "Ventana de datos a usar"),
            ("Mín. por activo (%):",  "min_w",       "5",    "Mínimo % por criptomoneda"),
            ("Máx. por activo (%):",  "max_w",       "60",   "Máximo % por criptomoneda"),
            ("Máx. en Short (%):",    "max_short",   "40",   "Límite del portafolio en short"),
        ]

        self._param_vars = {}
        for label_text, key, default, tooltip in params:
            row = tk.Frame(sec2, bg=COLORS["bg_panel"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label_text, font=("Segoe UI", 8),
                     fg=COLORS["text_dim"], bg=COLORS["bg_panel"], width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self._param_vars[key] = var
            tk.Entry(row, textvariable=var, font=("Segoe UI", 9),
                     width=8, bg=COLORS["bg_widget"], fg=COLORS["text"],
                     insertbackground=COLORS["text"], relief="flat",
                     highlightbackground=COLORS["border"], highlightthickness=1
                     ).pack(side="left", padx=4)
        tk.Frame(sec2, height=6, bg=COLORS["bg_panel"]).pack()

        # ── Modo de Inversion ──
        sec_mode = section("📉  Modo de Inversión")
        self._allow_short = tk.BooleanVar(value=False)

        def _on_short_toggle():
            """Actualiza la descripción según el estado del checkbox."""
            if self._allow_short.get():
                lbl_mode_desc.config(
                    text="  LONG + SHORT activo: el AG puede comprar\n"
                         "  (apuesta a subida) o vender en corto\n"
                         "  (apuesta a bajada). Util en mercados bajistas.",
                    fg=COLORS["yellow"]
                )
            else:
                lbl_mode_desc.config(
                    text="  Solo LONG activo: el AG busca la mejor\n"
                         "  combinacion de compras. Mas conservador.",
                    fg=COLORS["text_dim"]
                )

        tk.Checkbutton(
            sec_mode,
            text="Permitir SHORT (venta en corto)",
            variable=self._allow_short,
            command=_on_short_toggle,
            font=("Segoe UI", 9, "bold"), fg=COLORS["yellow"],
            bg=COLORS["bg_panel"], selectcolor=COLORS["bg_widget"],
            activebackground=COLORS["bg_panel"], activeforeground=COLORS["yellow"],
            cursor="hand2"
        ).pack(anchor="w", padx=14, pady=(4, 0))

        lbl_mode_desc = tk.Label(
            sec_mode,
            text="  Solo LONG activo: el AG busca la mejor\n"
                 "  combinacion de compras. Mas conservador.",
            font=("Segoe UI", 8), fg=COLORS["text_dim"],
            bg=COLORS["bg_panel"], justify="left"
        )
        lbl_mode_desc.pack(anchor="w", padx=14, pady=(2, 6))

        # ── Modo datos ──
        sec3 = section("🌐  Fuente de Datos")
        self._data_mode = tk.StringVar(value="api")
        for text, val in [("API CoinGecko (tiempo real)", "api"),
                          ("Datos simulados (offline)", "sim")]:
            tk.Radiobutton(
                sec3, text=text, variable=self._data_mode, value=val,
                font=("Segoe UI", 9), fg=COLORS["text"],
                bg=COLORS["bg_panel"], selectcolor=COLORS["bg_widget"],
                activebackground=COLORS["bg_panel"], cursor="hand2"
            ).pack(anchor="w", padx=14)
        tk.Frame(sec3, height=6, bg=COLORS["bg_panel"]).pack()

        # ── Botón de ejecución ──
        self._btn_run = tk.Button(
            parent, text="🚀  OPTIMIZAR PORTAFOLIO",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["accent"], fg="#0d1117",
            activebackground="#79c0ff", activeforeground="#0d1117",
            relief="flat", padx=10, pady=10,
            cursor="hand2", command=self._start_optimization
        )
        self._btn_run.pack(fill="x", pady=4)

        self._status_label = tk.Label(
            parent, text="Listo. Configure y presione Optimizar.",
            font=("Segoe UI", 8), fg=COLORS["text_dim"],
            bg=COLORS["bg_dark"], wraplength=290, justify="center"
        )
        self._status_label.pack(pady=4)

        self._progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self._progress.pack(fill="x", pady=2)

    def _build_right_panel(self, parent):
        """Panel derecho: gráficos de evolución y resultados."""

        # ── Notebook con pestañas ──
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=COLORS["bg_dark"], borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=COLORS["bg_widget"], foreground=COLORS["text_dim"],
                        padding=[12, 6], font=("Segoe UI", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["bg_panel"])],
                  foreground=[("selected", COLORS["accent"])])

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # Pestaña 1: Evolución del AG
        tab_evo = tk.Frame(nb, bg=COLORS["bg_dark"])
        nb.add(tab_evo, text="  📈 Convergencia del AG  ")

        self._fig_evo = Figure(figsize=(7, 4.5), facecolor=COLORS["bg_dark"])
        self._ax_evo = self._fig_evo.add_subplot(111)
        self._canvas_evo = FigureCanvasTkAgg(self._fig_evo, master=tab_evo)
        self._canvas_evo.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self._init_evo_plot()

        # Pestaña 2: Distribución del portafolio
        tab_port = tk.Frame(nb, bg=COLORS["bg_dark"])
        nb.add(tab_port, text="  🥧 Portafolio Óptimo  ")

        self._fig_port = Figure(figsize=(7, 4.5), facecolor=COLORS["bg_dark"])
        self._canvas_port = FigureCanvasTkAgg(self._fig_port, master=tab_port)
        self._canvas_port.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Pestaña 3: Métricas
        tab_metrics = tk.Frame(nb, bg=COLORS["bg_panel"])
        nb.add(tab_metrics, text="  📋 Métricas  ")
        self._build_metrics_tab(tab_metrics)

    def _build_metrics_tab(self, parent):
        """Panel de métricas del portafolio óptimo."""
        tk.Label(parent, text="Métricas del Portafolio Óptimo",
                 font=("Segoe UI", 12, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg_panel"]).pack(pady=(20, 10))

        self._metrics_frame = tk.Frame(parent, bg=COLORS["bg_panel"])
        self._metrics_frame.pack(fill="both", expand=True, padx=20)

        self._metrics_labels = {}
        metrics_config = [
            ("retorno_anual",          "📈 Retorno Anual Esperado",  "%"),
            ("volatilidad_anual",      "📉 Volatilidad Anual",       "%"),
            ("sharpe_ratio",           "⚖️  Sharpe Ratio",            ""),
            ("max_drawdown",           "🔻 Máximo Drawdown",         "%"),
            ("retorno_diario_promedio","📅 Retorno Diario Promedio", "%"),
        ]

        for key, label, unit in metrics_config:
            row = tk.Frame(self._metrics_frame, bg=COLORS["bg_widget"],
                           highlightbackground=COLORS["border"], highlightthickness=1)
            row.pack(fill="x", pady=4, ipady=10, ipadx=10)

            tk.Label(row, text=label, font=("Segoe UI", 10),
                     fg=COLORS["text_dim"], bg=COLORS["bg_widget"]).pack(side="left", padx=16)

            val_label = tk.Label(row, text="—", font=("Segoe UI", 12, "bold"),
                                  fg=COLORS["text"], bg=COLORS["bg_widget"])
            val_label.pack(side="right", padx=16)
            self._metrics_labels[key] = (val_label, unit)

        # Tabla de pesos
        tk.Label(parent, text="Distribución del Portafolio",
                 font=("Segoe UI", 11, "bold"), fg=COLORS["accent"],
                 bg=COLORS["bg_panel"]).pack(pady=(20, 6))

        cols = ("Criptomoneda", "Peso (%)", "Asignación USD (1000$)")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        style = ttk.Style()
        style.configure("Treeview",
                        background=COLORS["bg_widget"], foreground=COLORS["text"],
                        fieldbackground=COLORS["bg_widget"], rowheight=28)
        style.configure("Treeview.Heading",
                        background=COLORS["bg_panel"], foreground=COLORS["accent"],
                        font=("Segoe UI", 9, "bold"))

        for col in cols:
            self._tree.heading(col, text=col)
            self._tree.column(col, anchor="center", width=180)
        self._tree.pack(fill="x", padx=20, pady=(0, 20))

    def _init_evo_plot(self):
        ax = self._ax_evo
        ax.set_facecolor(COLORS["bg_panel"])
        ax.spines["bottom"].set_color(COLORS["border"])
        ax.spines["left"].set_color(COLORS["border"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=COLORS["text_dim"])
        ax.set_xlabel("Generación", color=COLORS["text_dim"], fontsize=9)
        ax.set_ylabel("Sharpe Ratio", color=COLORS["text_dim"], fontsize=9)
        ax.set_title("Convergencia del Algoritmo Genético", color=COLORS["text"], fontsize=11)
        ax.text(0.5, 0.5, "Ejecute la optimización para ver la evolución",
                ha="center", va="center", transform=ax.transAxes,
                color=COLORS["text_dim"], fontsize=10)
        self._fig_evo.tight_layout()
        self._canvas_evo.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # LÓGICA DE OPTIMIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    def _start_optimization(self):
        if self._running:
            return

        selected = [name for name, var in self._crypto_vars.items() if var.get()]
        if len(selected) < 2:
            messagebox.showwarning("Selección insuficiente",
                                   "Seleccione al menos 2 criptomonedas.")
            return
        if len(selected) > 8:
            messagebox.showwarning("Demasiadas criptomonedas",
                                   "Seleccione máximo 8 criptomonedas para mayor claridad.")
            return

        try:
            min_w = float(self._param_vars["min_w"].get()) / 100
            max_w = float(self._param_vars["max_w"].get()) / 100
            if min_w >= max_w:
                messagebox.showerror("Parámetros inválidos",
                                     "El mínimo por activo debe ser menor que el máximo.")
                return
            params = {
                "pop_size":  int(self._param_vars["pop_size"].get()),
                "n_gen":     int(self._param_vars["n_gen"].get()),
                "crossover": float(self._param_vars["crossover"].get()),
                "mutation":  float(self._param_vars["mutation"].get()),
                "elite":     int(self._param_vars["elite"].get()),
                "days":      int(self._param_vars["days"].get()),
                "min_w":     min_w,
                "max_w":     max_w,
                "max_short": float(self._param_vars["max_short"].get()) / 100,
            }
        except ValueError:
            messagebox.showerror("Parámetros inválidos",
                                 "Verifique que todos los parámetros sean números válidos.")
            return

        self._running = True
        self._btn_run.config(state="disabled", text="⏳  Optimizando...")
        self._progress["value"] = 0

        thread = threading.Thread(
            target=self._run_optimization,
            args=(selected, params),
            daemon=True
        )
        thread.start()

    def _run_optimization(self, selected: list, params: dict):
        try:
            # ── Obtener datos ──
            self._update_status("Obteniendo datos históricos...")

            if self._data_mode.get() == "api":
                def progress_cb(i, total, name):
                    pct = int((i / total) * 30)
                    self._progress["value"] = pct
                    self._update_status(f"Descargando {name}...")

                returns_matrix, valid_names = self.data_manager.get_returns_matrix(
                    selected, days=params["days"], progress_callback=progress_cb
                )
            else:
                returns_matrix, valid_names = generate_simulated_data(
                    n_assets=len(selected), n_days=params["days"]
                )
                valid_names = selected[:len(valid_names)]

            self._returns_matrix = returns_matrix
            self._update_status(f"Datos listos para {len(valid_names)} criptos. Ejecutando AG...")
            self._progress["value"] = 30

            # ── Ejecutar AG ──
            fitness_history = []
            avg_history = []
            n_gen = params["n_gen"]

            def gen_callback(gen, best_fit, avg_fit):
                fitness_history.append(best_fit)
                avg_history.append(avg_fit)
                pct = 30 + int((gen / n_gen) * 65)
                self._progress["value"] = pct
                if gen % 10 == 0:
                    self._update_status(
                        f"Generación {gen}/{n_gen} | Mejor Sharpe: {best_fit:.4f}"
                    )
                # Actualizar gráfico cada 20 generaciones
                if gen % 20 == 0 and gen > 0:
                    self.after(0, lambda fh=fitness_history[:], ah=avg_history[:]:
                               self._update_evo_plot(fh, ah))

            allow_short = self._allow_short.get()
            short_cap   = params["max_short"]  # viene de Mín./Max. short en params

            ag = GeneticAlgorithm(
                returns_matrix=returns_matrix,
                population_size=params["pop_size"],
                n_generations=n_gen,
                crossover_rate=params["crossover"],
                mutation_rate=params["mutation"],
                elite_size=params["elite"],
                min_weight=params["min_w"],
                max_weight=params["max_w"],
                allow_short=allow_short,
                short_cap=short_cap,
            )

            best = ag.evolve(callback=gen_callback)

            # ── Mostrar resultados ──
            metrics = ag.get_portfolio_metrics(best.weights)
            self._result_weights = best.weights
            self._result_names = valid_names

            self.after(0, lambda: self._show_results(
                best.weights, valid_names, metrics,
                ag.fitness_history, ag.avg_fitness_history
            ))
            self._progress["value"] = 100
            self._update_status("✅ Optimización completada.")

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self._update_status(f"❌ Error: {e}")
        finally:
            self._running = False
            self.after(0, lambda: self._btn_run.config(
                state="normal", text="🚀  OPTIMIZAR PORTAFOLIO"
            ))

    # ─────────────────────────────────────────────────────────────────────────
    # ACTUALIZACIÓN DE UI
    # ─────────────────────────────────────────────────────────────────────────
    def _update_status(self, msg: str):
        self.after(0, lambda: self._status_label.config(text=msg))

    def _update_evo_plot(self, fitness_history, avg_history):
        ax = self._ax_evo
        ax.clear()
        ax.set_facecolor(COLORS["bg_panel"])
        gens = list(range(len(fitness_history)))
        ax.plot(gens, fitness_history, color=COLORS["accent"],
                linewidth=2, label="Mejor Sharpe")
        ax.plot(gens, avg_history, color=COLORS["text_dim"],
                linewidth=1, linestyle="--", label="Promedio Sharpe", alpha=0.7)
        ax.set_xlabel("Generación", color=COLORS["text_dim"], fontsize=9)
        ax.set_ylabel("Sharpe Ratio", color=COLORS["text_dim"], fontsize=9)
        ax.set_title("Convergencia del Algoritmo Genético", color=COLORS["text"], fontsize=11)
        ax.tick_params(colors=COLORS["text_dim"])
        ax.legend(facecolor=COLORS["bg_widget"], edgecolor=COLORS["border"],
                  labelcolor=COLORS["text"], fontsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._fig_evo.tight_layout()
        self._canvas_evo.draw()

    def _show_results(self, weights, names, metrics, fitness_hist, avg_hist):
        """Actualiza todos los paneles con los resultados del AG."""
        # Gráfico de convergencia final
        self._update_evo_plot(fitness_hist, avg_hist)

        # Gráfico de torta del portafolio
        self._fig_port.clear()
        ax1 = self._fig_port.add_subplot(121)
        ax2 = self._fig_port.add_subplot(122)
        self._fig_port.patch.set_facecolor(COLORS["bg_dark"])

        # Separar longs y shorts para colores distintos
        bar_colors   = []
        pie_labels   = []
        pie_values   = []
        pie_colors   = []
        color_idx    = 0
        short_names  = [n.split("(")[1][:-1] if "(" in n else n for n in names]
        for i, (name, w) in enumerate(zip(names, weights)):
            tick  = name.split("(")[1][:-1] if "(" in name else name
            if w < -0.001:  # short
                bar_colors.append(COLORS["red"])
                pie_values.append(abs(w))
                pie_colors.append(COLORS["red"])
                pie_labels.append(f"{tick} SHORT: {abs(w)*100:.1f}%")
            else:            # long
                c = CRYPTO_COLORS[color_idx % len(CRYPTO_COLORS)]
                color_idx += 1
                bar_colors.append(c)
                pie_values.append(w)
                pie_colors.append(c)
                pie_labels.append(f"{tick}: {w*100:.1f}%")

        wedge_props = {"linewidth": 2, "edgecolor": COLORS["bg_dark"]}
        wedges, texts, autotexts = ax1.pie(
            pie_values, labels=None, autopct="%1.1f%%",
            colors=pie_colors, wedgeprops=wedge_props,
            startangle=140, pctdistance=0.75
        )
        for at in autotexts:
            at.set_color(COLORS["bg_dark"])
            at.set_fontsize(8)
            at.set_fontweight("bold")

        ax1.set_title("Distribución Óptima", color=COLORS["text"], fontsize=11, pad=12)
        ax1.legend(wedges, pie_labels,
                   loc="center left", bbox_to_anchor=(-0.4, 0.5),
                   facecolor=COLORS["bg_panel"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=8)

        # Barras de pesos (negativo = short, positivo = long)
        ax2.set_facecolor(COLORS["bg_panel"])
        bar_vals = weights * 100
        bars = ax2.barh(short_names, bar_vals, color=bar_colors, edgecolor=COLORS["bg_dark"])
        for bar, w in zip(bars, weights):
            xpos = bar.get_width()
            label = f"SHORT {abs(w)*100:.1f}%" if w < -0.001 else f"{w*100:.1f}%"
            ax2.text(xpos + (0.5 if xpos >= 0 else -0.5),
                     bar.get_y() + bar.get_height() / 2,
                     label, va="center",
                     ha="left" if xpos >= 0 else "right",
                     color=COLORS["text"], fontsize=8)
        ax2.axvline(0, color=COLORS["border"], linewidth=1)
        ax2.set_xlabel("Porcentaje (%) — Negativo = SHORT", color=COLORS["text_dim"], fontsize=9)
        ax2.set_title("Pesos por Activo", color=COLORS["text"], fontsize=11)
        ax2.tick_params(colors=COLORS["text_dim"])
        for spine in ax2.spines.values():
            spine.set_color(COLORS["border"])
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        max_abs = max(abs(bar_vals).max() * 1.3, 5)
        ax2.set_xlim(-max_abs, max_abs)

        self._fig_port.tight_layout(pad=2)
        self._canvas_port.draw()

        # Métricas
        color_map = {
            "retorno_anual":          (COLORS["green"] if metrics["retorno_anual"] > 0 else COLORS["red"]),
            "volatilidad_anual":      COLORS["yellow"],
            "sharpe_ratio":           (COLORS["green"] if metrics["sharpe_ratio"] > 1 else COLORS["yellow"]),
            "max_drawdown":           COLORS["red"],
            "retorno_diario_promedio":(COLORS["green"] if metrics["retorno_diario_promedio"] > 0 else COLORS["red"]),
        }
        for key, (label_widget, unit) in self._metrics_labels.items():
            val = metrics[key]
            label_widget.config(text=f"{val}{unit}", fg=color_map.get(key, COLORS["text"]))

        # Tabla de distribución con indicación de posición
        for row in self._tree.get_children():
            self._tree.delete(row)
        for name, w in sorted(zip(names, weights), key=lambda x: -abs(x[1])):
            if w < -0.001:
                pos_label = f"SHORT  {abs(w)*100:.2f}%"
                usd_label = f"-${abs(w)*1000:.2f} (ganancia si baja)"
            else:
                pos_label = f"LONG   {w*100:.2f}%"
                usd_label = f"${w*1000:.2f}"
            self._tree.insert("", "end", values=(name, pos_label, usd_label))


# ───────────────────────────────────────────────────────────────────────────────
# ENTRADA PRINCIPAL
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = CryptoPortfolioApp()
    app.mainloop()