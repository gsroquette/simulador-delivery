# simulador_delivery.py
import json, io, math
import streamlit as st
import pandas as pd
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador Financeiro — Delivery", page_icon="📊", layout="wide")

# =============================
# Utilidades
# =============================
def money(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def load_config(defaults: dict):
    if "config" not in st.session_state:
        st.session_state.config = defaults.copy()
    cfg = st.session_state.config
    for k, v in defaults.items():
        cfg.setdefault(k, v)  # garante chaves novas
    st.session_state.config = cfg
    return cfg

def save_button(config_dict):
    b = io.BytesIO(json.dumps(config_dict, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button("⬇️ Baixar configuração (JSON)", data=b, file_name="config_delivery.json", mime="application/json")

# =============================
# Valores padrão
# =============================
DEFAULTS = {
    # Entradas principais
    "ticket_medio": 70.0,
    "faturamento": 50000.0,

    # FIXOS base (sem motoboys)
    "fixo_aluguel": 2000.0,
    "fixo_gerente": 3000.0,
    "fixo_joao": 1000.0,
    "fixo_util_basico": 1000.0,
    "fixo_depreciacao": 200.0,
    "fixo_anotai": 279.79,
    "fixo_contador": 0.0,

    # COZINHEIROS (por faixa de faturamento)
    "cook_t1_limite": 50000.0,   "cook_t1_sal": 5000.0,   # 2 cozinheiros
    "cook_t2_limite": 150000.0,  "cook_t2_sal": 7500.0,   # 3 cozinheiros
    "cook_t3_sal": 10000.0,                               # 4 cozinheiros

    # VARIÁVEIS (% sobre faturamento)
    "pct_insumos": 0.333,
    "pct_embalagens": 0.02,
    "pct_energia_extra": 0.015,
    "pct_ifood": 0.11,
    "pct_marketing": 0.02,

    # MOTOBOYS — 100% variável por demanda
    "perc_fds": 0.59,           # ≈ 59%
    "dias_uteis": 18,
    "dias_fds": 12,
    "entregas_por_hora": 2.5,
    "horas_semana": 6.0,
    "horas_fds": 6.0,
    "mb_diaria": 60.0,
    "mb_custo_por_entrega": 5.0,
    "motoboys_minimos": 1,

    "mostrar_graficos": True
}

# =============================
# Núcleo de cálculo
# =============================
def calcular_metricas(fat: float, cfg: dict):
    ticket = cfg["ticket_medio"]
    pedidos_mes = fat / ticket if ticket else 0.0

    fixo_base = (
        cfg["fixo_aluguel"] + cfg["fixo_gerente"] + cfg["fixo_joao"] +
        cfg["fixo_util_basico"] + cfg["fixo_depreciacao"] + cfg["fixo_anotai"] +
        cfg["fixo_contador"]
    )

    # tiers de cozinheiros
    if fat <= cfg["cook_t1_limite"]:
        cozinheiros_custo, cozinheiros_qtd = cfg["cook_t1_sal"], 2
    elif fat <= cfg["cook_t2_limite"]:
        cozinheiros_custo, cozinheiros_qtd = cfg["cook_t2_sal"], 3
    else:
        cozinheiros_custo, cozinheiros_qtd = cfg["cook_t3_sal"], 4

    fixos_total = fixo_base + cozinheiros_custo

    # variáveis %
    insumos = fat * cfg["pct_insumos"]
    embalagens = fat * cfg["pct_embalagens"]
    energia_extra = fat * cfg["pct_energia_extra"]
    ifood = fat * cfg["pct_ifood"]
    marketing = fat * cfg["pct_marketing"]

    # motoboys por demanda
    perc_fds = cfg["perc_fds"]
    dias_uteis, dias_fds = cfg["dias_uteis"], cfg["dias_fds"]
    ent_h, h_sem, h_fds = cfg["entregas_por_hora"], cfg["horas_semana"], cfg["horas_fds"]
    mb_min = cfg["motoboys_minimos"]

    pedidos_fds, pedidos_sem = pedidos_mes * perc_fds, pedidos_mes * (1 - perc_fds)
    pedidos_dia_sem = (pedidos_sem / dias_uteis) if dias_uteis else 0.0
    pedidos_dia_fds = (pedidos_fds / dias_fds) if dias_fds else 0.0

    cap_sem, cap_fds = ent_h * h_sem, ent_h * h_fds
    motoboys_sem = max(mb_min, math.ceil(pedidos_dia_sem / cap_sem)) if cap_sem > 0 else mb_min
    motoboys_fds = max(mb_min, math.ceil(pedidos_dia_fds / cap_fds)) if cap_fds > 0 else mb_min

    diarias_sem = motoboys_sem * cfg["mb_diaria"] * dias_uteis
    diarias_fds = motoboys_fds * cfg["mb_diaria"] * dias_fds
    custo_por_entregas = pedidos_mes * cfg["mb_custo_por_entrega"]
    custo_motoboys_total = diarias_sem + diarias_fds + custo_por_entregas

    variaveis_total = insumos + embalagens + energia_extra + ifood + marketing + custo_motoboys_total
    total_custos = fixos_total + variaveis_total
    lucro = fat - total_custos
    margem = (lucro / fat * 100) if fat > 0 else 0.0

    return {
        "fat": fat, "ticket": ticket, "pedidos_mes": pedidos_mes,
        "fixos_total": fixos_total, "insumos": insumos, "embalagens": embalagens,
        "energia_extra": energia_extra, "ifood": ifood, "marketing": marketing,
        "custo_motoboys_total": custo_motoboys_total, "variaveis_total": variaveis_total,
        "total_custos": total_custos, "lucro": lucro, "margem": margem,
        "pedidos_dia_sem": pedidos_dia_sem, "pedidos_dia_fds": pedidos_dia_fds,
        "motoboys_sem": motoboys_sem, "motoboys_fds": motoboys_fds,
        "cozinheiros_qtd": cozinheiros_qtd,
    }

def calcular_break_even(cfg: dict, max_iter=40):
    """Bisseção até lucro≈0 (considera degraus de cozinheiros/motoboys)."""
    lo, hi = 0.0, max(cfg.get("faturamento", 1.0), 1.0)
    # garante hi com lucro positivo
    while calcular_metricas(hi, cfg)["lucro"] <= 0 and hi < 5_000_000:
        hi *= 1.5
    if calcular_metricas(hi, cfg)["lucro"] <= 0:
        return hi, calcular_metricas(hi, cfg)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if calcular_metricas(mid, cfg)["lucro"] > 0:
            hi = mid
        else:
            lo = mid
    be_fat = hi
    return be_fat, calcular_metricas(be_fat, cfg)

# =============================
# Sidebar (com sync do faturamento)
# =============================
st.title("📊 Simulador Financeiro — Delivery de Petiscos")

with st.sidebar:
    st.header("⚙️ Configuração")
    cfg = load_config(DEFAULTS)

    up = st.file_uploader("Carregar configuração (JSON)", type=["json"],
                          help="Suba um arquivo salvo pelo botão de download para restaurar todos os parâmetros.")
    if up:
        cfg = json.load(up)
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        st.session_state.config = cfg

    st.divider()
    st.caption("Entradas principais")

    # inicializa no BE na 1ª carga
    if "initialized" not in st.session_state:
        be_init, _ = calcular_break_even(cfg)
        st.session_state.fat = be_init
        st.session_state.fat_slider = be_init
        st.session_state.initialized = True

    # callbacks p/ sincronizar sidebar ↔ slider
    def _sync_from_input():
        st.session_state.fat_slider = st.session_state.fat

    def _sync_from_slider():
        st.session_state.fat = st.session_state.fat_slider

    cfg["faturamento"] = st.number_input(
        "Faturamento (R$)", min_value=0.0,
        value=float(st.session_state.get("fat", cfg["faturamento"])),
        step=1000.0, format="%.0f",
        help="Receita bruta estimada do mês.",
        key="fat", on_change=_sync_from_input
    )

    cfg["ticket_medio"] = st.number_input(
        "Ticket médio (R$)", min_value=0.01,
        value=float(cfg["ticket_medio"]),
        step=1.0, format="%.2f",
        help="Valor médio por pedido; usado para estimar o número de pedidos."
    )

    st.divider()
    st.caption("Custos fixos (mensais)")
    colf1, colf2 = st.columns(2)
    with colf1:
        cfg["fixo_aluguel"] = st.number_input("Aluguel", 0.0, value=float(cfg["fixo_aluguel"]), step=100.0)
        cfg["fixo_gerente"] = st.number_input("Gerente (c/ comissão)", 0.0, value=float(cfg["fixo_gerente"]), step=100.0)
        cfg["fixo_joao"] = st.number_input("João", 0.0, value=float(cfg["fixo_joao"]), step=100.0)
        cfg["fixo_depreciacao"] = st.number_input("Depreciação equipamento", 0.0, value=float(cfg["fixo_depreciacao"]), step=50.0)
    with colf2:
        cfg["fixo_util_basico"] = st.number_input("Luz/Água/IPTU/Tel/Internet (base)", 0.0, value=float(cfg["fixo_util_basico"]), step=100.0)
        cfg["fixo_anotai"] = st.number_input("Anota.i (fixo)", 0.0, value=float(cfg["fixo_anotai"]), step=10.0)
        cfg["fixo_contador"] = st.number_input("Contador (fixo)", 0.0, value=float(cfg["fixo_contador"]), step=50.0)

    st.divider()
    st.caption("Cozinheiros (escala por faturamento)")
    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        cfg["cook_t1_limite"] = st.number_input("Limite T1 (2 cozinheiros)", 0.0, value=float(cfg["cook_t1_limite"]), step=1000.0)
        cfg["cook_t1_sal"] = st.number_input("Custo T1 (2 cozinheiros)", 0.0, value=float(cfg["cook_t1_sal"]), step=100.0)
    with colc2:
        cfg["cook_t2_limite"] = st.number_input("Limite T2 (3 cozinheiros)", 0.0, value=float(cfg["cook_t2_limite"]), step=1000.0)
        cfg["cook_t2_sal"] = st.number_input("Custo T2 (3 cozinheiros)", 0.0, value=float(cfg["cook_t2_sal"]), step=100.0)
    with colc3:
        cfg["cook_t3_sal"] = st.number_input("Custo T3 (4 cozinheiros)", 0.0, value=float(cfg["cook_t3_sal"]), step=100.0)

    st.divider()
    st.caption("Percentuais variáveis (%) — aplicados sobre o faturamento")
    colp1, colp2, colp3, colp4, colp5 = st.columns(5)
    cfg["pct_insumos"] = st.number_input("Insumos (%)", 0.0, 100.0, value=float(cfg["pct_insumos"] * 100), step=0.5) / 100
    cfg["pct_embalagens"] = st.number_input("Embalagens (%)", 0.0, 100.0, value=float(cfg["pct_embalagens"] * 100), step=0.5) / 100
    cfg["pct_energia_extra"] = st.number_input("Energia extra (%)", 0.0, 100.0, value=float(cfg["pct_energia_extra"] * 100), step=0.1) / 100
    cfg["pct_ifood"] = st.number_input("iFood/cartão (%)", 0.0, 100.0, value=float(cfg["pct_ifood"] * 100), step=0.5) / 100
    cfg["pct_marketing"] = st.number_input("Marketing (%)", 0.0, 100.0, value=float(cfg["pct_marketing"] * 100), step=0.5) / 100

    st.divider()
    st.caption("Motoboys — modelo por demanda (100% variável)")
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        cfg["perc_fds"] = st.number_input("% dos pedidos no FDS", 0.0, 100.0, value=float(cfg["perc_fds"] * 100), step=1.0) / 100
        cfg["dias_uteis"] = st.number_input("Dias úteis no mês", 0, value=int(cfg["dias_uteis"]), step=1)
    with colD2:
        cfg["dias_fds"] = st.number_input("Dias de FDS no mês", 0, value=int(cfg["dias_fds"]), step=1)
        cfg["entregas_por_hora"] = st.number_input("Entregas por hora / motoboy", 0.0, value=float(cfg["entregas_por_hora"]), step=0.1)
    with colD3:
        cfg["horas_semana"] = st.number_input("Horas por dia (semana)", 0.0, value=float(cfg["horas_semana"]), step=0.5)
        cfg["horas_fds"] = st.number_input("Horas por dia (FDS)", 0.0, value=float(cfg["horas_fds"]), step=0.5)

    colD4, colD5, colD6 = st.columns(3)
    with colD4:
        cfg["mb_diaria"] = st.number_input("Diária por motoboy (R$)", 0.0, value=float(cfg["mb_diaria"]), step=5.0)
    with colD5:
        cfg["mb_custo_por_entrega"] = st.number_input("Custo por entrega (R$)", 0.0, value=float(cfg["mb_custo_por_entrega"]), step=0.5)
    with colD6:
        cfg["motoboys_minimos"] = st.number_input("Motoboys mínimos", 0, value=int(cfg["motoboys_minimos"]), step=1)

    st.divider()
    cfg["mostrar_graficos"] = st.checkbox("Mostrar gráficos", value=bool(cfg["mostrar_graficos"]))
    st.button("🔄 Resetar para padrão", on_click=lambda: st.session_state.update({"config": DEFAULTS.copy()}))
    save_button(cfg)

# =============================
# BE de referência (sempre atual)
# =============================
cfg["faturamento"] = float(st.session_state.get("fat", cfg["faturamento"]))
be_fat, be_res = calcular_break_even(cfg)

# =============================
# E se…?  (slider controla o mesmo faturamento)
# =============================
st.subheader("E se…? (simule diferentes faturamentos)")
row = st.columns([3, 1])
with row[0]:
    st.slider(
        "Simular faturamento (R$)",
        0.0, float(max(cfg["faturamento"] * 2, be_fat * 2, 10000.0)),
        float(st.session_state.get("fat_slider", cfg["faturamento"])),
        step=1000.0, format="%.0f",
        key="fat_slider",
        on_change=lambda: st.session_state.update({"fat": st.session_state["fat_slider"]})
    )
with row[1]:
    st.button(
        "🎯 Levar slider para o BE",
        use_container_width=True,
        help="Ajusta o faturamento simulado para o ponto de equilíbrio.",
        on_click=lambda: st.session_state.update({"fat": be_fat, "fat_slider": be_fat})
    )

cfg["faturamento"] = float(st.session_state["fat"])  # mantém em sync
# Cálculo com o valor atual (após slider/botão)
res = calcular_metricas(cfg["faturamento"], cfg)

# =============================
# Referência do BE (em expander para melhor leitura)
# =============================
with st.expander("📌 Referência: Ponto de Equilíbrio (não muda com o slider)", expanded=True):
    delta = res["lucro"]
    if abs(delta) <= 500:
        st.info(f"Você está **no ponto de equilíbrio** (± {money(500)}) — BE ≈ {money(be_fat)}.")
    elif delta > 0:
        st.success(f"**Acima do BE** por {money(delta)}. BE ≈ {money(be_fat)}.")
    else:
        st.warning(f"**Abaixo do BE** por {money(-delta)}. BE ≈ {money(be_fat)} — faltam **{money(be_fat - res['fat'])}**.")

    be_cols = st.columns(4)
    be_cols[0].metric("BE (faturamento)", money(be_fat))
    be_cols[1].metric("Pedidos no BE (estim.)", f"{int(round(be_fat / res['ticket'])):,}".replace(",", "."))
    be_cols[2].metric("Motoboys semana (BE)", f"{be_res['motoboys_sem']}")
    be_cols[3].metric("Motoboys FDS (BE)", f"{be_res['motoboys_fds']}")

# =============================
# Resultado atual (controlado pelo slider)
# =============================
st.subheader("Resultado da Simulação")

tabela = pd.DataFrame(
    {"Valores": [
        res["fat"], round(res["pedidos_mes"]),
        res["fixos_total"], res["insumos"], res["embalagens"], res["energia_extra"], res["ifood"], res["marketing"],
        res["custo_motoboys_total"], res["total_custos"], res["lucro"], res["margem"]
    ]},
    index=[
        "Faturamento", "Pedidos (estim.)",
        "Fixos (total)", "Insumos", "Embalagens", "Energia extra", "iFood/cartão", "Marketing",
        "Motoboys (total variável)", "Custos totais", "Lucro líquido", "Margem líquida (%)"
    ]
)
linhas_moeda = [
    "Faturamento","Fixos (total)","Insumos","Embalagens","Energia extra",
    "iFood/cartão","Marketing","Motoboys (total variável)","Custos totais","Lucro líquido"
]
styler = tabela.style \
    .format(money, subset=idx[linhas_moeda, "Valores"]) \
    .format(lambda v: f"{int(round(v)):,}".replace(",", "."), subset=idx[["Pedidos (estim.)"], "Valores"]) \
    .format(lambda v: f"{v:.1f}%", subset=idx[["Margem líquida (%)"], "Valores"])
st.dataframe(styler, use_container_width=True)

st.caption("Dimensionamento (estimado)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Pedidos/dia útil", f"{res['pedidos_dia_sem']:.1f}")
m2.metric("Pedidos/dia FDS", f"{res['pedidos_dia_fds']:.1f}")
m3.metric("Motoboys na semana", f"{res['motoboys_sem']}")
m4.metric("Motoboys no FDS", f"{res['motoboys_fds']}")
m5.metric("Cozinheiros", f"{res['cozinheiros_qtd']}")

# =============================
# Gráficos
# =============================
if cfg["mostrar_graficos"]:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Custos (quebra)")
        chart_df = pd.DataFrame({
            "Categoria": ["Fixos", "Insumos", "Embalagens", "Energia extra", "iFood/cartão", "Marketing", "Motoboys"],
            "Valor": [res["fixos_total"], res["insumos"], res["embalagens"], res["energia_extra"], res["ifood"], res["marketing"], res["custo_motoboys_total"]]
        }).set_index("Categoria")
        st.bar_chart(chart_df)

    with c2:
        st.caption("Lucro líquido e Margem")
        st.metric("Lucro líquido", money(res["lucro"]))
        st.metric("Margem líquida", f"{res['margem']:.1f}%")

    # Gráfico: Lucro vs. Faturamento com BE
    st.caption("Lucro vs. Faturamento (linha do ponto de equilíbrio)")
    plot_max = max(cfg["faturamento"] * 2, be_fat * 1.6, 20000.0)
    steps = 100
    xs = [i * (plot_max / steps) for i in range(steps + 1)]
    lucros = [calcular_metricas(x, cfg)["lucro"] for x in xs]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs, lucros, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1, color="#999")
    ax.axvline(be_fat, linestyle="--", linewidth=1, color="red")
    # marca o ponto atual
    ax.scatter([res["fat"]], [res["lucro"]], s=40, zorder=3)
    ax.set_xlabel("Faturamento (R$)")
    ax.set_ylabel("Lucro (R$)")
    ax.set_title("Lucro vs. Faturamento")
    ax.grid(True, alpha=0.2)
    ax.text(be_fat, ax.get_ylim()[1]*0.95, f"BE ≈ {money(be_fat)}", rotation=90, va="top", ha="right", color="red")
    st.pyplot(fig, clear_figure=True)
