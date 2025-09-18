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
        cfg.setdefault(k, v)
    st.session_state.config = cfg
    return cfg

def save_button(config_dict):
    b = io.BytesIO(json.dumps(config_dict, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button("⬇️ Baixar configuração (JSON)", data=b, file_name="config_delivery.json", mime="application/json")

def clone_cfg(cfg: dict) -> dict:
    return {k: (v.copy() if isinstance(v, dict) else v) for k, v in cfg.items()}

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
    "perc_fds": 0.59,
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
    lo, hi = 0.0, max(cfg.get("faturamento", 1.0), 1.0)
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
# Sidebar (parâmetros)
# =============================
st.title("📊 Simulador Financeiro — Petisco da Serra")

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
    if "initialized" not in st.session_state:
        be_init, _ = calcular_break_even(cfg)
        st.session_state.fat = be_init
        st.session_state.fat_slider = be_init
        st.session_state.initialized = True

    def _sync_from_input():
        st.session_state.fat_slider = st.session_state.fat

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
# BE e slider no topo (destaque)
# =============================
cfg["faturamento"] = float(st.session_state.get("fat", cfg["faturamento"]))
be_fat, be_res = calcular_break_even(cfg)

# --- Slider grande no topo
st.markdown("### Simular faturamento (R$)")
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    st.caption("Arraste a barra para testar diferentes faturamentos.")
    st.slider(
        "",  # label vazio — título acima
        0.0, float(max(cfg["faturamento"] * 2, be_fat * 2, 10000.0)),
        float(st.session_state.get("fat_slider", cfg["faturamento"])),
        step=1000.0, format="%.0f", key="fat_slider",
        on_change=lambda: st.session_state.update({"fat": st.session_state["fat_slider"]})
    )
with top_col2:
    st.write("")
    st.write("")
    st.button(
        "🎯 Levar slider para o BE",
        use_container_width=True,
        help="Ajusta o faturamento simulado para o ponto de equilíbrio.",
        on_click=lambda: st.session_state.update({"fat": be_fat, "fat_slider": be_fat})
    )

# usa o valor atual do slider
cfg["faturamento"] = float(st.session_state["fat"])
res = calcular_metricas(cfg["faturamento"], cfg)

# --- Informativo do BE
delta = res["lucro"]
if abs(delta) <= 500:
    st.info(f"No **ponto de equilíbrio** (± {money(500)}). **BE ≈ {money(be_fat)}**.")
elif delta > 0:
    st.success(f"**Acima do BE** por {money(delta)} — **BE ≈ {money(be_fat)}**.")
else:
    st.warning(f"**Abaixo do BE** por {money(-delta)} — **BE ≈ {money(be_fat)}**.")

# =============================
# Abas
# =============================
tab_atual, tab_proj, tab_cenarios, tab_goal = st.tabs(
    ["📈 Atual", "📅 Projeção 24 meses", "🧪 Cenários", "🎯 Goal seek"]
)

# ======== ABA ATUAL ========
with tab_atual:
    st.subheader("Resultado da Simulação (com base no faturamento do slider)")
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

        st.caption("Lucro vs. Faturamento (linha do ponto de equilíbrio)")
        plot_max = max(cfg["faturamento"] * 2, be_fat * 1.6, 20000.0)
        steps = 100
        xs = [i * (plot_max / steps) for i in range(steps + 1)]
        lucros = [calcular_metricas(x, cfg)["lucro"] for x in xs]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(xs, lucros, linewidth=2)
        ax.axhline(0, linestyle="--", linewidth=1, color="#999")
        ax.axvline(be_fat, linestyle="--", linewidth=1, color="red")
        ax.scatter([res["fat"]], [res["lucro"]], s=40, zorder=3)
        ax.set_xlabel("Faturamento (R$)")
        ax.set_ylabel("Lucro (R$)")
        ax.set_title("Lucro vs. Faturamento")
        ax.grid(True, alpha=0.2)
        ax.text(be_fat, ax.get_ylim()[1]*0.95, f"BE ≈ {money(be_fat)}", rotation=90, va="top", ha="right", color="red")
        st.pyplot(fig, clear_figure=True)

# ---------------- PROJEÇÃO 24 MESES ----------------
def projetar_24m(cfg: dict, meses: int, cresc_mensal: float, inflacao_fixos: float, sazonalidade: list):
    rows = []
    fat0 = cfg["faturamento"]
    for m in range(meses):
        fator_cresc = (1.0 + cresc_mensal) ** m
        saz = sazonalidade[m % 12] if sazonalidade else 1.0
        fat_m = fat0 * fator_cresc * saz
        cfg_m = clone_cfg(cfg)
        infl = (1.0 + inflacao_fixos) ** m
        for k in ["fixo_aluguel","fixo_gerente","fixo_joao","fixo_util_basico","fixo_depreciacao","fixo_anotai","fixo_contador"]:
            cfg_m[k] = cfg[k] * infl
        res_m = calcular_metricas(fat_m, cfg_m)
        rows.append({
            "Mês": m+1, "Faturamento": res_m["fat"], "Pedidos": round(res_m["pedidos_mes"]),
            "Fixos": res_m["fixos_total"], "Variáveis": res_m["variaveis_total"],
            "Lucro": res_m["lucro"], "Margem (%)": res_m["margem"],
            "Cozinheiros": res_m["cozinheiros_qtd"], "MB semana": res_m["motoboys_sem"], "MB FDS": res_m["motoboys_fds"]
        })
    return pd.DataFrame(rows)

with tab_proj:
    st.subheader("Parâmetros da projeção")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        meses = st.number_input("Meses (horizonte)", 1, 60, value=24, step=1)
    with colp2:
        cresc_mensal = st.number_input("Crescimento mensal (%)", -100.0, 200.0, value=5.0, step=0.5) / 100.0
    with colp3:
        inflacao_fixos = st.number_input("Inflação dos fixos (% ao mês)", 0.0, 20.0, value=0.0, step=0.1) / 100.0

    # --- Sazonalidade em % (não mais fator)
    st.markdown("Sazonalidade (percentual por mês, 12 valores). Ex.: Dezembro **+10** = +10% sobre a média.")
    with st.expander("❓ O que é e como usar a sazonalidade?", expanded=False):
        st.write(
            """
            - Digite um **percentual** para cada mês, **em relação à média** (0 = mês neutro).
            - Exemplos:
                - Dezembro **+10** → aumenta o faturamento de dez em **10%**.
                - Janeiro **-5** → reduz o faturamento de jan em **5%**.
            - Esses percentuais são aplicados **só no faturamento** da projeção.
            - Se não quiser sazonalidade, **deixe tudo em 0**.
            """
        )
    meses_nomes = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    saz_df = pd.DataFrame({"Mês": meses_nomes, "% vs média": [0.0]*12})
    saz_edit = st.data_editor(
        saz_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "% vs média": st.column_config.NumberColumn(
                "% vs média",
                help="Digite +10 para +10% ou -5 para -5% versus a média do ano.",
                format="%.2f", step=1.0
            )
        }
    )
    # converte de % para fator interno
    sazonalidade = [1.0 + (pct/100.0) for pct in saz_edit["% vs média"].to_list()]

    df_proj = projetar_24m(cfg, meses, cresc_mensal, inflacao_fixos, sazonalidade)

    st.subheader("Resultados projetados")
    show = df_proj.copy()
    for col in ["Faturamento","Fixos","Variáveis","Lucro"]:
        show[col] = show[col].apply(money)
    show["Pedidos"] = show["Pedidos"].apply(lambda v: f"{v:,}".replace(",", "."))
    show["Margem (%)"] = show["Margem (%)"].map(lambda v: f"{v:.1f}%")
    st.dataframe(show, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Faturamento projetado")
        st.line_chart(df_proj.set_index("Mês")[["Faturamento"]])
    with c2:
        st.caption("Lucro projetado")
        st.line_chart(df_proj.set_index("Mês")[["Lucro"]])

    st.download_button("⬇️ Baixar projeção (CSV)",
                       data=df_proj.to_csv(index=False).encode("utf-8"),
                       file_name="projecao_24m.csv", mime="text/csv")

# ---------------- CENÁRIOS (sliders + reset) ----------------
def aplicar_cenario(cfg: dict, *, ticket_delta_pct=0.0, insumos_delta_pp=0.0,
                    ifood_delta_pp=0.0, mkt_delta_pp=0.0,
                    ent_h_delta_pct=0.0, diaria_mb_delta=0.0):
    c = clone_cfg(cfg)
    c["ticket_medio"] = max(0.01, c["ticket_medio"] * (1 + ticket_delta_pct))
    c["pct_insumos"] = max(0.0, c["pct_insumos"] + insumos_delta_pp)
    c["pct_ifood"] = max(0.0, c["pct_ifood"] + ifood_delta_pp)
    c["pct_marketing"] = max(0.0, c["pct_marketing"] + mkt_delta_pp)
    c["entregas_por_hora"] = max(0.1, c["entregas_por_hora"] * (1 + ent_h_delta_pct))
    c["mb_diaria"] = max(0.0, c["mb_diaria"] + diaria_mb_delta)
    return c

def ui_cenario(nome: str, key_prefix: str, defaults: dict):
    st.markdown(f"**{nome}**")
    c1, c2, c3 = st.columns(3)

    def _slider(label, minv, maxv, step, key, help_text, default):
        return st.slider(label, min_value=float(minv), max_value=float(maxv),
                         value=float(st.session_state.get(key, default)), step=float(step),
                         key=key, help=help_text)

    # Ticket Δ%  |  Insumos Δpp
    with c1:
        t_pct = _slider(
            f"Ticket Δ% ({nome})", -20.0, 20.0, 1.0,
            f"{key_prefix}_ticket", "Variação percentual do ticket médio. Ex.: +5 = +5%.",
            defaults.get("ticket", 0.0)
        )
        i_pp = _slider(
            f"Insumos Δpp ({nome})", -5.0, 5.0, 0.5,
            f"{key_prefix}_ins", "Mudança em pontos percentuais. Ex.: -1 pp → 33% vira 32%.",
            defaults.get("ins", 0.0)
        )

    # iFood Δpp  |  Marketing Δpp
    with c2:
        f_pp = _slider(
            f"iFood Δpp ({nome})", -5.0, 5.0, 0.5,
            f"{key_prefix}_ifood", "Mudança em pontos percentuais nas taxas de iFood/cartões.",
            defaults.get("ifood", 0.0)
        )
        m_pp = _slider(
            f"Marketing Δpp ({nome})", -5.0, 5.0, 0.5,
            f"{key_prefix}_mkt", "Mudança em pontos percentuais no marketing sobre o faturamento.",
            defaults.get("mkt", 0.0)
        )

    # Entregas/h Δ%  |  Diária MB ΔR$
    with c3:
        e_pct = _slider(
            f"Entregas/h Δ% ({nome})", -30.0, 30.0, 1.0,
            f"{key_prefix}_ent", "Produtividade do motoboy. Ex.: +10 = +10% de entregas por hora.",
            defaults.get("ent", 0.0)
        )
        d_rs = _slider(
            f"Diária MB ΔR$ ({nome})", -20.0, 20.0, 1.0,
            f"{key_prefix}_diaria", "Variação absoluta na diária por motoboy em reais.",
            defaults.get("diaria", 0.0)
        )

    # Botão de reset do cenário
    if st.button(f"↺ Resetar {nome}", key=f"reset_{key_prefix}"):
        st.session_state[f"{key_prefix}_ticket"] = defaults.get("ticket", 0.0)
        st.session_state[f"{key_prefix}_ins"] = defaults.get("ins", 0.0)
        st.session_state[f"{key_prefix}_ifood"] = defaults.get("ifood", 0.0)
        st.session_state[f"{key_prefix}_mkt"] = defaults.get("mkt", 0.0)
        st.session_state[f"{key_prefix}_ent"] = defaults.get("ent", 0.0)
        st.session_state[f"{key_prefix}_diaria"] = defaults.get("diaria", 0.0)

    # Converte para os formatos esperados (frações/pp/R$)
    return dict(
        ticket=t_pct/100.0,
        ins=i_pp/100.0,
        ifood=f_pp/100.0,
        mkt=m_pp/100.0,
        ent=e_pct/100.0,
        diaria=d_rs
    )

with tab_cenarios:
    st.subheader("Comparação de cenários no faturamento atual do slider")

    colA, colB, colC = st.columns(3)
    with colA:
        A = ui_cenario("A (Base)", "A", dict(ticket=0.0, ins=0.0, ifood=0.0, mkt=0.0, ent=0.0, diaria=0.0))
    with colB:
        B = ui_cenario("B (Otimista)", "B", dict(ticket=5.0, ins=-1.0, ifood=-1.0, mkt=0.0, ent=10.0, diaria=0.0))
    with colC:
        C = ui_cenario("C (Pessimista)", "C", dict(ticket=-5.0, ins=1.0, ifood=1.0, mkt=1.0, ent=-10.0, diaria=0.0))

    cen_cfgs = [
        ("A (Base)", aplicar_cenario(cfg, ticket_delta_pct=A["ticket"], insumos_delta_pp=A["ins"],
                                     ifood_delta_pp=A["ifood"], mkt_delta_pp=A["mkt"],
                                     ent_h_delta_pct=A["ent"], diaria_mb_delta=A["diaria"])),
        ("B (Otimista)", aplicar_cenario(cfg, ticket_delta_pct=B["ticket"], insumos_delta_pp=B["ins"],
                                         ifood_delta_pp=B["ifood"], mkt_delta_pp=B["mkt"],
                                         ent_h_delta_pct=B["ent"], diaria_mb_delta=B["diaria"])),
        ("C (Pessimista)", aplicar_cenario(cfg, ticket_delta_pct=C["ticket"], insumos_delta_pp=C["ins"],
                                           ifood_delta_pp=C["ifood"], mkt_delta_pp=C["mkt"],
                                           ent_h_delta_pct=C["ent"], diaria_mb_delta=C["diaria"])),
    ]

    cols = st.columns(3)
    for i, (nome, caux) in enumerate(cen_cfgs):
        with cols[i]:
            r = calcular_metricas(caux["faturamento"], caux)
            st.metric(f"{nome} — Lucro", money(r["lucro"]))
            st.metric("Margem", f"{r['margem']:.1f}%")
            st.metric("Motoboys (sem/FDS)", f"{r['motoboys_sem']}/{r['motoboys_fds']}")
            be_local, _ = calcular_break_even(caux)
            st.caption(f"BE: {money(be_local)}")

# ---------------- GOAL SEEK ----------------
def goal_seek(cfg: dict, alvo_tipo: str, alvo_valor: float, variavel: str,
              low: float, high: float, max_iter=40, expand_factor=1.5):
    def medir(v):
        c = clone_cfg(cfg)
        if variavel == "faturamento": c["faturamento"] = v
        elif variavel == "ticket": c["ticket_medio"] = v
        elif variavel == "pct_insumos": c["pct_insumos"] = v
        elif variavel == "pct_ifood": c["pct_ifood"] = v
        elif variavel == "pct_mkt": c["pct_marketing"] = v
        r = calcular_metricas(c["faturamento"], c)
        return r["margem"] if alvo_tipo == "margem" else r["lucro"]

    f_low, f_high = medir(low), medir(high)
    tries = 0
    while (f_low - alvo_valor) * (f_high - alvo_valor) > 0 and tries < 12:
        if abs(f_high - alvo_valor) <= abs(f_low - alvo_valor):
            high *= expand_factor
            f_high = medir(high)
        else:
            low = low / expand_factor if low > 0 else low - 1
            f_low = medir(low)
        tries += 1

    if (f_low - alvo_valor) * (f_high - alvo_valor) > 0:
        best_v, best_f = (low, f_low) if abs(f_low - alvo_valor) <= abs(f_high - alvo_valor) else (high, f_high)
        return best_v, best_f, False

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = medir(mid)
        if (f_low - alvo_valor) * (f_mid - alvo_valor) <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    sol = high
    return sol, medir(sol), True

with tab_goal:
    st.subheader("Resolver variável para atingir um alvo")
    col1, col2, col3 = st.columns(3)
    with col1:
        alvo_tipo = st.selectbox("Alvo", ["Margem (%)", "Lucro (R$)"])
        alvo_input = st.number_input("Valor do alvo", value=20.0 if alvo_tipo.startswith("Margem") else 10000.0, step=1.0)
    with col2:
        variavel = st.selectbox("Resolver para", ["Faturamento (R$)", "Ticket médio (R$)", "% Insumos", "% iFood", "% Marketing"])
    with col3:
        st.write("")
        if st.button("Calcular"):
            alvo_kind = "margem" if alvo_tipo.startswith("Margem") else "lucro"
            var_key = {"Faturamento (R$)":"faturamento","Ticket médio (R$)":"ticket",
                       "% Insumos":"pct_insumos","% iFood":"pct_ifood","% Marketing":"pct_mkt"}[variavel]
            bounds = {
                "faturamento": (0.0, max(2*cfg["faturamento"], 200000.0)),
                "ticket": (10.0, max(2*cfg["ticket_medio"], 300.0)),
                "pct_insumos": (0.05, 0.60),
                "pct_ifood": (0.00, 0.30),
                "pct_mkt": (0.00, 0.20),
            }[var_key]
            alvo_val = alvo_input if alvo_kind == "lucro" else float(alvo_input)
            sol, valor_final, ok = goal_seek(cfg, alvo_kind, alvo_val, var_key, bounds[0], bounds[1])

            def fmt_val(key, v):
                if key in ("pct_insumos","pct_ifood","pct_mkt"): return f"{v*100:.2f}%"
                if key == "faturamento": return money(v)
                if key == "ticket": return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                return str(v)

            if not ok:
                alvo_txt = f"{alvo_input:.1f}%" if alvo_kind == "margem" else money(alvo_input)
                ating_txt = f"{valor_final:.1f}%" if alvo_kind == "margem" else money(valor_final)
                st.warning(
                    f"Com **{variavel}** não dá para atingir **{alvo_txt}** nos limites testados.\n\n"
                    f"Melhor que conseguimos foi **{ating_txt}** com **{variavel} = {fmt_val(var_key, sol)}**.\n\n"
                    f"Tente resolver para **Faturamento (R$)** ou ajustar percentuais/fixos."
                )
            else:
                if alvo_kind == "margem":
                    st.success(f"Solução: **{variavel} = {fmt_val(var_key, sol)}** → Margem ≈ **{valor_final:.1f}%**")
                else:
                    st.success(f"Solução: **{variavel} = {fmt_val(var_key, sol)}** → Lucro ≈ **{money(valor_final)}**")
