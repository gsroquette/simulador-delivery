# simulador_delivery.py
import json, io, math
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulador Financeiro — Delivery", page_icon="📊", layout="wide")

# -----------------------------
# Utilidades
# -----------------------------
def money(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def load_config(defaults: dict):
    if "config" not in st.session_state:
        st.session_state.config = defaults.copy()
    return st.session_state.config

def save_button(config_dict):
    b = io.BytesIO(json.dumps(config_dict, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button("⬇️ Baixar configuração (JSON)", data=b, file_name="config_delivery.json", mime="application/json")

# -----------------------------
# Valores padrão
# -----------------------------
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

    # COZINHEIROS (por faixa de faturamento)
    "cook_t1_limite": 50000.0,   "cook_t1_sal": 5000.0,   # 2 cozinheiros
    "cook_t2_limite": 150000.0,  "cook_t2_sal": 7500.0,   # 3 cozinheiros
    "cook_t3_sal": 10000.0,                                 # 4 cozinheiros

    # VARIÁVEIS (em fração; exibimos como % na UI)
    "pct_insumos": 0.333,
    "pct_embalagens": 0.02,
    "pct_energia_extra": 0.015,
    "pct_ifood": 0.11,
    "pct_marketing": 0.02,

    # MOTOBOYS — novo modelo 100% variável por demanda
    "perc_fds": 0.45,                 # % dos pedidos que caem em sext/sáb/dom
    "dias_uteis": 18,
    "dias_fds": 12,
    "entregas_por_hora": 2.5,
    "horas_semana": 6.0,
    "horas_fds": 10.0,
    "mb_diaria": 60.0,                # diária paga a cada motoboy
    "mb_custo_por_entrega": 5.0,      # adicional por entrega (todos os motoboys)
    "motoboys_minimos": 2,            # mínimo operacional sempre 2

    "mostrar_graficos": True
}

# -----------------------------
# Interface
# -----------------------------
st.title("📊 Simulador Financeiro — Delivery de Petiscos")

with st.sidebar:
    st.header("⚙️ Configuração")

    cfg = load_config(DEFAULTS)

    up = st.file_uploader("Carregar configuração (JSON)", type=["json"],
                          help="Suba um arquivo salvo pelo botão de download para restaurar todos os parâmetros.")
    if up:
        cfg = json.load(up)
        st.session_state.config = cfg

    st.divider()
    st.caption("Entradas principais")
    cfg["faturamento"] = st.number_input(
        "Faturamento (R$)", min_value=0.0, step=1000.0, value=float(cfg["faturamento"]),
        help="Receita bruta estimada do mês."
    )
    cfg["ticket_medio"] = st.number_input(
        "Ticket médio (R$)", min_value=1.0, step=1.0, value=float(cfg["ticket_medio"]),
        help="Valor médio por pedido; usado para estimar o número de pedidos."
    )

    st.divider()
    st.caption("Custos fixos (mensais)")
    colf1, colf2 = st.columns(2)
    with colf1:
        cfg["fixo_aluguel"] = st.number_input("Aluguel", 0.0, value=float(cfg["fixo_aluguel"]), step=100.0,
                                              help="Aluguel mensal do ponto.")
        cfg["fixo_gerente"] = st.number_input("Gerente (c/ comissão)", 0.0, value=float(cfg["fixo_gerente"]), step=100.0,
                                              help="Salário do gerente com comissão média embutida.")
        cfg["fixo_joao"] = st.number_input("João", 0.0, value=float(cfg["fixo_joao"]), step=100.0,
                                           help="Salário/ajuda de custo do João.")
        cfg["fixo_depreciacao"] = st.number_input("Depreciação equipamento", 0.0, value=float(cfg["fixo_depreciacao"]), step=50.0,
                                                  help="Quota mensal de depreciação dos equipamentos.")
    with colf2:
        cfg["fixo_util_basico"] = st.number_input("Luz/Água/IPTU/Tel/Internet (base)", 0.0, value=float(cfg["fixo_util_basico"]), step=100.0,
                                                  help="Conta base, independente do volume de produção. O excedente vai em 'Energia extra (%)'.")
        cfg["fixo_anotai"] = st.number_input("Anota.i (fixo)", 0.0, value=float(cfg["fixo_anotai"]), step=10.0,
                                             help="Assinatura mensal do sistema Anota.i.")

    st.divider()
    st.caption("Cozinheiros (escala por faturamento)")
    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        cfg["cook_t1_limite"] = st.number_input("Limite T1 (2 cozinheiros)", 0.0, value=float(cfg["cook_t1_limite"]), step=1000.0,
                                                help="Até este faturamento utiliza 2 cozinheiros.")
        cfg["cook_t1_sal"] = st.number_input("Custo T1 (2 cozinheiros)", 0.0, value=float(cfg["cook_t1_sal"]), step=100.0,
                                             help="Custo total mensal de 2 cozinheiros.")
    with colc2:
        cfg["cook_t2_limite"] = st.number_input("Limite T2 (3 cozinheiros)", 0.0, value=float(cfg["cook_t2_limite"]), step=1000.0,
                                                help="Até este faturamento utiliza 3 cozinheiros.")
        cfg["cook_t2_sal"] = st.number_input("Custo T2 (3 cozinheiros)", 0.0, value=float(cfg["cook_t2_sal"]), step=100.0,
                                             help="Custo total mensal de 3 cozinheiros.")
    with colc3:
        cfg["cook_t3_sal"] = st.number_input("Custo T3 (4 cozinheiros)", 0.0, value=float(cfg["cook_t3_sal"]), step=100.0,
                                             help="Custo total mensal de 4 cozinheiros (acima do Limite T2).")

    # -------- Percentuais em %
    st.divider()
    st.caption("Percentuais variáveis (%) — aplicados sobre o faturamento")
    colp1, colp2, colp3, colp4, colp5 = st.columns(5)

    cfg["pct_insumos"] = st.number_input(
        "Insumos (%)", 0.0, 100.0, value=float(cfg["pct_insumos"] * 100), step=0.5,
        help="Ingredientes/alimentos. Relação 1:3 ≈ 33,3%."
    ) / 100

    cfg["pct_embalagens"] = st.number_input(
        "Embalagens (%)", 0.0, 100.0, value=float(cfg["pct_embalagens"] * 100), step=0.5,
        help="Gasto com embalagens descartáveis."
    ) / 100

    cfg["pct_energia_extra"] = st.number_input(
        "Energia extra (%)", 0.0, 100.0, value=float(cfg["pct_energia_extra"] * 100), step=0.1,
        help="Parcela variável da energia (fritadeiras/freezers conforme produção)."
    ) / 100

    cfg["pct_ifood"] = st.number_input(
        "iFood/cartão (%)", 0.0, 100.0, value=float(cfg["pct_ifood"] * 100), step=0.5,
        help="Taxas de plataforma e cartões (média)."
    ) / 100

    cfg["pct_marketing"] = st.number_input(
        "Marketing (%)", 0.0, 100.0, value=float(cfg["pct_marketing"] * 100), step=0.5,
        help="Investimento em marketing (Google Ads, redes sociais, cupons)."
    ) / 100

    # -------- Motoboys — 100% variáveis por demanda
    st.divider()
    st.caption("Motoboys — modelo por demanda (100% variável)")
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        cfg["perc_fds"] = st.number_input("% dos pedidos no FDS", 0.0, 100.0, value=float(cfg["perc_fds"] * 100), step=1.0,
                                          help="Percentual de pedidos que cai em sext/sáb/dom.") / 100
        cfg["dias_uteis"] = st.number_input("Dias úteis no mês", 0, value=int(cfg["dias_uteis"]), step=1,
                                            help="Quantidade de dias úteis de operação no mês.")
    with colD2:
        cfg["dias_fds"] = st.number_input("Dias de FDS no mês", 0, value=int(cfg["dias_fds"]), step=1,
                                          help="Quantidade de sextas/sábados/domingos no mês.")
        cfg["entregas_por_hora"] = st.number_input("Entregas por hora / motoboy", 0.0, value=float(cfg["entregas_por_hora"]), step=0.1,
                                                   help="Produtividade média por hora de um motoboy.")
    with colD3:
        cfg["horas_semana"] = st.number_input("Horas por dia (semana)", 0.0, value=float(cfg["horas_semana"]), step=0.5,
                                              help="Jornada média por dia útil.")
        cfg["horas_fds"] = st.number_input("Horas por dia (FDS)", 0.0, value=float(cfg["horas_fds"]), step=0.5,
                                           help="Jornada média por dia de sext/sáb/dom.")

    colD4, colD5, colD6 = st.columns(3)
    with colD4:
        cfg["mb_diaria"] = st.number_input("Diária por motoboy (R$)", 0.0, value=float(cfg["mb_diaria"]), step=5.0,
                                           help="Valor pago por diária a cada motoboy.")
    with colD5:
        cfg["mb_custo_por_entrega"] = st.number_input("Custo por entrega (R$)", 0.0, value=float(cfg["mb_custo_por_entrega"]), step=0.5,
                                                      help="Adicional por entrega realizada.")
    with colD6:
        cfg["motoboys_minimos"] = st.number_input("Motoboys mínimos", 0, value=int(cfg["motoboys_minimos"]), step=1,
                                                  help="Mínimo operacional (sempre cobertos via diárias).")

    st.divider()
    cfg["mostrar_graficos"] = st.checkbox("Mostrar gráficos", value=bool(cfg["mostrar_graficos"]),
                                          help="Exibe gráficos de custos e métricas.")
    st.button("🔄 Resetar para padrão", on_click=lambda: st.session_state.update({"config": DEFAULTS.copy()}))
    save_button(cfg)

# -----------------------------
# Cálculos
# -----------------------------
fat = cfg["faturamento"]
ticket = cfg["ticket_medio"]
pedidos_mes = fat / ticket if ticket else 0

# Fixos
fixo_base = (
    cfg["fixo_aluguel"] + cfg["fixo_gerente"] + cfg["fixo_joao"] +
    cfg["fixo_util_basico"] + cfg["fixo_depreciacao"] + cfg["fixo_anotai"]
)

# Cozinheiros por faixa
if fat <= cfg["cook_t1_limite"]:
    cozinheiros = cfg["cook_t1_sal"]
elif fat <= cfg["cook_t2_limite"]:
    cozinheiros = cfg["cook_t2_sal"]
else:
    cozinheiros = cfg["cook_t3_sal"]

fixos_total = fixo_base + cozinheiros

# Variáveis (% do faturamento)
insumos = fat * cfg["pct_insumos"]
embalagens = fat * cfg["pct_embalagens"]
energia_extra = fat * cfg["pct_energia_extra"]
ifood = fat * cfg["pct_ifood"]
marketing = fat * cfg["pct_marketing"]

# -------- Motoboys (100% variáveis por demanda)
perc_fds = cfg["perc_fds"]
dias_uteis = cfg["dias_uteis"]
dias_fds = cfg["dias_fds"]
entregas_por_hora = cfg["entregas_por_hora"]
horas_semana = cfg["horas_semana"]
horas_fds = cfg["horas_fds"]
motoboys_min = cfg["motoboys_minimos"]

# Quebra de pedidos
pedidos_fds = pedidos_mes * perc_fds
pedidos_sem = pedidos_mes * (1 - perc_fds)

pedidos_dia_sem = (pedidos_sem / dias_uteis) if dias_uteis else 0
pedidos_dia_fds = (pedidos_fds / dias_fds) if dias_fds else 0

# Capacidade por motoboy/dia
cap_sem = entregas_por_hora * horas_semana
cap_fds = entregas_por_hora * horas_fds

motoboys_sem = max(motoboys_min, math.ceil(pedidos_dia_sem / cap_sem)) if cap_sem > 0 else motoboys_min
motoboys_fds = max(motoboys_min, math.ceil(pedidos_dia_fds / cap_fds)) if cap_fds > 0 else motoboys_min

# Custo dos motoboys: diárias + por entrega (todos)
diarias_sem = motoboys_sem * cfg["mb_diaria"] * dias_uteis
diarias_fds = motoboys_fds * cfg["mb_diaria"] * dias_fds
custo_por_entregas = pedidos_mes * cfg["mb_custo_por_entrega"]

custo_motoboys_total = diarias_sem + diarias_fds + custo_por_entregas

variaveis_total = insumos + embalagens + energia_extra + ifood + marketing + custo_motoboys_total
total_custos = fixos_total + variaveis_total
lucro = fat - total_custos
margem = (lucro / fat * 100) if fat > 0 else 0

# -----------------------------
# Saída
# -----------------------------
st.subheader("Resultado da Simulação")

tabela = pd.DataFrame({
    "Valores": [
        fat, round(pedidos_mes),
        fixos_total, insumos, embalagens, energia_extra, ifood, marketing,
        custo_motoboys_total, total_custos, lucro, margem
    ]},
    index=[
        "Faturamento", "Pedidos (estim.)",
        "Fixos (total)", "Insumos", "Embalagens", "Energia extra", "iFood/cartão", "Marketing",
        "Motoboys (total variável)", "Custos totais", "Lucro líquido", "Margem líquida (%)"
    ]
)

st.dataframe(
    tabela.style.format(lambda v: money(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v),
    use_container_width=True
)

# Métricas adicionais de operação dos motoboys
st.caption("Dimensionamento de motoboys (estimado)")
colx1, colx2, colx3, colx4 = st.columns(4)
colx1.metric("Pedidos/dia útil", f"{pedidos_dia_sem:.1f}")
colx2.metric("Pedidos/dia FDS", f"{pedidos_dia_fds:.1f}")
colx3.metric("Motoboys na semana", f"{motoboys_sem}")
colx4.metric("Motoboys no FDS", f"{motoboys_fds}")

if cfg["mostrar_graficos"]:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Custos (quebra)")
        chart_df = pd.DataFrame({
            "Categoria": ["Fixos", "Insumos", "Embalagens", "Energia extra", "iFood/cartão", "Marketing", "Motoboys"],
            "Valor": [fixos_total, insumos, embalagens, energia_extra, ifood, marketing, custo_motoboys_total]
        }).set_index("Categoria")
        st.bar_chart(chart_df)
    with c2:
        st.metric("Lucro líquido", money(lucro), help="Faturamento - (Fixos + Variáveis).")
        st.metric("Margem líquida", f"{margem:.1f}%", help="Lucro líquido / Faturamento.")
