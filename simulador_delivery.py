# simulador_delivery.py
import json
import io
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulador Financeiro — Delivery", page_icon="📊", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def money(x): 
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def load_config(defaults: dict):
    """Tenta carregar config enviada pelo usuário; se não houver, usa defaults."""
    if "config" not in st.session_state:
        st.session_state.config = defaults.copy()
    return st.session_state.config

def save_button(config_dict):
    b = io.BytesIO(json.dumps(config_dict, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button("⬇️ Baixar configuração (JSON)", data=b, file_name="config_delivery.json", mime="application/json")

# -----------------------------
# Defaults do modelo
# -----------------------------
DEFAULTS = {
    "ticket_medio": 70.0,
    "faturamento": 50000.0,

    # FIXOS base (sem cozinheiros)
    "fixo_aluguel": 2000.0,
    "fixo_gerente": 3000.0,
    "fixo_joao": 1000.0,
    "fixo_util_basico": 1000.0,      # luz/água/IPTU/tel/internet base
    "fixo_depreciacao": 200.0,
    "fixo_motoboys_fixos_qtd": 2,
    "fixo_motoboy_salario": 1800.0,  # por motoboy fixo/mês
    "fixo_anotai": 279.79,

    # COZINHEIROS (escada por faturamento)
    "cook_t1_limite": 50000.0, "cook_t1_sal": 5000.0,  # até este limite
    "cook_t2_limite": 150000.0, "cook_t2_sal": 7500.0, # até este limite
    "cook_t3_sal": 10000.0,                            # acima disso

    # VARIÁVEIS (% do faturamento)
    "pct_insumos": 0.333,
    "pct_embalagens": 0.02,
    "pct_energia_extra": 0.015,
    "pct_ifood": 0.11,
    "pct_marketing": 0.02,

    # MOTOBOYS EXTRAS (por fim de semana)
    "mb_threshold1": 45000.0,     # até aqui: 0 extras
    "mb_threshold2": 150000.0,    # até aqui: 2 extras
    "mb_extras_t2": 2,            # qtd extras no t2 (semanas)
    "mb_extras_t3": 3,            # qtd extras no t3 (semanas)
    "mb_diaria": 60.0,            # R$ por diária
    "mb_fds_dias_mes": 12,        # sext/sáb/dom no mês
    "mb_por_entrega": 5.0,        # R$ por entrega
    "mb_entregas_por_extra_dia": 15,

    # Exibição
    "mostrar_graficos": True
}

# -----------------------------
# Carga / Edição de Config
# -----------------------------
st.title("📊 Simulador Financeiro — Delivery de Petiscos")

with st.sidebar:
    st.header("⚙️ Configuração")
    cfg = load_config(DEFAULTS)

    # Upload de JSON opcional
    up = st.file_uploader("Carregar configuração (JSON)", type=["json"])
    if up:
        cfg = json.load(up)
        st.session_state.config = cfg

    st.divider()
    st.caption("Entradas principais")
    cfg["faturamento"] = st.number_input("Faturamento (R$)", 0.0, step=1000.0, value=float(cfg["faturamento"]))
    cfg["ticket_medio"] = st.number_input("Ticket médio (R$)", 1.0, step=1.0, value=float(cfg["ticket_medio"]))

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
        cfg["fixo_motoboys_fixos_qtd"] = st.number_input("Motoboys fixos (qtd)", 0, value=int(cfg["fixo_motoboys_fixos_qtd"]), step=1)
        cfg["fixo_motoboy_salario"] = st.number_input("Salário por motoboy fixo", 0.0, value=float(cfg["fixo_motoboy_salario"]), step=50.0)

    st.divider()
    st.caption("Cozinheiros (escala por faturamento)")
    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        cfg["cook_t1_limite"] = st.number_input("Limite T1 (2 cozinheiros)", 0.0, value=float(cfg["cook_t1_limite"]), step=1000.0)
        cfg["cook_t1_sal"] = st.number_input("Custo T1 (2 coz.)", 0.0, value=float(cfg["cook_t1_sal"]), step=100.0)
    with colc2:
        cfg["cook_t2_limite"] = st.number_input("Limite T2 (3 cozinheiros)", 0.0, value=float(cfg["cook_t2_limite"]), step=1000.0)
        cfg["cook_t2_sal"] = st.number_input("Custo T2 (3 coz.)", 0.0, value=float(cfg["cook_t2_sal"]), step=100.0)
    with colc3:
        cfg["cook_t3_sal"] = st.number_input("Custo T3 (4 coz.)", 0.0, value=float(cfg["cook_t3_sal"]), step=100.0)

    st.divider()
    st.caption("Percentuais variáveis")
    colp1, colp2, colp3, colp4, colp5 = st.columns(5)
    with colp1:
        cfg["pct_insumos"] = st.number_input("Insumos (%)", 0.0, 1.0, step=0.005, value=float(cfg["pct_insumos"]))
    with colp2:
        cfg["pct_embalagens"] = st.number_input("Embalagens (%)", 0.0, 1.0, step=0.005, value=float(cfg["pct_embalagens"]))
    with colp3:
        cfg["pct_energia_extra"] = st.number_input("Energia extra (%)", 0.0, 1.0, step=0.005, value=float(cfg["pct_energia_extra"]))
    with colp4:
        cfg["pct_ifood"] = st.number_input("iFood/cartão (%)", 0.0, 1.0, step=0.005, value=float(cfg["pct_ifood"]))
    with colp5:
        cfg["pct_marketing"] = st.number_input("Marketing (%)", 0.0, 1.0, step=0.005, value=float(cfg["pct_marketing"]))

    st.divider()
    st.caption("Motoboys extras (fim de semana)")
    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        cfg["mb_threshold1"] = st.number_input("Limite T1 (0 extras)", 0.0, value=float(cfg["mb_threshold1"]), step=1000.0)
        cfg["mb_threshold2"] = st.number_input("Limite T2 (2 extras)", 0.0, value=float(cfg["mb_threshold2"]), step=1000.0)
    with colm2:
        cfg["mb_extras_t2"] = st.number_input("Extras no T2 (qtd)", 0, value=int(cfg["mb_extras_t2"]), step=1)
        cfg["mb_extras_t3"] = st.number_input("Extras no T3 (qtd)", 0, value=int(cfg["mb_extras_t3"]), step=1)
    with colm3:
        cfg["mb_diaria"] = st.number_input("Diária por extra (R$)", 0.0, value=float(cfg["mb_diaria"]), step=5.0)
        cfg["mb_fds_dias_mes"] = st.number_input("Dias de FDS no mês", 0, value=int(cfg["mb_fds_dias_mes"]), step=1)

    colm4, colm5 = st.columns(2)
    with colm4:
        cfg["mb_por_entrega"] = st.number_input("Custo por entrega (R$)", 0.0, value=float(cfg["mb_por_entrega"]), step=1.0)
    with colm5:
        cfg["mb_entregas_por_extra_dia"] = st.number_input("Entregas/extra/dia de FDS", 0, value=int(cfg["mb_entregas_por_extra_dia"]), step=1)

    st.divider()
    cfg["mostrar_graficos"] = st.checkbox("Mostrar gráficos", value=bool(cfg["mostrar_graficos"]))

    st.button("🔄 Resetar para padrão", on_click=lambda: st.session_state.update({"config": DEFAULTS.copy()}))
    save_button(cfg)

# -----------------------------
# Cálculos
# -----------------------------
faturamento = cfg["faturamento"]
ticket_medio = cfg["ticket_medio"]
pedidos = faturamento / ticket_medio if ticket_medio else 0

# Fixos base
fixo_base = (
    cfg["fixo_aluguel"] + cfg["fixo_gerente"] + cfg["fixo_joao"] +
    cfg["fixo_util_basico"] + cfg["fixo_depreciacao"] + cfg["fixo_anotai"] +
    cfg["fixo_motoboys_fixos_qtd"] * cfg["fixo_motoboy_salario"]
)

# Cozinheiros por faixa
if faturamento <= cfg["cook_t1_limite"]:
    cozinheiros = cfg["cook_t1_sal"]
elif faturamento <= cfg["cook_t2_limite"]:
    cozinheiros = cfg["cook_t2_sal"]
else:
    cozinheiros = cfg["cook_t3_sal"]

fixos_total = fixo_base + cozinheiros

# Variáveis em %
insumos = faturamento * cfg["pct_insumos"]
embalagens = faturamento * cfg["pct_embalagens"]
energia_extra = faturamento * cfg["pct_energia_extra"]
ifood = faturamento * cfg["pct_ifood"]
marketing = faturamento * cfg["pct_marketing"]

# Motoboys extras por faixa
if faturamento <= cfg["mb_threshold1"]:
    extras = 0
elif faturamento <= cfg["mb_threshold2"]:
    extras = cfg["mb_extras_t2"]
else:
    extras = cfg["mb_extras_t3"]

mb_fixos_extras = extras * cfg["mb_diaria"] * cfg["mb_fds_dias_mes"]
mb_entregas_mes = extras * cfg["mb_entregas_por_extra_dia"] * cfg["mb_fds_dias_mes"]
mb_var_extras = mb_entregas_mes * cfg["mb_por_entrega"]
mb_total_extras = mb_fixos_extras + mb_var_extras

variaveis_total = insumos + embalagens + energia_extra + ifood + marketing + mb_total_extras
total_custos = fixos_total + variaveis_total
lucro = faturamento - total_custos
margem = (lucro / faturamento * 100) if faturamento > 0 else 0

# -----------------------------
# Saída
# -----------------------------
st.subheader("Resultado da Simulação")
tabela = pd.DataFrame({
    "Valores": [
        faturamento, round(pedidos),
        fixos_total, insumos, embalagens, energia_extra, ifood, marketing,
        mb_total_extras, total_custos, lucro, margem
    ]},
    index=[
        "Faturamento", "Pedidos (estim.)",
        "Fixos (total)", "Insumos", "Embalagens", "Energia extra", "iFood/cartão", "Marketing",
        "Motoboys extra (total)", "Custos totais", "Lucro líquido", "Margem líquida (%)"
    ]
)
st.dataframe(tabela.style.format(lambda v: money(v) if isinstance(v, (int, float)) and "Margem" not in str(v) else v),
             use_container_width=True)

if cfg["mostrar_graficos"]:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Custos (quebra)")
        pie_df = pd.DataFrame({
            "Categoria": ["Fixos", "Insumos", "Embalagens", "Energia extra", "iFood/cartão", "Marketing", "Motoboys extra"],
            "Valor": [fixos_total, insumos, embalagens, energia_extra, ifood, marketing, mb_total_extras]
        })
        st.bar_chart(pie_df.set_index("Categoria"))
    with c2:
        st.metric("Lucro líquido", money(lucro), help="Faturamento - (Fixos + Variáveis)")
        st.metric("Margem líquida", f"{margem:.1f}%")

st.info(
    "Dica: ajuste os percentuais e salários na barra lateral. "
    "Use **⬇️ Baixar configuração (JSON)** para salvar seus parâmetros e depois faça upload quando quiser replicar."
)
