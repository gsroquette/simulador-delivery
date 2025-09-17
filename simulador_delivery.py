# arquivo: simulador_delivery.py
import streamlit as st
import pandas as pd

# ==============================
# Funções auxiliares
# ==============================

def custo_cozinheiros(faturamento):
    if faturamento <= 50000:
        return 5000
    elif faturamento <= 150000:
        return 7500
    else:
        return 10000

def calc_motoboy_extras(faturamento, ticket_medio):
    pedidos = faturamento / ticket_medio
    if faturamento <= 45000:
        extras = 0
    elif faturamento <= 150000:
        extras = 2
    else:
        extras = 3

    # custo fixo por diária
    custo_diaria = extras * 60 * 12
    # entregas por extra motoboy (15/dia * 12 dias)
    entregas_extras = extras * 15 * 12
    custo_por_entrega = entregas_extras * 5
    custo_total = custo_diaria + custo_por_entrega
    return custo_total, extras, pedidos

def calcular(faturamento, ticket_medio):
    # Custos fixos base (sem cozinheiros)
    fixo_base = 11079.79  # aluguel + gerente + João + contas básicas + depreciação + 2 motoboys + Anota.i

    # Cozinheiros
    cozinheiros = custo_cozinheiros(faturamento)

    # Fixos totais
    fixos = fixo_base + cozinheiros

    # Variáveis (% do faturamento)
    insumos = faturamento * 0.333
    embalagens = faturamento * 0.02
    energia = faturamento * 0.015
    ifood = faturamento * 0.11
    marketing = faturamento * 0.02

    # Motoboys extras
    motoboy_extra, extras, pedidos = calc_motoboy_extras(faturamento, ticket_medio)

    # Totais
    variaveis = insumos + embalagens + energia + ifood + marketing + motoboy_extra
    custos_totais = fixos + variaveis
    lucro = faturamento - custos_totais
    margem = (lucro / faturamento * 100) if faturamento > 0 else 0

    return {
        "Faturamento": faturamento,
        "Pedidos": round(pedidos),
        "Fixos": round(fixos, 2),
        "Insumos": round(insumos, 2),
        "Embalagens": round(embalagens, 2),
        "Energia": round(energia, 2),
        "iFood": round(ifood, 2),
        "Marketing": round(marketing, 2),
        "Motoboy extra": round(motoboy_extra, 2),
        "Total Custos": round(custos_totais, 2),
        "Lucro": round(lucro, 2),
        "Margem %": round(margem, 1),
        "Extras usados": extras
    }

# ==============================
# Interface Streamlit
# ==============================

st.title("📊 Simulador Financeiro — Delivery de Petiscos")

# Entradas
faturamento = st.number_input("Faturamento (R$)", min_value=0, value=50000, step=1000)
ticket_medio = st.number_input("Ticket médio (R$)", min_value=1, value=70, step=1)

# Calcular
resultados = calcular(faturamento, ticket_medio)

st.subheader("Resultado da Simulação")
st.write(pd.DataFrame([resultados]).T.rename(columns={0: "Valores"}))

# Gráficos
st.subheader("Distribuição dos Custos")
custos = {
    "Fixos": resultados["Fixos"],
    "Insumos": resultados["Insumos"],
    "Embalagens": resultados["Embalagens"],
    "Energia": resultados["Energia"],
    "iFood": resultados["iFood"],
    "Marketing": resultados["Marketing"],
    "Motoboy extra": resultados["Motoboy extra"],
}
st.bar_chart(pd.DataFrame(custos, index=["Custos"]))

st.success(f"Lucro líquido: R$ {resultados['Lucro']:,}  |  Margem: {resultados['Margem %']}%")
