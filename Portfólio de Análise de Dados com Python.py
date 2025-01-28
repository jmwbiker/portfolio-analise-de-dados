import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# Criando dados fictícios
np.random.seed(42)

# Dados de vendas mensais
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
vendas = np.random.normal(10000, 2000, 12)
metas = np.full(12, 10000)

df_vendas = pd.DataFrame({
    'Mês': meses,
    'Vendas': vendas,
    'Meta': metas
})

# Dados de produtos
produtos = ['Produto A', 'Produto B', 'Produto C', 'Produto D', 'Produto E']
vendas_produtos = np.random.normal(5000, 1000, 5)
custos_produtos = vendas_produtos * np.random.uniform(0.4, 0.6, 5)

df_produtos = pd.DataFrame({
    'Produto': produtos,
    'Vendas': vendas_produtos,
    'Custos': custos_produtos
})
df_produtos['Margem'] = df_produtos['Vendas'] - df_produtos['Custos']

# Dados de clientes por região
regioes = ['Norte', 'Sul', 'Leste', 'Oeste', 'Centro']
clientes = np.random.randint(1000, 3000, 5)
satisfacao = np.random.uniform(7, 9.5, 5)

df_regioes = pd.DataFrame({
    'Região': regioes,
    'Clientes': clientes,
    'Satisfação': satisfacao
})

# Criando subplots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Vendas Mensais vs Meta',
        'Análise de Produtos',
        'Distribuição de Clientes por Região',
        'Satisfação por Região',
        'Margem de Lucro por Produto',
        'Tendência de Vendas'
    ),
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'pie'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'scatter'}]]
)

# 1. Gráfico de Linha: Vendas vs Meta
fig.add_trace(
    go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas'],
               name='Vendas', mode='lines+markers',
               line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df_vendas['Mês'], y=df_vendas['Meta'],
               name='Meta', line=dict(dash='dash', color='gray')),
    row=1, col=1
)

# 2. Gráfico de Barras: Análise de Produtos com cores diferentes por produto
fig.add_trace(
    go.Bar(x=df_produtos['Produto'], y=df_produtos['Vendas'],
           name='Vendas Produtos',
           marker_color=px.colors.qualitative.Set2[:len(df_produtos)]),
    row=1, col=2
)

# 3. Gráfico de Pizza: Distribuição de Clientes
fig.add_trace(
    go.Pie(labels=df_regioes['Região'], values=df_regioes['Clientes'],
           name='Clientes por Região',
           marker_colors=px.colors.qualitative.Pastel),
    row=2, col=1
)

# 4. Gráfico de Barras: Satisfação por Região com cores personalizadas
fig.add_trace(
    go.Bar(x=df_regioes['Região'], y=df_regioes['Satisfação'],
           name='Satisfação',
           marker_color=px.colors.qualitative.Bold[:len(df_regioes)]),
    row=2, col=2
)

# 5. Gráfico de Barras: Margem de Lucro com cores diferentes para cada produto
fig.add_trace(
    go.Bar(x=df_produtos['Produto'], y=df_produtos['Margem'],
           name='Margem de Lucro',
           marker_color=px.colors.qualitative.Dark24[:len(df_produtos)]),
    row=3, col=1
)

# 6. Gráfico de Linha: Tendência de Vendas com linha de tendência usando IA
X = np.arange(len(df_vendas['Vendas'])).reshape(-1, 1)
y = df_vendas['Vendas']
model = LinearRegression()
model.fit(X, y)
predicted = model.predict(X)

fig.add_trace(
    go.Scatter(x=df_vendas['Mês'], y=df_vendas['Vendas'],
               name='Vendas Mensais', mode='lines+markers',
               line=dict(color='purple')),
    row=3, col=2
)
fig.add_trace(
    go.Scatter(x=df_vendas['Mês'], y=predicted,
               name='Tendência (IA)', line=dict(dash='dash', color='orange')),
    row=3, col=2
)

# Atualização do layout
fig.update_layout(height=1200, width=1000, showlegend=True,
                  title_text="Portfólio de Análise de Dados com IA",
                  template="plotly_white")

# Configurações adicionais de layout
fig.update_yaxes(title_text="Valor (R$)", row=1, col=1)
fig.update_yaxes(title_text="Valor (R$)", row=1, col=2)
fig.update_yaxes(title_text="Nível de Satisfação", row=2, col=2)
fig.update_yaxes(title_text="Margem (R$)", row=3, col=1)
fig.update_yaxes(title_text="Valor (R$)", row=3, col=2)

# Exibir o gráfico
fig.show()

# Gerar análises estatísticas
analises = {
    'Vendas Totais': f"R$ {df_vendas['Vendas'].sum():,.2f}",
    'Média de Vendas': f"R$ {df_vendas['Vendas'].mean():,.2f}",
    'Produto Mais Rentável': df_produtos.loc[df_produtos['Margem'].idxmax(), 'Produto'],
    'Região com Mais Clientes': df_regioes.loc[df_regioes['Clientes'].idxmax(), 'Região'],
    'Satisfação Média': f"{df_regioes['Satisfação'].mean():.1f}/10"
}

# Imprimir análises
print("\nAnálises Principais:")
for metrica, valor in analises.items():
    print(f"{metrica}: {valor}")
