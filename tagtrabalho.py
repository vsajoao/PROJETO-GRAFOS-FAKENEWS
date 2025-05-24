import pandas as pd
import gzip
import networkx as nx
import matplotlib.pyplot as plt

def load_edgelist(filename):
    with gzip.open(filename, 'rt') as f:
        data = [line.strip().split() for line in f]
    df = pd.DataFrame(data, columns=['user1', 'user2', 'weight'])
    df['weight'] = df['weight'].astype(int)
    return df

# Etapa 1 - Carregar os três tipos de interação
df_retweet = load_edgelist('higgs-retweet_network.edgelist.gz')
df_reply = load_edgelist('higgs-reply_network.edgelist.gz')
df_mention = load_edgelist('higgs-mention_network.edgelist.gz')

# Juntar todos os dados
df_all = pd.concat([df_retweet, df_reply, df_mention])

# Agrupar somando os pesos de múltiplas interações
df_all = df_all.groupby(['user1', 'user2']).sum().reset_index()

# Salvar para uso posterior
df_all.to_csv('twitter_network.csv', index=False)

# =========================================================
# Etapa 2 - Construir o grafo direcionado e ponderado
# =========================================================

# Carregar os dados do CSV (poderia pular esta etapa se seguir direto do df_all)
df = pd.read_csv('twitter_network.csv')

# Criar o grafo direcionado
G = nx.DiGraph()

# Adicionar arestas com pesos
for _, row in df.iterrows():
    G.add_edge(row['user1'], row['user2'], weight=row['weight'])

# Mostrar algumas informações básicas do grafo
print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")

# (Opcional) Mostrar visualização simples do grafo
plt.figure(figsize=(10, 7))
nx.draw(G, with_labels=False, node_size=20, edge_color='gray')
plt.title("Visualização simples do grafo de interações")
plt.show()
