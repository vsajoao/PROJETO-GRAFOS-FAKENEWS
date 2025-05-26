import pandas as pd
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities, girvan_newman
import warnings
warnings.filterwarnings('ignore')

# ⚡ CONFIGURAÇÃO DE VELOCIDADE
FAST_MODE = True  # Mude para False se quiser análise completa (mais lenta)
MAX_NODES = 1000 if FAST_MODE else 10000
MAX_EDGES = 3000 if FAST_MODE else 50000

print(f"🚀 Modo: {'RÁPIDO' if FAST_MODE else 'COMPLETO'}")
print(f"Limites: {MAX_NODES} nós, {MAX_EDGES} arestas")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_edgelist(filename):
    """Load and process edgelist from gzipped file"""
    try:
        with gzip.open(filename, 'rt') as f:
            data = [line.strip().split() for line in f]
        df = pd.DataFrame(data, columns=['user1', 'user2', 'weight'])
        df['weight'] = df['weight'].astype(int)
        return df
    except FileNotFoundError:
        print(f"File {filename} not found. Using sample data for demonstration.")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample Twitter-like network data for demonstration"""
    np.random.seed(42)
    n_users = 300 if FAST_MODE else 500
    n_edges = 1000 if FAST_MODE else 2000
    
    users = [f"user_{i}" for i in range(n_users)]
    
    # Create realistic network structure
    edges = []
    for i in range(n_edges):
        user1 = np.random.choice(users)
        user2 = np.random.choice(users)
        if user1 != user2:
            weight = np.random.randint(1, 10)
            edges.append([user1, user2, weight])
    
    return pd.DataFrame(edges, columns=['user1', 'user2', 'weight'])

# =========================================================
# ETAPA 1: COLETA E PREPARAÇÃO DOS DADOS
# =========================================================
print("=== ETAPA 1: COLETA E PREPARAÇÃO DOS DADOS ===")

# Tentar carregar dados reais, usar dados de exemplo se não encontrar
try:
    df_retweet = load_edgelist('higgs-retweet_network.edgelist.gz')
    df_reply = load_edgelist('higgs-reply_network.edgelist.gz')
    df_mention = load_edgelist('higgs-mention_network.edgelist.gz')
    print("Dados reais carregados com sucesso!")
except:
    print("Usando dados de exemplo para demonstração...")
    df_retweet = generate_sample_data()
    df_reply = generate_sample_data()
    df_mention = generate_sample_data()

# Juntar todos os dados
df_all = pd.concat([df_retweet, df_reply, df_mention])
df_all = df_all.groupby(['user1', 'user2']).sum().reset_index()

# OTIMIZAÇÃO: Filtrar apenas interações com peso significativo e limitar tamanho
min_weight = 2  # Filtrar interações fracas
df_all = df_all[df_all['weight'] >= min_weight]

# Limitar a um subconjunto se muito grande
if len(df_all) > MAX_EDGES:
    print(f"Dataset muito grande ({len(df_all)} arestas). Usando subset de {MAX_EDGES} arestas com maiores pesos.")
    df_all = df_all.nlargest(MAX_EDGES, 'weight')

# Filtrar usuários com poucas conexões para reduzir ruído
user_counts = pd.concat([df_all['user1'], df_all['user2']]).value_counts()
active_users = user_counts[user_counts >= 2].index.tolist()  # Pelo menos 2 interações
df_all = df_all[df_all['user1'].isin(active_users) & df_all['user2'].isin(active_users)]

print(f"Total de interações após filtros: {len(df_all)}")
print(f"Usuários únicos: {len(set(df_all['user1'].tolist() + df_all['user2'].tolist()))}")

# =========================================================
# ETAPA 2: CONSTRUÇÃO DO GRAFO
# =========================================================
print("\n=== ETAPA 2: CONSTRUÇÃO DO GRAFO ===")

# Criar grafo direcionado
G = nx.DiGraph()

# Adicionar arestas com pesos
for _, row in df_all.iterrows():
    G.add_edge(row['user1'], row['user2'], weight=row['weight'])

print(f"Número de nós: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")
print(f"Densidade do grafo: {nx.density(G):.6f}")

# =========================================================
# ETAPA 3: ALGORITMOS DE DETECÇÃO
# =========================================================
print("\n=== ETAPA 3: ANÁLISE COM ALGORITMOS DE GRAFOS ===")

# 3.1 PAGERANK - Identificar usuários mais influentes
print("\n3.1 Calculando PageRank...")
pagerank_scores = nx.pagerank(G, alpha=0.85, weight='weight')
top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 usuários mais influentes (PageRank):")
for i, (user, score) in enumerate(top_pagerank, 1):
    print(f"{i}. {user}: {score:.6f}")

# 3.2 DETECÇÃO DE COMUNIDADES
print("\n3.2 Detectando comunidades...")
# Usar grafo não-direcionado para detecção de comunidades
G_undirected = G.to_undirected()

# OTIMIZAÇÃO: Usar apenas componente gigante se grafo for desconectado
if not nx.is_connected(G_undirected):
    # Pegar apenas o maior componente conectado
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    G_undirected = G_undirected.subgraph(largest_cc).copy()
    print(f"Usando maior componente conectado: {len(largest_cc)} nós")

communities = list(greedy_modularity_communities(G_undirected))

print(f"Número de comunidades detectadas: {len(communities)}")
print("Tamanho das 5 maiores comunidades:")
community_sizes = sorted([len(c) for c in communities], reverse=True)
for i, size in enumerate(community_sizes[:5], 1):
    print(f"Comunidade {i}: {size} usuários")

# 3.3 MEDIDAS DE CENTRALIDADE
print("\n3.3 Calculando medidas de centralidade...")

# Centralidade de grau (rápida)
degree_centrality = nx.degree_centrality(G)
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# OTIMIZAÇÃO: Calcular betweenness apenas para subset de nós importantes se grafo for muito grande
if G.number_of_nodes() > 1000:
    print("Grafo grande detectado. Calculando betweenness para top 200 nós por grau...")
    # Pegar apenas os top nós por grau para calcular betweenness
    top_nodes_by_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:200]
    important_nodes = [node for node, _ in top_nodes_by_degree]
    betweenness_centrality = nx.betweenness_centrality_subset(G, sources=important_nodes, targets=important_nodes, weight='weight')
else:
    print("Calculando centralidade de intermediação...")
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# OTIMIZAÇÃO: Closeness centralidade também pode ser limitada
if G.number_of_nodes() > 1000:
    print("Calculando closeness centrality para subset...")
    closeness_centrality = {}
    for node in important_nodes:
        try:
            closeness_centrality[node] = nx.closeness_centrality(G, u=node, distance='weight')
        except:
            closeness_centrality[node] = 0
else:
    print("Calculando centralidade de proximidade...")
    closeness_centrality = nx.closeness_centrality(G, distance='weight')

top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 por Centralidade de Grau:")
for user, score in top_degree:
    print(f"  {user}: {score:.6f}")

print("\nTop 5 por Centralidade de Intermediação:")
for user, score in top_betweenness:
    print(f"  {user}: {score:.6f}")

print("\nTop 5 por Centralidade de Proximidade:")
for user, score in top_closeness:
    print(f"  {user}: {score:.6f}")

# =========================================================
# ETAPA 4: IDENTIFICAÇÃO DE POTENCIAIS ESPALHADORES
# =========================================================
print("\n=== ETAPA 4: IDENTIFICAÇÃO DE ESPALHADORES DE FAKE NEWS ===")

# Combinar métricas para identificar potenciais espalhadores
def identify_potential_spreaders(pagerank_dict, degree_dict, betweenness_dict, top_n=10):
    """Identifica potenciais espalhadores baseado em múltiplas métricas"""
    all_users = set(pagerank_dict.keys())
    
    # Normalizar scores
    max_pr = max(pagerank_dict.values())
    max_deg = max(degree_dict.values())
    max_bet = max(betweenness_dict.values())
    
    combined_scores = {}
    for user in all_users:
        pr_norm = pagerank_dict[user] / max_pr
        deg_norm = degree_dict[user] / max_deg
        bet_norm = betweenness_dict[user] / max_bet if max_bet > 0 else 0
        
        # Score combinado (pode ajustar os pesos)
        combined_scores[user] = 0.4 * pr_norm + 0.3 * deg_norm + 0.3 * bet_norm
    
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

potential_spreaders = identify_potential_spreaders(
    pagerank_scores, degree_centrality, betweenness_centrality
)

print("Top 10 Potenciais Espalhadores de Fake News:")
for i, (user, score) in enumerate(potential_spreaders, 1):
    print(f"{i}. {user}: Score combinado {score:.6f}")

# =========================================================
# ETAPA 5: VISUALIZAÇÕES
# =========================================================
print("\n=== ETAPA 5: VISUALIZAÇÕES ===")

# 5.1 Distribuição de graus
plt.figure(figsize=(15, 10))

# Subplot 1: Distribuição de graus
plt.subplot(2, 3, 1)
degrees = [d for n, d in G.degree()]
plt.hist(degrees, bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribuição de Graus')
plt.xlabel('Grau')
plt.ylabel('Frequência')
plt.yscale('log')

# Subplot 2: Distribuição de PageRank
plt.subplot(2, 3, 2)
pr_values = list(pagerank_scores.values())
plt.hist(pr_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribuição PageRank')
plt.xlabel('Score PageRank')
plt.ylabel('Frequência')
plt.yscale('log')

# Subplot 3: Top usuários por PageRank
plt.subplot(2, 3, 3)
users_pr = [user[:10] + '...' for user, _ in top_pagerank[:5]]
scores_pr = [score for _, score in top_pagerank[:5]]
plt.barh(users_pr, scores_pr)
plt.title('Top 5 PageRank')
plt.xlabel('Score')

# Subplot 4: Centralidades comparadas
plt.subplot(2, 3, 4)
top_users = [user for user, _ in top_pagerank[:5]]
deg_scores = [degree_centrality[user] for user in top_users]
bet_scores = [betweenness_centrality[user] for user in top_users]
close_scores = [closeness_centrality[user] for user in top_users]

x = np.arange(len(top_users))
width = 0.25

plt.bar(x - width, deg_scores, width, label='Grau', alpha=0.8)
plt.bar(x, bet_scores, width, label='Intermediação', alpha=0.8)
plt.bar(x + width, close_scores, width, label='Proximidade', alpha=0.8)

plt.title('Comparação de Centralidades')
plt.xlabel('Usuários (Top PageRank)')
plt.ylabel('Score de Centralidade')
plt.xticks(x, [u[:8] + '...' for u in top_users], rotation=45)
plt.legend()

# Subplot 5: Tamanho das comunidades
plt.subplot(2, 3, 5)
plt.bar(range(1, min(11, len(community_sizes)+1)), community_sizes[:10])
plt.title('Tamanho das Comunidades')
plt.xlabel('Comunidade')
plt.ylabel('Número de Usuários')

# Subplot 6: Rede das principais interações
plt.subplot(2, 3, 6)
# Criar subgrafo com apenas os top usuários
top_20_users = [user for user, _ in potential_spreaders[:20]]
G_sub = G.subgraph(top_20_users)

pos = nx.spring_layout(G_sub, k=1, iterations=50)
node_sizes = [pagerank_scores[node] * 10000 for node in G_sub.nodes()]

nx.draw(G_sub, pos, node_size=node_sizes, node_color='red', 
        alpha=0.6, with_labels=False, edge_color='gray', arrows=True)
plt.title('Rede dos Principais Espalhadores')

plt.tight_layout()
plt.show()

# =========================================================
# ETAPA 6: ANÁLISE DE RESULTADOS
# =========================================================
print("\n=== ETAPA 6: ANÁLISE DE RESULTADOS ===")

print(f"\n📊 RESUMO DA ANÁLISE:")
print(f"• Rede analisada com {G.number_of_nodes()} usuários e {G.number_of_edges()} interações")
print(f"• Densidade da rede: {nx.density(G):.6f} (rede esparsa)")
print(f"• {len(communities)} comunidades detectadas")
print(f"• Usuário mais influente (PageRank): {top_pagerank[0][0]}")
print(f"• Maior intermediador: {top_betweenness[0][0]}")

print(f"\n🎯 POTENCIAIS ESPALHADORES IDENTIFICADOS:")
print("Os usuários com maior potencial para espalhar fake news são aqueles que combinam:")
print("• Alto PageRank (influência)")
print("• Alta centralidade de grau (muitas conexões)")
print("• Alta centralidade de intermediação (ponte entre comunidades)")

print(f"\n💡 INSIGHTS PARA DETECÇÃO DE FAKE NEWS:")
print("1. Monitorar usuários com score combinado alto")
print("2. Focar em usuários que conectam diferentes comunidades")
print("3. Analisar padrões de disseminação rápida")
print("4. Verificar usuários com muitos retweets/mentions")

# Salvar resultados principais
results_df = pd.DataFrame({
    'user': [user for user, _ in potential_spreaders],
    'combined_score': [score for _, score in potential_spreaders],
    'pagerank': [pagerank_scores[user] for user, _ in potential_spreaders],
    'degree_centrality': [degree_centrality[user] for user, _ in potential_spreaders],
    'betweenness_centrality': [betweenness_centrality[user] for user, _ in potential_spreaders]
})

results_df.to_csv('potential_fake_news_spreaders.csv', index=False)
print(f"\n✅ Resultados salvos em 'potential_fake_news_spreaders.csv'")

print("\n" + "="*60)
print("ANÁLISE COMPLETA! 🎉")
print("="*60)
