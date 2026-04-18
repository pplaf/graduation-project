"""
语义网络模块 - 共现网络构建与社区发现
"""
import os
import sys
import json
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from loguru import logger

import networkx as nx
from community import community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.font_manager as fm
FONT_TITLE = fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10.5)
FONT_BODY = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=10.5)
FONT_BODY_SMALL = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=9)


def build_cooccurrence_network(tokens_list, min_cooc=2, top_n=80):
    """从分词列表构建共现网络"""
    import re

    # 无意义词过滤规则
    def is_valid_token(w):
        if len(w) < 2:                          # 单字符全过滤
            return False
        if re.fullmatch(r'[\d\s\.\,\!\?\;\:\-\/\\\(\)\[\]\{\}\"\']+', w):
            return False                         # 纯数字/标点
        if re.fullmatch(r'[^\u4e00-\u9fffa-zA-Z0-9]+', w):
            return False                         # 全是特殊符号
        if re.search(r'https?|www\.|\.com|\.cn|youtube|bilibili', w, re.I):
            return False                         # URL残留
        if w in {'AI', 'ai', '-', '/', ':', '.', ',', '(', ')', '（', '）',
                 '！', '？', '、', '。', '，', '；', '：', '"', '"', '…',
                 'http', 'www', 'com', 'cn', 'mp4', 'jpg', 'png'}:
            return False
        return True

    # 统计词频，先过滤无意义词
    word_freq = Counter()
    for tokens in tokens_list:
        word_freq.update(t for t in tokens if is_valid_token(t))

    top_words = set(w for w, _ in word_freq.most_common(top_n))

    # 构建共现关系
    cooc_counter = Counter()
    for tokens in tokens_list:
        filtered = [t for t in tokens if t in top_words]
        for w1, w2 in combinations(set(filtered), 2):
            pair = tuple(sorted([w1, w2]))
            cooc_counter[pair] += 1

    # 构建网络
    G = nx.Graph()
    for (w1, w2), weight in cooc_counter.items():
        if weight >= min_cooc:
            G.add_edge(w1, w2, weight=weight)

    for node in G.nodes():
        G.nodes[node]['freq'] = word_freq.get(node, 0)

    logger.info(f"共现网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def detect_communities(G):
    """Louvain社区发现"""
    partition = community_louvain.best_partition(G, weight='weight')
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    logger.info(f"发现 {len(communities)} 个社区")
    for cid, nodes in sorted(communities.items(), key=lambda x: -len(x[1])):
        logger.info(f"  社区{cid}: {len(nodes)} 个词 - {', '.join(nodes[:5])}...")

    return partition, dict(communities)


def compute_network_metrics(G):
    """计算网络指标"""
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_clustering": round(nx.average_clustering(G), 4),
    }

    if nx.is_connected(G):
        metrics["avg_path_length"] = round(nx.average_shortest_path_length(G), 4)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        sub = G.subgraph(largest_cc)
        metrics["avg_path_length"] = round(nx.average_shortest_path_length(sub), 4)

    # 中心性
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    metrics["top_degree_centrality"] = sorted(degree_cent.items(), key=lambda x: -x[1])[:10]
    metrics["top_betweenness_centrality"] = sorted(betweenness_cent.items(), key=lambda x: -x[1])[:10]

    return metrics


def plot_network(G, partition, title, filepath, top_n=50):
    """绘制网络图"""
    # 只绘制top_n节点
    nodes_by_freq = sorted(G.nodes(data=True), key=lambda x: -x[1].get('freq', 0))
    top_nodes = [n for n, _ in nodes_by_freq[:top_n]]
    sub = G.subgraph(top_nodes)

    pos = nx.spring_layout(sub, k=1.5, iterations=50, seed=42)

    # 按社区着色
    colors = [partition.get(n, 0) for n in sub.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    node_sizes = [sub.nodes[n].get('freq', 1) * 30 for n in sub.nodes()]

    nx.draw_networkx(
        sub, pos, ax=ax,
        node_color=colors, node_size=node_sizes,
        cmap=plt.cm.Set3, alpha=0.8,
        font_size=9, font_family='SimSun',
        edge_color='#cccccc', width=[sub[u][v]['weight'] * 0.3 for u, v in sub.edges()],
    )
    ax.set_title(title, fontproperties=FONT_TITLE)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"网络图已保存: {filepath}")


def run_semantic_network():
    """运行完整语义网络分析（三端）"""
    # 加载预处理数据
    proc_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    with open(proc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prop_tokens = [d.get('propagator_tokens', []) for d in data]
    recv_tokens = [d.get('receiver_tokens', []) for d in data]
    asr_tokens = [d.get('asr_tokens', []) for d in data if d.get('asr_tokens')]

    def build_and_analyze(tokens_list, label_cn, label_en, fig_num, filename):
        logger.info(f"=== {label_cn}共现网络 ===")
        G = build_cooccurrence_network(tokens_list, min_cooc=2, top_n=80)
        partition, communities = detect_communities(G)
        metrics = compute_network_metrics(G)
        plot_network(G, partition,
                     f"图{fig_num} {label_cn}语义共现网络\nFig.{fig_num} {label_en} Semantic Co-occurrence Network",
                     os.path.join(OUTPUT_DIR, filename))
        return {
            "metrics": {k: v for k, v in metrics.items()
                        if k not in ('top_degree_centrality', 'top_betweenness_centrality')},
            "top_degree": [(w, round(c, 4)) for w, c in metrics.get('top_degree_centrality', [])],
            "top_betweenness": [(w, round(c, 4)) for w, c in metrics.get('top_betweenness_centrality', [])],
            "communities": {str(k): v for k, v in communities.items()},
        }

    # 三端网络
    result = {}
    result["propagator"] = build_and_analyze(prop_tokens, "传播端", "Propagator", 8, "network_propagator.png")
    result["receiver"] = build_and_analyze(recv_tokens, "接收端", "Receiver", 9, "network_receiver.png")

    if asr_tokens:
        result["asr"] = build_and_analyze(asr_tokens, "ASR音频转写端", "ASR Transcript", 10, "network_asr.png")
    else:
        logger.warning("无ASR分词数据，跳过ASR语义网络")

    with open(os.path.join(RESULTS_DIR, "semantic_network.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("语义网络分析结果已保存")

    return result


if __name__ == "__main__":
    run_semantic_network()
