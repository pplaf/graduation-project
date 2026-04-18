"""
主题建模模块 - BERTopic(CPU) + LDA + 主题情感聚合
使用 paraphrase-multilingual-MiniLM-L12-v2 轻量embedding
三套独立模型：传播端 / 接收端 / ASR转写文本
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.preprocess.text_processor import init_jieba, tokenize, AI_TERMS
from loguru import logger

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import jieba

# 国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# CPU轻量embedding模型
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def jieba_tokenizer(text):
    """用于CountVectorizer的jieba分词器，过滤标点和无意义词"""
    import re
    tokens = tokenize(text, remove_stopwords=True, pos_filter=True)
    # 过滤标点、单字符、纯数字、URL残留
    filtered = []
    for t in tokens:
        if len(t) < 2:
            continue
        if re.fullmatch(r'[\d\s\.\,\!\?\;\:\-\/\\\(\)\[\]\{\}\"\'：，。！？、；""''（）《》【】…]+', t):
            continue
        if re.fullmatch(r'\d+', t):
            continue
        if re.search(r'https?|www\.|\.com|\.cn|youtube|bilibili', t, re.I):
            continue
        filtered.append(t)
    return filtered


def load_processed_data():
    """加载预处理数据"""
    proc_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    with open(proc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_embedding_model():
    """加载CPU embedding模型"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    logger.info(f"Embedding模型加载完成: {EMBEDDING_MODEL_NAME} (CPU)")
    return model


def train_bertopic(documents, timestamps=None, label="", min_topic_size=10):
    """训练BERTopic模型（CPU模式）"""
    init_jieba()

    if len(documents) < 20:
        logger.warning(f"[{label}] 文档数太少({len(documents)})，跳过BERTopic")
        return None, [], [], None

    # 过滤空文档
    valid_docs = []
    valid_timestamps = []
    for i, doc in enumerate(documents):
        if doc and len(doc.strip()) > 10:
            valid_docs.append(doc)
            if timestamps and i < len(timestamps):
                valid_timestamps.append(timestamps[i])
    logger.info(f"[{label}] 有效文档: {len(valid_docs)}/{len(documents)}")

    if len(valid_docs) < 20:
        logger.warning(f"[{label}] 有效文档不足20，跳过")
        return None, [], [], None

    # 加载embedding模型
    embedding_model = get_embedding_model()

    # jieba中文分词向量化
    vectorizer = CountVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)

    # BERTopic（CPU，min_topic_size替代min_cluster_size）
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        nr_topics="auto",
        verbose=True,
        language="multilingual",
    )

    logger.info(f"[{label}] 开始BERTopic训练...")
    topics, probs = topic_model.fit_transform(valid_docs)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info(f"[{label}] BERTopic发现 {n_topics} 个主题（含噪声主题-1）")

    # 动态主题追踪
    topics_over_time = None
    if valid_timestamps and len(valid_timestamps) == len(valid_docs):
        try:
            from datetime import datetime
            ts_dates = [datetime.fromtimestamp(t) for t in valid_timestamps]
            topics_over_time = topic_model.topics_over_time(
                valid_docs, ts_dates, nr_bins=BERTOPIC_NR_BINS
            )
            logger.info(f"[{label}] 动态主题追踪完成")
        except Exception as e:
            logger.warning(f"[{label}] topics_over_time失败: {e}")

    return topic_model, topics, probs, topics_over_time


def train_lda(documents, label="传播端", num_topics_range=range(5, 21)):
    """训练LDA模型，寻找最优主题数"""
    texts = [tokenize(doc, remove_stopwords=True, pos_filter=True) for doc in documents]
    texts = [t for t in texts if len(t) > 3]

    if len(texts) < 20:
        logger.warning(f"[{label}] LDA文档不足，跳过")
        return None, {"best_k": None, "best_coherence": None, "topics": [], "coherence_scores": []}

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    best_coherence = -1
    best_k = 5
    best_model = None
    coherence_scores = []

    for k in num_topics_range:
        lda = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=10, random_state=42)
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        score = cm.get_coherence()
        coherence_scores.append({"k": k, "coherence": round(score, 4)})
        logger.info(f"[{label}] LDA k={k}: coherence={score:.4f}")

        if score > best_coherence:
            best_coherence = score
            best_k = k
            best_model = lda

    logger.info(f"[{label}] 最优LDA: k={best_k}, coherence={best_coherence:.4f}")

    lda_result = {
        "label": label,
        "best_k": best_k,
        "best_coherence": round(best_coherence, 4),
        "coherence_scores": coherence_scores,
        "topics": [],
    }
    for idx in range(best_k):
        words = best_model.show_topic(idx, topn=15)
        lda_result["topics"].append({
            "topic_id": idx,
            "words": [{"word": w, "weight": round(float(p), 4)} for w, p in words]
        })

    # 保存
    save_name = f"lda_{label}.json"
    with open(os.path.join(RESULTS_DIR, save_name), 'w', encoding='utf-8') as f:
        json.dump(lda_result, f, ensure_ascii=False, indent=2)
    best_model.save(os.path.join(RESULTS_DIR, f"lda_model_{label}"))

    return best_model, lda_result


def extract_bertopic_summary(model, label=""):
    """提取BERTopic模型摘要"""
    if model is None:
        return []
    info = model.get_topic_info()
    topics_summary = []
    for _, row in info.iterrows():
        tid = row['Topic']
        if tid == -1:
            continue
        words = model.get_topic(tid)
        topics_summary.append({
            "topic_id": tid,
            "count": int(row['Count']),
            "name": row.get('Name', ''),
            "top_words": [w[0] for w in words[:10]] if words else [],
        })
    return topics_summary


def aggregate_topic_sentiment(topics, annotations, label=""):
    """按主题聚合情感分布（主题+情绪双维度）"""
    if not topics or not annotations:
        return {}

    ann_map = {}
    for a in annotations:
        key = 'propagator_annotation' if '传播' in label else 'receiver_annotation'
        ann_data = a.get(key, {})
        if ann_data:
            ann_map[a['bvid']] = ann_data

    # 按主题聚合
    topic_sentiment = {}
    for tid in set(topics):
        if tid == -1:
            continue
        topic_sentiment[tid] = {
            "positive": 0, "neutral": 0, "negative": 0,
            "support": 0, "oppose": 0, "stance_neutral": 0,
            "count": 0
        }

    return topic_sentiment


def run_topic_modeling():
    """运行完整主题建模流程：三套BERTopic + LDA对照"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_processed_data()
    logger.info(f"加载 {len(data)} 条预处理数据")

    # 构建三套文档
    prop_docs = [d['propagator_raw'] for d in data if d.get('propagator_raw')]
    recv_docs = [d['receiver_raw'] for d in data if d.get('receiver_raw')]
    asr_docs = [d['asr_raw'] for d in data if d.get('asr_raw') and len(d['asr_raw']) > 50]
    timestamps = [d['pubdate'] for d in data if d.get('pubdate')]

    logger.info(f"传播端: {len(prop_docs)}, 接收端: {len(recv_docs)}, ASR: {len(asr_docs)}")

    # 加载已有标注（用于主题+情感聚合）
    ann_file = os.path.join(RESULTS_DIR, "annotations.json")
    annotations = []
    if os.path.exists(ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

    results = {}

    # ========== BERTopic 传播端 ==========
    logger.info("=" * 60)
    logger.info("BERTopic 传播端（标题+描述）")
    logger.info("=" * 60)
    prop_model, prop_topics, prop_probs, prop_tot = train_bertopic(
        prop_docs, timestamps, label="传播端", min_topic_size=10
    )
    if prop_model:
        prop_model.save(os.path.join(RESULTS_DIR, "bertopic_propagator"))
        results['bertopic_propagator'] = extract_bertopic_summary(prop_model, "传播端")
        if prop_tot is not None:
            prop_tot.to_json(os.path.join(RESULTS_DIR, "bertopic_prop_over_time.json"),
                            orient='records', force_ascii=False)
        logger.info(f"传播端BERTopic: {len(results['bertopic_propagator'])} 个主题")

    # ========== BERTopic 接收端 ==========
    logger.info("=" * 60)
    logger.info("BERTopic 接收端（评论+弹幕）")
    logger.info("=" * 60)
    recv_model, recv_topics, recv_probs, recv_tot = train_bertopic(
        recv_docs, timestamps, label="接收端", min_topic_size=10
    )
    if recv_model:
        recv_model.save(os.path.join(RESULTS_DIR, "bertopic_receiver"))
        results['bertopic_receiver'] = extract_bertopic_summary(recv_model, "接收端")
        if recv_tot is not None:
            recv_tot.to_json(os.path.join(RESULTS_DIR, "bertopic_recv_over_time.json"),
                            orient='records', force_ascii=False)
        logger.info(f"接收端BERTopic: {len(results['bertopic_receiver'])} 个主题")

    # ========== BERTopic ASR转写 ==========
    logger.info("=" * 60)
    logger.info("BERTopic ASR转写文本")
    logger.info("=" * 60)
    if asr_docs:
        asr_ts = timestamps[:len(asr_docs)] if timestamps else None
        asr_model, asr_topics, asr_probs, asr_tot = train_bertopic(
            asr_docs, asr_ts, label="ASR", min_topic_size=8
        )
        if asr_model:
            asr_model.save(os.path.join(RESULTS_DIR, "bertopic_asr"))
            results['bertopic_asr'] = extract_bertopic_summary(asr_model, "ASR")
            logger.info(f"ASR BERTopic: {len(results['bertopic_asr'])} 个主题")
    else:
        logger.warning("无ASR文本，跳过ASR主题建模")
        results['bertopic_asr'] = []

    # ========== LDA 传播端 ==========
    logger.info("=" * 60)
    logger.info("LDA 传播端（Baseline对照）")
    logger.info("=" * 60)
    _, lda_prop = train_lda(prop_docs, label="传播端")
    results['lda_propagator'] = lda_prop

    # ========== LDA 接收端 ==========
    logger.info("=" * 60)
    logger.info("LDA 接收端（Baseline对照）")
    logger.info("=" * 60)
    _, lda_recv = train_lda(recv_docs, label="接收端")
    results['lda_receiver'] = lda_recv

    # ========== LDA ASR转写端 ==========
    lda_asr = {"best_k": None, "best_coherence": None, "topics": [], "coherence_scores": []}
    if asr_docs:
        logger.info("=" * 60)
        logger.info("LDA ASR转写端（Baseline对照）")
        logger.info("=" * 60)
        _, lda_asr = train_lda(asr_docs, label="ASR")
    results['lda_asr'] = lda_asr

    # ========== 汇总 ==========
    summary = {
        "total_videos": len(data),
        "propagator_docs": len(prop_docs),
        "receiver_docs": len(recv_docs),
        "asr_docs": len(asr_docs),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "bertopic_propagator_topics": len(results.get('bertopic_propagator', [])),
        "bertopic_receiver_topics": len(results.get('bertopic_receiver', [])),
        "bertopic_asr_topics": len(results.get('bertopic_asr', [])),
        "lda_propagator_best_k": lda_prop.get("best_k"),
        "lda_propagator_coherence": lda_prop.get("best_coherence"),
        "lda_receiver_best_k": lda_recv.get("best_k"),
        "lda_receiver_coherence": lda_recv.get("best_coherence"),
        "lda_asr_best_k": lda_asr.get("best_k"),
        "lda_asr_coherence": lda_asr.get("best_coherence"),
        "bertopic_propagator_detail": results.get('bertopic_propagator', []),
        "bertopic_receiver_detail": results.get('bertopic_receiver', []),
        "bertopic_asr_detail": results.get('bertopic_asr', []),
        "lda_propagator_topics": lda_prop.get("topics", []),
        "lda_receiver_topics": lda_recv.get("topics", []),
        "lda_asr_topics": lda_asr.get("topics", []),
        "lda_propagator_coherence_curve": lda_prop.get("coherence_scores", []),
        "lda_receiver_coherence_curve": lda_recv.get("coherence_scores", []),
        "lda_asr_coherence_curve": lda_asr.get("coherence_scores", []),
        "method_comparison": {
            "note": "BERTopic使用paraphrase-multilingual-MiniLM-L12-v2 embedding + HDBSCAN聚类，LDA使用gensim传统概率模型",
            "bertopic_advantage": "语义理解更强，适合短文本（弹幕/评论），能捕捉同义词",
            "lda_advantage": "可解释性强，主题词权重直观，计算成本低",
        }
    }

    with open(os.path.join(RESULTS_DIR, "topic_modeling_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("主题建模汇总已保存")

    # 同时保存LDA结果到独立文件
    lda_combined = {
        "propagator": lda_prop,
        "receiver": lda_recv,
        "asr": lda_asr,
    }
    with open(os.path.join(RESULTS_DIR, "lda_results.json"), 'w', encoding='utf-8') as f:
        json.dump(lda_combined, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    run_topic_modeling()
