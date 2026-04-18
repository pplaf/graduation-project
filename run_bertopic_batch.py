"""
分批运行BERTopic - 每次只跑一套，避免内存溢出
用法: python run_bertopic_batch.py [propagator|receiver|asr|all]
"""
import sys, os, json, gc
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
logger.add("logs/bertopic_batch.log", level="INFO")

from src.config import *
from src.analysis.topic_modeling import (
    train_bertopic, train_lda, extract_bertopic_summary,
    load_processed_data, EMBEDDING_MODEL_NAME
)


def run_single_bertopic(corpus_name):
    """运行单个语料库的BERTopic"""
    data = load_processed_data()

    if corpus_name == "propagator":
        docs = [d['propagator_raw'] for d in data if d.get('propagator_raw')]
        timestamps = [d['pubdate'] for d in data if d.get('propagator_raw')]
        min_size = 10
    elif corpus_name == "receiver":
        docs = [d['receiver_raw'] for d in data if d.get('receiver_raw')]
        timestamps = [d['pubdate'] for d in data if d.get('receiver_raw')]
        min_size = 10
    elif corpus_name == "asr":
        docs = [d['asr_raw'] for d in data if d.get('asr_raw') and len(d['asr_raw']) > 50]
        timestamps = [d['pubdate'] for d in data if d.get('asr_raw') and len(d.get('asr_raw', '')) > 50]
        min_size = 8
    else:
        logger.error(f"未知语料库: {corpus_name}")
        return

    logger.info(f"=== BERTopic {corpus_name}: {len(docs)} 文档 ===")

    model, topics, probs, tot = train_bertopic(docs, timestamps, label=corpus_name, min_topic_size=min_size)

    if model:
        # 保存模型
        model_path = os.path.join(RESULTS_DIR, f"bertopic_{corpus_name}")
        model.save(model_path)
        logger.info(f"模型已保存: {model_path}")

        # 保存摘要
        summary = extract_bertopic_summary(model, corpus_name)
        summary_path = os.path.join(RESULTS_DIR, f"bertopic_{corpus_name}_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"摘要已保存: {len(summary)} 个主题")

        # 保存topics_over_time
        if tot is not None:
            tot_path = os.path.join(RESULTS_DIR, f"bertopic_{corpus_name}_over_time.json")
            tot.to_json(tot_path, orient='records', force_ascii=False)
            logger.info(f"动态主题已保存")

        # 打印主题
        for t in summary[:8]:
            logger.info(f"  Topic {t['topic_id']}: {t['top_words'][:5]} (count={t['count']})")
    else:
        logger.warning(f"BERTopic {corpus_name} 训练失败")

    # 强制释放内存
    del model, topics, probs, tot
    gc.collect()


def merge_results():
    """合并三套BERTopic结果到topic_modeling_summary.json"""
    data = load_processed_data()
    prop_docs = [d['propagator_raw'] for d in data if d.get('propagator_raw')]
    recv_docs = [d['receiver_raw'] for d in data if d.get('receiver_raw')]
    asr_docs = [d['asr_raw'] for d in data if d.get('asr_raw') and len(d['asr_raw']) > 50]

    # 读取各套BERTopic摘要
    def load_summary(name):
        fp = os.path.join(RESULTS_DIR, f"bertopic_{name}_summary.json")
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    prop_summary = load_summary("propagator")
    recv_summary = load_summary("receiver")
    asr_summary = load_summary("asr")

    # 读取LDA结果
    lda_path = os.path.join(RESULTS_DIR, "lda_results.json")
    lda_data = {}
    if os.path.exists(lda_path):
        with open(lda_path, 'r', encoding='utf-8') as f:
            lda_data = json.load(f)

    lda_prop = lda_data.get('propagator', {})
    lda_recv = lda_data.get('receiver', {})
    lda_asr = lda_data.get('asr', {})

    summary = {
        "total_videos": len(data),
        "propagator_docs": len(prop_docs),
        "receiver_docs": len(recv_docs),
        "asr_docs": len(asr_docs),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "bertopic_propagator_topics": len(prop_summary),
        "bertopic_receiver_topics": len(recv_summary),
        "bertopic_asr_topics": len(asr_summary),
        "lda_propagator_best_k": lda_prop.get("best_k"),
        "lda_propagator_coherence": lda_prop.get("best_coherence"),
        "lda_receiver_best_k": lda_recv.get("best_k"),
        "lda_receiver_coherence": lda_recv.get("best_coherence"),
        "lda_asr_best_k": lda_asr.get("best_k"),
        "lda_asr_coherence": lda_asr.get("best_coherence"),
        "bertopic_propagator_detail": prop_summary,
        "bertopic_receiver_detail": recv_summary,
        "bertopic_asr_detail": asr_summary,
        "lda_propagator_topics": lda_prop.get("topics", []),
        "lda_receiver_topics": lda_recv.get("topics", []),
        "lda_asr_topics": lda_asr.get("topics", []),
        "lda_propagator_coherence_curve": lda_prop.get("coherence_scores", []),
        "lda_receiver_coherence_curve": lda_recv.get("coherence_scores", []),
        "lda_asr_coherence_curve": lda_asr.get("coherence_scores", []),
        "method_comparison": {
            "note": f"BERTopic使用{EMBEDDING_MODEL_NAME} + HDBSCAN，LDA使用gensim",
            "bertopic_advantage": "语义理解强，适合短文本弹幕评论",
            "lda_advantage": "可解释性强，主题词权重直观",
        }
    }

    with open(os.path.join(RESULTS_DIR, "topic_modeling_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"合并完成: BERTopic 传播端{len(prop_summary)}+接收端{len(recv_summary)}+ASR{len(asr_summary)} 主题")
    return summary


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        # 依次跑三套，每套独立进程内存隔离
        for corpus in ["propagator", "receiver", "asr"]:
            logger.info(f"\n{'='*60}\n开始 {corpus}\n{'='*60}")
            run_single_bertopic(corpus)
            logger.info(f"{corpus} 完成，释放内存\n")
        merge_results()
        logger.info("全部BERTopic完成!")
    elif target == "merge":
        merge_results()
    else:
        run_single_bertopic(target)
