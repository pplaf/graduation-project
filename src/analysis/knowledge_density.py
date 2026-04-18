"""
知识密度与内容分析模块
"""
import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.preprocess.text_processor import AI_TERMS, init_jieba
from loguru import logger

import jieba.posseg as pseg


# 实词词性集合
CONTENT_POS = {'n', 'nr', 'ns', 'nt', 'nz', 'ng', 'nrt', 'nrfg',
               'v', 'vd', 'vn', 'vg', 'a', 'ad', 'an', 'ag', 'eng'}


def compute_knowledge_density(tokens):
    """知识密度 = AI术语命中数 / 总词数"""
    if not tokens:
        return 0.0
    ai_term_set = set(AI_TERMS)
    hits = sum(1 for t in tokens if t in ai_term_set)
    return hits / len(tokens)


def compute_lexical_density(text):
    """词汇密度 = 实词数 / 总词数"""
    if not text or len(text.strip()) < 2:
        return 0.0
    words = pseg.lcut(text)
    total = len(words)
    if total == 0:
        return 0.0
    content_count = sum(1 for w in words if w.flag in CONTENT_POS)
    return content_count / total


def classify_video_type(knowledge_density, duration_min):
    """根据知识密度和时长分类视频类型"""
    if knowledge_density >= 0.08:
        return "hardcore_tech"     # 硬核技术
    elif knowledge_density >= 0.04:
        return "tech科普"          # 技术科普
    elif duration_min >= 30:
        return "深度讨论"           # 深度讨论（长视频）
    else:
        return "泛资讯"             # 泛资讯/娱乐


def run_knowledge_analysis():
    """运行完整知识密度分析"""
    init_jieba()

    proc_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    with open(proc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"加载 {len(data)} 条数据")

    results = []
    type_distribution = Counter()

    for d in data:
        prop_tokens = d.get('propagator_tokens', [])
        recv_tokens = d.get('receiver_tokens', [])
        asr_tokens = d.get('asr_tokens', [])
        prop_raw = d.get('propagator_raw', '')
        recv_raw = d.get('receiver_raw', '')
        asr_raw = d.get('asr_raw', '')
        duration_sec = d.get('duration', 0)
        if duration_sec == 0:
            duration_sec = 300
        duration_min = duration_sec / 60

        # 三端知识密度
        prop_kd = compute_knowledge_density(prop_tokens)
        prop_ld = compute_lexical_density(prop_raw)
        recv_kd = compute_knowledge_density(recv_tokens)
        recv_ld = compute_lexical_density(recv_raw)
        asr_kd = compute_knowledge_density(asr_tokens) if asr_tokens else 0.0
        asr_ld = compute_lexical_density(asr_raw) if asr_raw else 0.0

        # AI术语命中统计（三端）
        ai_term_set = set(AI_TERMS)
        prop_ai_hits = Counter(t for t in prop_tokens if t in ai_term_set)
        recv_ai_hits = Counter(t for t in recv_tokens if t in ai_term_set)
        asr_ai_hits = Counter(t for t in asr_tokens if t in ai_term_set)

        # 分类
        video_type = classify_video_type(prop_kd, duration_min)
        type_distribution[video_type] += 1

        results.append({
            "bvid": d['bvid'],
            "title": d.get('title', ''),
            "category": d.get('category', ''),
            "video_type": video_type,
            "duration_min": round(duration_min, 1),
            "has_asr": d.get('has_asr', False),
            # 传播端
            "propagator_knowledge_density": round(prop_kd, 4),
            "propagator_lexical_density": round(prop_ld, 4),
            "propagator_ai_terms": prop_ai_hits.most_common(10),
            "propagator_token_count": len(prop_tokens),
            # 接收端
            "receiver_knowledge_density": round(recv_kd, 4),
            "receiver_lexical_density": round(recv_ld, 4),
            "receiver_ai_terms": recv_ai_hits.most_common(10),
            "receiver_token_count": len(recv_tokens),
            # ASR音频转写端
            "asr_knowledge_density": round(asr_kd, 4),
            "asr_lexical_density": round(asr_ld, 4),
            "asr_ai_terms": asr_ai_hits.most_common(10),
            "asr_token_count": len(asr_tokens),
        })

    # 统计（三端）
    prop_kd_values = [r['propagator_knowledge_density'] for r in results]
    recv_kd_values = [r['receiver_knowledge_density'] for r in results]
    asr_kd_values = [r['asr_knowledge_density'] for r in results if r['has_asr']]

    def kd_stats(values, label):
        if not values:
            return {f"{label}_kd_mean": 0, f"{label}_kd_max": 0, f"{label}_kd_min": 0}
        return {
            f"{label}_kd_mean": round(sum(values) / len(values), 4),
            f"{label}_kd_max": round(max(values), 4),
            f"{label}_kd_min": round(min(values), 4),
        }

    summary = {
        "total_videos": len(results),
        "has_asr_count": sum(1 for r in results if r['has_asr']),
        "type_distribution": dict(type_distribution),
        **kd_stats(prop_kd_values, "propagator"),
        **kd_stats(recv_kd_values, "receiver"),
        **kd_stats(asr_kd_values, "asr"),
    }

    logger.info(f"类型分布: {dict(type_distribution)}")
    logger.info(f"传播端知识密度: 均值={summary['propagator_kd_mean']}, 最高={summary['propagator_kd_max']}")
    logger.info(f"接收端知识密度: 均值={summary['receiver_kd_mean']}, 最高={summary['receiver_kd_max']}")
    logger.info(f"ASR知识密度: 均值={summary.get('asr_kd_mean',0)}, 最高={summary.get('asr_kd_max',0)} ({summary['has_asr_count']}条)")

    # 保存
    with open(os.path.join(RESULTS_DIR, "knowledge_density.json"), 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    logger.info("知识密度分析结果已保存")

    return summary, results


if __name__ == "__main__":
    run_knowledge_analysis()
