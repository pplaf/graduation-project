"""
主控Pipeline - 一键运行全部分析流程
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import *
from loguru import logger

# 配置日志
log_file = os.path.join(LOGS_DIR, "pipeline_{time}.log")
logger.add(log_file, rotation="10 MB", encoding="utf-8")


def step(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"\n{'='*60}\n步骤: {name}\n{'='*60}")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"步骤 [{name}] 完成，耗时 {elapsed:.1f}秒")
            return result
        return wrapper
    return decorator


@step("音频下载")
def run_download():
    from src.asr.audio_downloader import batch_download
    raw_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    with open(raw_file, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    bvids = [v['bvid'] for v in videos]
    return batch_download(bvids)


@step("ASR转写")
def run_asr():
    from src.asr.transcriber import batch_transcribe
    return batch_transcribe()


@step("文本预处理")
def run_preprocess():
    from src.preprocess.text_processor import preprocess_all
    return preprocess_all()


@step("主题建模")
def run_topic_modeling():
    from src.analysis.topic_modeling import run_topic_modeling
    return run_topic_modeling()


@step("LLM标注")
def run_annotation():
    from src.analysis.llm_annotator import run_annotation
    return run_annotation()


@step("知识密度分析")
def run_knowledge_density():
    from src.analysis.knowledge_density import run_knowledge_analysis
    return run_knowledge_analysis()


@step("语义网络分析")
def run_semantic_network():
    from src.analysis.semantic_network import run_semantic_network
    return run_semantic_network()


@step("可视化")
def run_visualization():
    from src.visualization.visualizer import run_all_visualizations
    return run_all_visualizations()


@step("对比分析")
def run_cross_analysis():
    from src.analysis.cross_analysis import run_cross_analysis
    return run_cross_analysis()


@step("整合导出")
def run_integration():
    """整合所有结果"""
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.json'):
            fpath = os.path.join(RESULTS_DIR, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                results[fname] = json.load(f)

    # 生成数据摘要表
    raw_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    if os.path.exists(raw_file):
        with open(raw_file, 'r', encoding='utf-8') as f:
            videos = json.load(f)
        results['data_summary'] = {
            "total_videos": len(videos),
            "news_count": sum(1 for v in videos if v.get('category') == 'news'),
            "interview_count": sum(1 for v in videos if v.get('category') == 'interview'),
            "avg_quality_score": round(sum(v.get('quality_score', 0) for v in videos) / max(len(videos), 1), 1),
            "total_views": sum(v.get('view', 0) for v in videos),
            "total_comments": sum(len(v.get('comments', [])) for v in videos),
            "total_danmakus": sum(len(v.get('danmakus', [])) for v in videos),
        }

    # 保存整合结果
    output = os.path.join(RESULTS_DIR, "final_report.json")
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"最终报告已保存: {output}")
    return results


def main():
    """运行完整Pipeline"""
    logger.info("开始运行完整分析Pipeline")

    pipeline_steps = [
        ("download", run_download),
        ("asr", run_asr),
        ("preprocess", run_preprocess),
        ("topic_modeling", run_topic_modeling),
        ("annotation", run_annotation),
        ("knowledge_density", run_knowledge_density),
        ("semantic_network", run_semantic_network),
        ("visualization", run_visualization),
        ("cross_analysis", run_cross_analysis),
        ("integration", run_integration),
    ]

    for name, func in pipeline_steps:
        try:
            func()
        except Exception as e:
            logger.error(f"步骤 [{name}] 失败: {e}")
            logger.warning(f"跳过步骤 [{name}]，继续执行...")
            continue

    logger.info("Pipeline执行完成")


if __name__ == "__main__":
    main()
