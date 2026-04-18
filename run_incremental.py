"""
增量更新Pipeline - ASR完成后补充运行
只对新增ASR文本做增量标注，其余步骤全部重走
"""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loguru import logger
logger.add("logs/incremental_pipeline.log", level="INFO")

from src.config import *

# ============================================================
# Step 1: 等待ASR完成
# ============================================================
def wait_for_asr(timeout_min=180):
    asr_file = os.path.join(PROCESSED_DIR, "asr_results.json")
    batch_log = "logs/asr_batch.log"
    logger.info("等待ASR批量转写完成...")

    for i in range(timeout_min * 2):  # 每30秒检查一次
        # 检查进程是否结束（日志最后一行含"完成"）
        if os.path.exists(batch_log):
            with open(batch_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            last_lines = ''.join(lines[-3:])
            if "ASR批量转写完成" in last_lines or "无新文件" in last_lines:
                logger.info("ASR转写已完成")
                break

        if os.path.exists(asr_file):
            d = json.load(open(asr_file, 'r', encoding='utf-8'))
            logger.info(f"  ASR进度: {len(d)} 条已转写 (等待{i*30}秒)")

        time.sleep(30)
    else:
        logger.warning("等待超时，使用当前已有ASR结果继续")

    if os.path.exists(asr_file):
        d = json.load(open(asr_file, 'r', encoding='utf-8'))
        has_text = sum(1 for r in d if r.get('text', '').strip())
        logger.info(f"ASR结果: {len(d)} 条, 有效文本: {has_text} 条")
        return d
    return []


# ============================================================
# Step 2: 增量预处理（将ASR文本合并进processed_videos.json）
# ============================================================
def step_incremental_preprocess(asr_results):
    from src.preprocess.text_processor import preprocess_all
    logger.info("=== 增量预处理：合并ASR文本 ===")

    # preprocess_all 内部会自动读取 asr_results.json 并合并
    data = preprocess_all()
    has_asr = sum(1 for d in data if d.get('has_asr'))
    logger.info(f"预处理完成: {len(data)} 条, 含ASR文本: {has_asr} 条")
    return data


# ============================================================
# Step 3: 增量LLM标注（10线程并发，5个API Key轮询）
# ============================================================
def step_incremental_annotation(processed_data):
    from src.analysis.llm_annotator import run_annotation
    logger.info("=== 增量LLM标注（10线程并发）===")
    # run_annotation 内部自动检测已有标注，只标注缺失的
    annotations = run_annotation()
    logger.info(f"标注完成: {len(annotations)} 条")
    return annotations


# ============================================================
# Step 4-8: 重走知识密度、语义网络、可视化、对比分析、报告
# ============================================================
def step_rerun_analysis():
    import numpy as np
    from collections import Counter

    # 知识密度
    logger.info("=== 知识密度分析 ===")
    from src.analysis.knowledge_density import run_knowledge_analysis
    run_knowledge_analysis()

    # 语义网络
    logger.info("=== 语义网络分析 ===")
    from src.analysis.semantic_network import run_semantic_network
    run_semantic_network()

    # 可视化
    logger.info("=== 可视化 ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from src.visualization.visualizer import run_all_visualizations
    run_all_visualizations()

    # 对比分析
    logger.info("=== 对比分析 ===")
    from src.analysis.cross_analysis import run_cross_analysis
    run_cross_analysis()

    # 整合报告
    logger.info("=== 整合报告 ===")
    final = {}
    for fname in os.listdir(RESULTS_DIR):
        fp = os.path.join(RESULTS_DIR, fname)
        if fname.endswith('.json') and os.path.isfile(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                final[fname] = json.load(f)

    with open(os.path.join(RAW_DIR, "bilibili_videos.json"), 'r', encoding='utf-8') as f:
        vids = json.load(f)
    cats = dict(Counter(v.get('category', '?') for v in vids))
    final['data_summary'] = {
        'total_videos': len(vids),
        'categories': cats,
        'avg_quality_score': round(np.mean([v.get('quality_score', 0) for v in vids]), 1),
        'total_views': sum(v.get('view', 0) for v in vids),
        'total_comments': sum(len(v.get('comments', [])) for v in vids),
        'total_danmakus': sum(len(v.get('danmakus', [])) for v in vids),
    }
    with open(os.path.join(RESULTS_DIR, "final_report.json"), 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    logger.info(f"报告更新完成: {final['data_summary']}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("增量更新Pipeline启动")
    logger.info("=" * 60)

    # 1. 等待ASR
    asr_results = wait_for_asr(timeout_min=180)

    # 2. 增量预处理
    processed_data = step_incremental_preprocess(asr_results)

    # 3. 增量标注
    step_incremental_annotation(processed_data)

    # 4-8. 重走分析
    step_rerun_analysis()

    logger.info("增量更新Pipeline全部完成!")
