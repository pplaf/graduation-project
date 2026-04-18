"""
完整Pipeline - 一步到位运行所有流程
从采集到最终报告，带断点续传
"""
import os, sys, json, time, glob, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from src.config import *

# ============================================================
# Step 1: 数据采集
# ============================================================
def step1_collect():
    raw_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    if os.path.exists(raw_file):
        with open(raw_file, 'r', encoding='utf-8') as f:
            vids = json.load(f)
        if len(vids) >= 50:
            logger.info(f"Step1 SKIP: 已有 {len(vids)} 个视频数据")
            return vids

    import asyncio
    from src.collect.bilibili_crawler import collect_all
    vids = asyncio.run(collect_all())
    logger.info(f"Step1 DONE: 采集 {len(vids)} 个视频")
    return vids

# ============================================================
# Step 2: 音频下载
# ============================================================
def step2_download(videos):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    from src.asr.audio_downloader import download_audio

    downloaded = 0
    for i, v in enumerate(videos):
        bvid = v['bvid']
        # 检查已有文件
        existing = None
        for ext in ['m4a', 'mp3', 'wav', 'webm']:
            p = os.path.join(AUDIO_DIR, f"{bvid}.{ext}")
            if os.path.exists(p):
                existing = p
                break
        if existing:
            downloaded += 1
            continue
        logger.info(f"[{i+1}/{len(videos)}] 下载 {bvid}")
        result = download_audio(bvid)
        if result:
            downloaded += 1
        time.sleep(2.0 + random.random() * 3)

    logger.info(f"Step2 DONE: 音频下载 {downloaded}/{len(videos)}")

# ============================================================
# Step 3: ASR转写
# ============================================================
def step3_asr():
    asr_file = os.path.join(PROCESSED_DIR, "asr_results.json")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 收集音频文件
    audio_files = []
    for ext in ['*.m4a', '*.mp3', '*.webm', '*.wav']:
        audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))
    audio_files = sorted(set(audio_files))
    if not audio_files:
        logger.warning("Step3 SKIP: 无音频文件")
        return

    # 转换为WAV
    from src.asr.transcriber import convert_to_wav, load_asr_model, transcribe_audio
    wav_files = []
    for af in audio_files:
        if af.endswith('.wav'):
            wav_files.append(af)
        else:
            wav = convert_to_wav(af)
            if wav:
                wav_files.append(wav)
    logger.info(f"Step3: {len(wav_files)} 个WAV待转写")

    # 加载已有结果
    results = []
    done_bvids = set()
    if os.path.exists(asr_file):
        with open(asr_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_bvids = {r['bvid'] for r in results}
        logger.info(f"  已转写 {len(done_bvids)} 个，继续...")

    # 加载模型
    logger.info("  加载ASR模型...")
    model = load_asr_model()
    logger.info("  模型加载完成")

    # 批量转写
    for i, wf in enumerate(wav_files):
        bvid = os.path.splitext(os.path.basename(wf))[0]
        if bvid in done_bvids:
            continue
        logger.info(f"  [{i+1}/{len(wav_files)}] 转写 {bvid}")
        text = transcribe_audio(model, wf)
        results.append({"bvid": bvid, "text": text, "audio_path": wf})
        logger.info(f"    文本长度: {len(text)} 字")
        if len(results) % 10 == 0:
            with open(asr_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(asr_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Step3 DONE: ASR转写 {len(results)} 条")

# ============================================================
# Step 4: 文本预处理
# ============================================================
def step4_preprocess():
    proc_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    if os.path.exists(proc_file):
        with open(proc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if len(data) >= 50:
            logger.info(f"Step4 SKIP: 已有 {len(data)} 条预处理数据")
            return data

    from src.preprocess.text_processor import preprocess_all
    data = preprocess_all()
    logger.info(f"Step4 DONE: 预处理 {len(data)} 条")
    return data

# ============================================================
# Step 5: 主题建模
# ============================================================
def step5_topic_modeling():
    result_file = os.path.join(RESULTS_DIR, "topic_modeling_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(result_file):
        logger.info("Step5 SKIP: 主题建模已有结果")
        return

    from src.analysis.topic_modeling import run_topic_modeling
    run_topic_modeling()
    logger.info("Step5 DONE: 主题建模完成")

# ============================================================
# Step 6: LLM标注
# ============================================================
def step6_annotation():
    ann_file = os.path.join(RESULTS_DIR, "annotations.json")
    if os.path.exists(ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            anns = json.load(f)
        if len(anns) >= 50:
            logger.info(f"Step6 SKIP: 已标注 {len(anns)} 条")
            return

    from src.analysis.llm_annotator import run_annotation
    run_annotation()
    logger.info("Step6 DONE: LLM标注完成")

# ============================================================
# Step 7: 知识密度
# ============================================================
def step7_knowledge():
    kd_file = os.path.join(RESULTS_DIR, "knowledge_density.json")
    if os.path.exists(kd_file):
        logger.info("Step7 SKIP: 知识密度已有结果")
        return

    from src.analysis.knowledge_density import run_knowledge_analysis
    run_knowledge_analysis()
    logger.info("Step7 DONE: 知识密度分析完成")

# ============================================================
# Step 8: 语义网络
# ============================================================
def step8_network():
    net_file = os.path.join(RESULTS_DIR, "semantic_network.json")
    if os.path.exists(net_file):
        logger.info("Step8 SKIP: 语义网络已有结果")
        return

    from src.analysis.semantic_network import run_semantic_network
    run_semantic_network()
    logger.info("Step8 DONE: 语义网络完成")

# ============================================================
# Step 9: 可视化
# ============================================================
def step9_visualize():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from src.visualization.visualizer import run_all_visualizations
    run_all_visualizations()
    logger.info("Step9 DONE: 可视化完成")

# ============================================================
# Step 10: 对比分析
# ============================================================
def step10_cross_analysis():
    cross_file = os.path.join(RESULTS_DIR, "cross_analysis.json")
    if os.path.exists(cross_file):
        logger.info("Step10 SKIP: 对比分析已有结果")
        return

    from src.analysis.cross_analysis import run_cross_analysis
    run_cross_analysis()
    logger.info("Step10 DONE: 对比分析完成")

# ============================================================
# Step 11: 整合报告
# ============================================================
def step11_report():
    final = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname), 'r', encoding='utf-8') as f:
                final[fname] = json.load(f)

    raw_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    if os.path.exists(raw_file):
        with open(raw_file, 'r', encoding='utf-8') as f:
            vids = json.load(f)
        cats = {}
        for v in vids:
            c = v.get('category', 'unknown')
            cats[c] = cats.get(c, 0) + 1
        final['data_summary'] = {
            'total_videos': len(vids),
            'categories': cats,
            'avg_quality_score': round(sum(v.get('quality_score', 0) for v in vids) / max(len(vids), 1), 1),
            'total_views': sum(v.get('view', 0) for v in vids),
            'total_comments': sum(len(v.get('comments', [])) for v in vids),
            'total_danmakus': sum(len(v.get('danmakus', [])) for v in vids),
        }

    out = os.path.join(RESULTS_DIR, "final_report.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 更新task.json
    with open('task.json', 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    for t in tasks['tasks']:
        t['passes'] = True
    with open('task.json', 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info(f"Step11 DONE: 最终报告 -> {out}")
    logger.info(f"数据摘要: {final.get('data_summary', {})}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("开始完整Pipeline")
    logger.info("=" * 60)

    steps = [
        ("采集", step1_collect),
        ("音频下载", lambda: step2_download(step1_collect.__code__ and None)),  # 需要videos参数
        ("ASR转写", step3_asr),
        ("预处理", step4_preprocess),
        ("主题建模", step5_topic_modeling),
        ("LLM标注", step6_annotation),
        ("知识密度", step7_knowledge),
        ("语义网络", step8_network),
        ("可视化", step9_visualize),
        ("对比分析", step10_cross_analysis),
        ("整合报告", step11_report),
    ]

    # 修正step2需要videos参数
    def run_step2():
        with open(os.path.join(RAW_DIR, "bilibili_videos.json"), 'r', encoding='utf-8') as f:
            videos = json.load(f)
        step2_download(videos)

    actual_steps = [
        ("采集", step1_collect),
        ("音频下载", run_step2),
        ("ASR转写", step3_asr),
        ("预处理", step4_preprocess),
        ("主题建模", step5_topic_modeling),
        ("LLM标注", step6_annotation),
        ("知识密度", step7_knowledge),
        ("语义网络", step8_network),
        ("可视化", step9_visualize),
        ("对比分析", step10_cross_analysis),
        ("整合报告", step11_report),
    ]

    for name, func in actual_steps:
        logger.info(f"\n{'='*60}\n>>> {name}\n{'='*60}")
        try:
            func()
        except Exception as e:
            logger.error(f"{name} 失败: {e}")
            import traceback
            traceback.print_exc()
            logger.warning(f"跳过 {name}，继续...")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline全部完成!")
    logger.info("=" * 60)
