"""
音频下载模块 - 使用yt-dlp下载B站视频音频
"""
import os
import sys
import json
import subprocess
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from loguru import logger


def download_audio(bvid, output_dir=AUDIO_DIR):
    """下载单个视频的音频（m4a格式）"""
    url = f"https://www.bilibili.com/video/{bvid}"
    output_template = os.path.join(output_dir, f"{bvid}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",  # 中等质量，节省空间
        "-o", output_template,
        "--no-playlist",
        "--socket-timeout", "30",
        "--retries", "3",
        "--quiet",
        "--no-warnings",
        url
    ]

    # 添加Cookie认证
    cookie_args = [
        "--add-header",
        f"Cookie: SESSDATA={BILIBILI_SESSDATA}"
    ]

    try:
        result = subprocess.run(
            cmd + cookie_args,
            capture_output=True, text=True, timeout=300
        )
        # 检查输出文件
        for ext in ["m4a", "mp3", "webm", "ogg"]:
            filepath = os.path.join(output_dir, f"{bvid}.{ext}")
            if os.path.exists(filepath):
                return filepath
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"下载超时: {bvid}")
        return None
    except Exception as e:
        logger.warning(f"下载失败 {bvid}: {e}")
        return None


def batch_download(bvid_list, delay_range=(2.0, 5.0)):
    """批量下载音频"""
    results = {"success": [], "failed": []}

    for i, bvid in enumerate(bvid_list):
        logger.info(f"[{i+1}/{len(bvid_list)}] 下载 {bvid}")
        filepath = download_audio(bvid)
        if filepath:
            results["success"].append({"bvid": bvid, "path": filepath})
            logger.info(f"  成功: {filepath}")
        else:
            results["failed"].append(bvid)
            logger.warning(f"  失败: {bvid}")

        if i < len(bvid_list) - 1:
            time.sleep(random.uniform(*delay_range))

    # 保存下载记录
    record_file = os.path.join(AUDIO_DIR, "download_record.json")
    with open(record_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"下载完成: 成功{len(results['success'])}, 失败{len(results['failed'])}")
    return results


if __name__ == "__main__":
    # 从采集结果中读取视频列表
    raw_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    if os.path.exists(raw_file):
        with open(raw_file, 'r', encoding='utf-8') as f:
            videos = json.load(f)
        bvids = [v['bvid'] for v in videos]
        logger.info(f"准备下载 {len(bvids)} 个视频的音频")
        batch_download(bvids)
    else:
        logger.error("未找到采集数据文件")
