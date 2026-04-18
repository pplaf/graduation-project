"""
ASR转写模块 - 使用FunASR Paraformer批量转写音频
"""
import os
import sys
import json
import glob
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from loguru import logger


def convert_to_wav(input_path, output_path=None, target_sr=16000):
    """将音频转换为16kHz单声道WAV
    优先用imageio-ffmpeg内置二进制（解决m4a格式不识别），fallback到torchaudio
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".wav"

    if os.path.exists(output_path):
        return output_path

    # 方法1：imageio-ffmpeg（内置ffmpeg二进制，无需系统安装）
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-ar", str(target_sr), "-ac", "1",
            "-f", "wav", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        logger.warning(f"ffmpeg转换失败: {result.stderr.decode('utf-8','ignore')[:200]}")
    except Exception as e:
        logger.warning(f"imageio-ffmpeg不可用: {e}")

    # 方法2：torchaudio fallback
    try:
        waveform, sr = torchaudio.load(input_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        torchaudio.save(output_path, waveform, target_sr)
        return output_path
    except Exception as e:
        logger.warning(f"音频转换失败 {input_path}: {e}")
        return None


def load_asr_model():
    """加载FunASR模型（Paraformer + VAD + Punc）"""
    from funasr import AutoModel
    model = AutoModel(
        model=ASR_MODEL,
        vad_model=VAD_MODEL,
        punc_model=PUNC_MODEL,
        device=ASR_DEVICE,
    )
    return model


def transcribe_audio(model, audio_path):
    """转写单个音频文件"""
    try:
        result = model.generate(
            input=audio_path,
            batch_size_s=ASR_BATCH_SIZE_S,
        )
        if result and len(result) > 0:
            text = result[0].get("text", "")
            return text
        return ""
    except Exception as e:
        logger.warning(f"转写失败 {audio_path}: {e}")
        return ""


def batch_transcribe():
    """批量转写所有音频文件"""
    logger.info("加载ASR模型...")
    model = load_asr_model()
    logger.info("ASR模型加载完成")

    # 查找所有音频文件
    audio_files = []
    for ext in ["*.wav", "*.m4a", "*.mp3"]:
        audio_files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))
    audio_files = sorted(set(audio_files))
    logger.info(f"找到 {len(audio_files)} 个音频文件")

    # 加载已有转写记录（断点续传）
    output_file = os.path.join(PROCESSED_DIR, "asr_results.json")
    done_bvids = set()
    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_bvids = {r['bvid'] for r in results}
        logger.info(f"断点续传: 已转写 {len(done_bvids)} 个")

    for i, audio_path in enumerate(audio_files):
        bvid = os.path.splitext(os.path.basename(audio_path))[0]
        if bvid in done_bvids:
            continue

        # 如果不是wav，先转换
        if not audio_path.endswith('.wav'):
            wav_path = convert_to_wav(audio_path)
            if not wav_path:
                continue
            audio_path = wav_path

        logger.info(f"[{i+1}/{len(audio_files)}] 转写 {bvid}")
        text = transcribe_audio(model, audio_path)

        results.append({
            "bvid": bvid,
            "text": text,
            "audio_path": audio_path,
        })
        logger.info(f"  文本长度: {len(text)} 字")

        # 每20个保存一次
        if len(results) % 20 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"断点保存: {len(results)} 条")

    # 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"转写完成: 共 {len(results)} 条, 保存到 {output_file}")
    return results


if __name__ == "__main__":
    batch_transcribe()
