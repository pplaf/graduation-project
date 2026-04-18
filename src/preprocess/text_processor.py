"""
文本预处理模块 - 清洗、分词、去停用词、AI术语词典
"""
import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

import jieba
import jieba.posseg as pseg
from loguru import logger


# ============ 停用词表 ============
STOPWORDS = set()

# 基本停用词
_BASIC_STOPWORDS = """
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
他 她 它 们 吧 啊 呢 哦 嗯 呀 哈 哎 那 这个 那个 什么 怎么 为什么 谁 哪 多少 几 可以 能
应该 如果 因为 所以 但是 而且 或者 虽然 不过 然后已 已经 还 再 又 将 被 把 从 向 对 与 给
为 以 之 其 此 某每 各 该 本 另 其中 这些 那些 这么 那么 这样 那样 怎样 多么 如何
可能 大概 也许 确实 的确 真的 简直 几乎 比较 相当 非常 特别 十分 极其 最 更 越 太 过
做 进行 得 及 等 等等 吧 嘛 啦 喽 哟 哈 咧 呗 嘞
"""
STOPWORDS.update(_BASIC_STOPWORDS.split())

# B站常见无意义词
_BILIBILI_STOPWORDS = [
    "UP主", "up主", "UP", "up", "视频", "播放", "投币", "收藏", "转发",
    "关注", "点赞", "三连", "弹幕", "评论", "谢谢", "感谢", "支持",
    "沙发", "前排", "打卡", "签到", "路过", "看看", "顶",
    "哈哈哈哈", "哈哈哈", "2333", "666", "牛", "牛逼", "卧槽",
]
STOPWORDS.update(_BILIBILI_STOPWORDS)


# ============ AI术语词典 ============
AI_TERMS = [
    # 核心概念
    "人工智能", "机器学习", "深度学习", "神经网络", "大语言模型", "大模型", "语言模型",
    "自然语言处理", "计算机视觉", "强化学习", "迁移学习", "联邦学习",
    "生成对抗网络", "扩散模型", "Transformer", "注意力机制", "自注意力",
    "预训练", "微调", "提示工程", "提示词", "思维链", "Chain of Thought",
    "RAG", "检索增强生成", "知识蒸馏", "模型压缩", "量化",
    # 模型
    "GPT", "GPT-4", "GPT-4o", "GPT-5", "ChatGPT", "o1", "o3",
    "BERT", "T5", "PaLM", "Gemini", "Claude", "Llama", "Llama2", "Llama3",
    "Sora", "DALL-E", "Midjourney", "Stable Diffusion",
    "Whisper", "Parler", "Qwen", "通义千问", "文心一言", "文心",
    "Kimi", "月之暗面", "DeepSeek", "深度求索",
    "Baichuan", "百川", "ChatGLM", "Yi", "零一万物",
    "Mistral", "Gemma", "Phi",
    # 公司/组织
    "OpenAI", "Google", "DeepMind", "Meta", "Microsoft", "微软", "谷歌",
    "Anthropic", "NVIDIA", "英伟达", "百度", "阿里", "字节跳动", "腾讯",
    "华为", "商汤", "旷视", "科大讯飞",
    # 人物
    "Sam Altman", "Altman", "奥特曼", "Ilya", "Sutskever",
    "黄仁勋", "Jensen", "李飞飞", "Hinton", "Bengio", "LeCun",
    "Karpathy", "Demis", "Hassabis", "Mustafa Suleyman",
    # 技术
    "AGI", "ASI", "AI Agent", "Agent", "多模态", "视觉语言模型",
    "文生图", "文生视频", "文生音频", "语音合成", "语音识别",
    "图像生成", "视频生成", "代码生成", "文本生成",
    "情感分析", "命名实体识别", "文本分类", "机器翻译",
    "目标检测", "语义分割", "图像分类",
    "LLM", "NLP", "CV", "ASR", "TTS",
    "API", "Fine-tuning", "Inference", "Training", "Benchmark",
    "GPU", "TPU", "CUDA", "PyTorch", "TensorFlow",
    "AutoGPT", "BabyAGI", "LangChain", "LlamaIndex",
    "CoT", "Few-shot", "Zero-shot", "In-context learning",
    "Alignment", "RLHF", "DPO", "Safety", "Hallucination",
    "幻觉", "对齐", "涌现能力", "泛化", "过拟合", "欠拟合",
    "CNN", "RNN", "LSTM", "GAN", "VAE", "GAN",
    "反向传播", "梯度下降", "学习率", "批量大小", "Dropout",
    "Embedding", "向量", "Token", "Tokenizer",
    "参数量", "算力", "推理", "部署", "开源", "闭源",
    "Copilot", "CodePilot", "Cursor", "Claude Code",
    "Figure", "Figure01", "Tesla Bot", "机器人", "具身智能",
    "世界模型", "物理世界", "自动驾驶", "端到端",
]


def init_jieba():
    """初始化jieba分词，加载AI术语"""
    for term in AI_TERMS:
        jieba.add_word(term, freq=1000)
    logger.info(f"jieba已加载 {len(AI_TERMS)} 个AI术语")


def clean_text(text):
    """清洗文本：去除HTML标签、特殊符号、表情等"""
    if not text:
        return ""
    # 去HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去B站表情代码 [xxx]
    text = re.sub(r'\[.*?\]', '', text)
    # 去特殊符号，保留中文、英文、数字、基本标点
    text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z0-9\s，。！？、；：""''（）《》【】\.\,\!\?\;\:\"\'\(\)\[\]\-\/]', '', text)
    # 合并空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text, remove_stopwords=True, pos_filter=False):
    """分词，可选去停用词和词性过滤"""
    words = jieba.lcut(text)
    if remove_stopwords:
        words = [w for w in words if w.strip() and w not in STOPWORDS]
    if pos_filter:
        # 只保留名词、动词、形容词
        words_with_pos = pseg.lcut(text)
        valid_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'ng', 'nrt', 'nrfg',
                     'v', 'vd', 'vn', 'vg', 'a', 'ad', 'an', 'ag',
                     'eng', 'x'}
        words = [w.word for w in words_with_pos
                 if w.flag in valid_pos and w.word.strip() and w.word not in STOPWORDS]
    return words


def preprocess_all():
    """完整预处理流程"""
    init_jieba()

    # 加载原始数据
    raw_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    with open(raw_file, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    logger.info(f"加载 {len(videos)} 个视频数据")

    # 加载ASR结果
    asr_file = os.path.join(PROCESSED_DIR, "asr_results.json")
    asr_dict = {}
    if os.path.exists(asr_file):
        with open(asr_file, 'r', encoding='utf-8') as f:
            asr_results = json.load(f)
        asr_dict = {r['bvid']: r['text'] for r in asr_results}
        logger.info(f"加载 {len(asr_dict)} 条ASR转写结果")

    # 处理每个视频
    processed = []
    for v in videos:
        bvid = v['bvid']

        # 传播端文本：仅标题 + 描述（不含ASR）
        title_clean = clean_text(v.get('title', ''))
        desc_clean = clean_text(v.get('desc', ''))
        propagator_text = f"{title_clean} {desc_clean}".strip()

        # 接收端文本：评论 + 弹幕
        comments = v.get('comments', [])
        danmakus = v.get('danmakus', [])
        comment_texts = [clean_text(c['content']) for c in comments if c.get('content')]
        danmaku_texts = [clean_text(d['text']) for d in danmakus if d.get('text')]
        receiver_text = " ".join(comment_texts + danmaku_texts).strip()

        # ASR音频转写端：独立第三端
        asr_text = clean_text(asr_dict.get(bvid, ''))

        # 分词（三端独立）
        prop_tokens = tokenize(propagator_text, remove_stopwords=True, pos_filter=True)
        recv_tokens = tokenize(receiver_text, remove_stopwords=True, pos_filter=True)
        asr_tokens = tokenize(asr_text, remove_stopwords=True, pos_filter=True) if asr_text else []

        processed.append({
            'bvid': bvid,
            'title': title_clean,
            'category': v.get('category', 'news'),
            'quality_score': v.get('quality_score', 0),
            'owner': v.get('owner', ''),
            'owner_mid': v.get('owner_mid', 0),
            'view': v.get('view', 0),
            'duration': int(v.get('duration', 0) or 0),
            'pubdate': v.get('pubdate', 0),
            'keyword': v.get('keyword', ''),
            # 传播端（标题+描述）
            'propagator_raw': propagator_text,
            'propagator_tokens': prop_tokens,
            # 接收端（评论+弹幕）
            'receiver_raw': receiver_text,
            'receiver_tokens': recv_tokens,
            'comment_count': len(comments),
            'danmaku_count': len(danmakus),
            # ASR音频转写端（独立第三端）
            'asr_raw': asr_text,
            'asr_tokens': asr_tokens,
            'has_asr': bool(asr_text),
        })

    # 保存
    output_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    logger.info(f"预处理完成: {len(processed)} 个视频, 保存到 {output_file}")

    # 统计
    total_prop_tokens = sum(len(p['propagator_tokens']) for p in processed)
    total_recv_tokens = sum(len(p['receiver_tokens']) for p in processed)
    total_asr_tokens = sum(len(p['asr_tokens']) for p in processed)
    has_asr_count = sum(1 for p in processed if p['has_asr'])
    logger.info(f"传播端总词数: {total_prop_tokens}, 接收端总词数: {total_recv_tokens}, ASR总词数: {total_asr_tokens}")
    logger.info(f"含ASR文本: {has_asr_count}/{len(processed)}")

    return processed


if __name__ == "__main__":
    preprocess_all()
