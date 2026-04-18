"""
LLM标注模块 - 使用GLM-4-Flash进行NER、情感分析、立场检测
支持多API Key轮询 + 多线程并发标注
"""
import os
import sys
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from loguru import logger

import requests

# 从config读取API Keys
API_KEYS = ZHIPU_API_KEYS

NUM_WORKERS = 10  # 10个并发线程

# 线程安全的Key轮询计数器
_key_counter = 0
_key_lock = threading.Lock()


def get_next_api_key():
    """线程安全地轮询获取下一个API Key"""
    global _key_counter
    with _key_lock:
        key = API_KEYS[_key_counter % len(API_KEYS)]
        _key_counter += 1
    return key


# 多任务Prompt模板
ANNOTATION_PROMPT = """分析以下文本，直接返回JSON（不要```包裹）：
文本：{text}
返回格式：
{{"entities":[{{"name":"实体名","type":"PERSON/ORG/TECH/PRODUCT/EVENT","sentiment":"positive/negative/neutral"}}],
"overall_sentiment":"positive/negative/neutral","sentiment_confidence":0.8,"stance":"support/oppose/neutral",
"stance_confidence":0.8,"key_topics":["主题1"],"summary":"一句话摘要"}}
要求：entities只提取最重要的5个AI相关实体，只返回JSON"""


import re


def extract_json_from_text(text):
    """从GLM返回的文本中鲁棒提取JSON，处理截断、格式问题"""
    if not text:
        return None
    text = text.strip()

    # 方法1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 方法2：提取```json...```代码块
    for pat in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                # 代码块内容可能也被截断，继续尝试修复
                text = m.group(1).strip()
                break

    # 方法3：找{到}之间的内容
    first_brace = text.find('{')
    if first_brace == -1:
        return None
    candidate = text[first_brace:]

    # 方法4：尝试直接解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 方法5：修复截断的JSON（核心修复逻辑）
    fixed = candidate
    fixed = re.sub(r',\s*$', '', fixed)          # 去尾部逗号
    fixed = re.sub(r',\s*}', '}', fixed)         # }前逗号
    fixed = re.sub(r',\s*]', ']', fixed)         # ]前逗号
    fixed = re.sub(r'"\s*$', '"', fixed)         # 截断的字符串

    # 补全未闭合的括号
    open_braces = fixed.count('{') - fixed.count('}')
    open_brackets = fixed.count('[') - fixed.count(']')
    # 截断位置可能在字符串中间，先尝试截断到最后一个完整的值
    # 找最后一个完整的 } 或 ] 后截断
    last_complete = max(fixed.rfind('}'), fixed.rfind(']'))
    if last_complete > 0 and (open_braces > 0 or open_brackets > 0):
        truncated = fixed[:last_complete + 1]
        # 重新计算
        ob = truncated.count('{') - truncated.count('}')
        ol = truncated.count('[') - truncated.count(']')
        truncated += ']' * max(ol, 0) + '}' * max(ob, 0)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # 方法6：暴力补全
    fixed += ']' * max(open_brackets, 0) + '}' * max(open_braces, 0)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 方法7：只提取到entities结束（最低保障）
    m = re.search(r'\{[^{}]*"entities"\s*:\s*\[.*?\]', fixed, re.DOTALL)
    if m:
        minimal = m.group(0)
        # 补上缺失字段的默认值
        if '"overall_sentiment"' not in minimal:
            minimal += ', "overall_sentiment": "neutral", "sentiment_confidence": 0.5, "stance": "neutral", "stance_confidence": 0.5, "key_topics": [], "summary": ""'
        minimal += '}'
        try:
            return json.loads(minimal)
        except json.JSONDecodeError:
            pass

    return None


def call_glm_api(prompt, api_key=None, retry=3):
    """调用智谱GLM-4-Flash API（支持指定Key）"""
    if api_key is None:
        api_key = get_next_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    for attempt in range(retry):
        try:
            resp = requests.post(
                f"{ZHIPU_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            if resp.status_code == 200:
                result = resp.json()
                content = result['choices'][0]['message']['content']
                parsed = extract_json_from_text(content)
                if parsed is not None:
                    return parsed
                logger.warning(f"JSON提取失败，attempt {attempt+1}, content[:100]={content[:100]}")
            elif resp.status_code == 429:
                api_key = get_next_api_key()
                headers["Authorization"] = f"Bearer {api_key}"
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"API返回 {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            logger.warning(f"API调用失败: {e}, attempt {attempt+1}")

        if attempt < retry - 1:
            time.sleep(1 + attempt)

    return None


def annotate_single(idx, text, side="prop"):
    """标注单条文本（供线程池调用）"""
    if not text or len(text.strip()) < 10:
        return idx, side, None

    # ASR文本分段采样（开头+中间+结尾各10000字，总计3万字覆盖全文）
    if side == 'asr' and len(text) > 30000:
        seg_len = 10000
        head = text[:seg_len]
        mid_start = len(text) // 2 - seg_len // 2
        mid = text[mid_start:mid_start + seg_len]
        tail = text[-seg_len:]
        truncated = f"[开头]{head}\n[中间]{mid}\n[结尾]{tail}"
    else:
        truncated = text[:30000] if side == 'asr' else text[:2000]

    prompt = ANNOTATION_PROMPT.format(text=truncated)
    api_key = get_next_api_key()
    result = call_glm_api(prompt, api_key=api_key)

    # 补全缺失字段
    if result and isinstance(result, dict):
        result.setdefault('overall_sentiment', 'neutral')
        result.setdefault('sentiment_confidence', 0.5)
        result.setdefault('stance', 'neutral')
        result.setdefault('stance_confidence', 0.5)
        result.setdefault('key_topics', [])
        result.setdefault('summary', '')
        if 'entities' not in result:
            result['entities'] = []

    return idx, side, result


def batch_annotate_parallel(tasks, label=""):
    """
    多线程并行标注
    tasks: list of (idx, text, side)
    返回: dict of {(idx, side): result}
    """
    results = {}
    total = len(tasks)
    completed = 0
    lock = threading.Lock()

    def worker(task):
        nonlocal completed
        idx, text, side = task
        result = annotate_single(idx, text, side)
        with lock:
            completed += 1
            if completed % 50 == 0:
                logger.info(f"[{label}] 并发标注进度: {completed}/{total}")
        return result

    logger.info(f"[{label}] 启动 {NUM_WORKERS} 线程并发标注 {total} 条")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                idx, side, result = future.result()
                results[(idx, side)] = result
            except Exception as e:
                logger.warning(f"标注异常: {e}")

    success = sum(1 for v in results.values() if v is not None)
    logger.info(f"[{label}] 标注完成: {success}/{total} 成功")
    return results


def save_annotations(results, filepath):
    """保存标注结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_annotation():
    """运行完整标注流程（10线程并发）"""
    proc_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    with open(proc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"加载 {len(data)} 条数据待标注")

    # 加载已有标注
    ann_file = os.path.join(RESULTS_DIR, "annotations.json")
    existing_map = {}
    if os.path.exists(ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing_map = {a['bvid']: a for a in existing}
        logger.info(f"已有标注: {len(existing_map)} 条")

    # 构建待标注任务列表（三端）
    tasks = []
    for i, d in enumerate(data):
        bvid = d['bvid']
        existing_ann = existing_map.get(bvid, {})

        # 传播端
        prop_text = d.get('propagator_raw', '')
        if not existing_ann.get('propagator_annotation') and prop_text:
            tasks.append((i, prop_text, 'prop'))

        # 接收端
        recv_text = d.get('receiver_raw', '')
        if not existing_ann.get('receiver_annotation') and recv_text:
            tasks.append((i, recv_text, 'recv'))

        # ASR音频转写端
        asr_text = d.get('asr_raw', '')
        if not existing_ann.get('asr_annotation') and asr_text and len(asr_text.strip()) > 10:
            tasks.append((i, asr_text, 'asr'))

    logger.info(f"需要标注: {len(tasks)} 条（已跳过已有标注）")

    if not tasks:
        logger.info("无需新增标注")
        return list(existing_map.values())

    # 并发标注
    results = batch_annotate_parallel(tasks, label="GLM-4-Flash")

    # 合并结果（三端）
    annotations = []
    for i, d in enumerate(data):
        bvid = d['bvid']
        existing_ann = existing_map.get(bvid, {})

        prop_ann = results.get((i, 'prop'), existing_ann.get('propagator_annotation'))
        recv_ann = results.get((i, 'recv'), existing_ann.get('receiver_annotation'))
        asr_ann = results.get((i, 'asr'), existing_ann.get('asr_annotation'))

        annotations.append({
            "bvid": bvid,
            "category": d.get('category', ''),
            "propagator_annotation": prop_ann,
            "receiver_annotation": recv_ann,
            "asr_annotation": asr_ann,
        })

    # 保存
    save_annotations(annotations, ann_file)
    prop_ok = sum(1 for a in annotations if a.get('propagator_annotation'))
    recv_ok = sum(1 for a in annotations if a.get('receiver_annotation'))
    asr_ok = sum(1 for a in annotations if a.get('asr_annotation'))
    logger.info(f"标注完成: {len(annotations)} 条 (传播端{prop_ok}, 接收端{recv_ok}, ASR{asr_ok})")

    return annotations


if __name__ == "__main__":
    run_annotation()
