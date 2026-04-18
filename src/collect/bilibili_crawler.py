"""
B站数据采集模块 - 视频元数据、评论、弹幕
使用 bilibili-api-python 异步采集
基于六维视频质量评价体系: 时效性、来源可信度、内容相关性、内容深度、传播力、受众价值
"""
import asyncio
import json
import math
import random
import os
import sys
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

from bilibili_api import video, comment, search, Credential
from bilibili_api.search import SearchObjectType
from loguru import logger

# 访谈类标题识别关键词
INTERVIEW_TITLE_KEYWORDS = [
    "访谈", "采访", "对话", "专访", "圆桌", "对谈", "座谈",
    "Altman", "黄仁勋", "Ilya", "Lex Fridman", "李飞飞",
    "Hinton", "Sam", "Jensen", "Demis", "Karpathy",
]

# AI领域重大事件时间节点（Unix timestamp近似值，用于时效性评分）
AI_MILESTONE_DATES = [
    1669852800,  # 2022-12-01 ChatGPT发布
    1675296000,  # 2023-02-02 Bing Chat
    1682640000,  # 2023-04-28 GPT-4发布后热潮
    1696118400,  # 2023-10-01 GPT-4V
    1702425600,  # 2023-12-13 Gemini发布
    1706745600,  # 2024-02-01 Sora预告
    1713398400,  # 2024-04-18 Llama3
    1721865600,  # 2024-07-25 GPT-4o
    1734835200,  # 2024-12-22 o3发布
    1740787200,  # 2025-03-01 GPT-5/新模型周期
]


def get_credential():
    """创建B站认证对象"""
    return Credential(
        sessdata=BILIBILI_SESSDATA,
        bili_jct=BILIBILI_BILI_JCT,
        buvid3=BILIBILI_BUVID3,
        dedeuserid=BILIBILI_DEDEUSERID
    )


def parse_duration(val):
    """将搜索结果中的duration转为秒数，支持 '3:14' 格式和纯数字"""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        val = val.strip()
        if ':' in val:
            parts = val.split(':')
            try:
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            except ValueError:
                return 0
        try:
            return int(val)
        except ValueError:
            return 0
    return 0


def is_blacklisted(title):
    """检查标题是否命中黑名单（教程/卖课等低质量内容）"""
    if not title:
        return False
    for kw in TITLE_BLACKLIST:
        if kw in title:
            return True
    return False


def classify_video(keyword, title):
    """根据搜索关键词和标题判断视频分类: news / interview"""
    if keyword in INTERVIEW_KEYWORDS:
        return "interview"
    for kw in INTERVIEW_TITLE_KEYWORDS:
        if kw.lower() in title.lower():
            return "interview"
    return "news"


def compute_quality_score(video_data, category="news"):
    """
    六维视频质量评分（满分100）
    D1 时效性(20) + D2 来源可信度(15) + D3 内容相关性(25)
    + D4 内容深度(15) + D5 传播力(15) + D6 受众价值(10)
    """
    title = video_data.get('title', '')
    desc = video_data.get('desc', '')
    view = max(int(video_data.get('view', 0) or 0), 1)
    like = int(video_data.get('like', 0) or 0)
    reply = int(video_data.get('reply_count', 0) or 0)
    danmaku_count = int(video_data.get('danmaku_count', 0) or 0)
    duration_sec = max(int(video_data.get('duration', 0) or 0), 1)
    pubdate = int(video_data.get('pubdate', 0) or 0)
    owner_mid = int(video_data.get('owner_mid', 0) or 0)
    combined_text = (title + " " + desc)

    score = 0

    # ---- D1 时效性 (0-20分) ----
    # 距离AI重大事件越近分越高
    timeliness_score = 0
    if pubdate > 0:
        for milestone_ts in AI_MILESTONE_DATES:
            days_diff = abs(pubdate - milestone_ts) / 86400
            if days_diff <= 7:
                timeliness_score = max(timeliness_score, 20)
            elif days_diff <= 30:
                timeliness_score = max(timeliness_score, 15)
            elif days_diff <= 90:
                timeliness_score = max(timeliness_score, 8)
    score += timeliness_score

    # 标题含时效性加分词
    timeliness_bonus_words = ["最新", "发布", "突破", "更新", "重磅", "刚刚", "突发", "首发"]
    for bw in timeliness_bonus_words:
        if bw in title:
            score += 3
            break

    # ---- D2 来源可信度 (0-15分) ----
    # 推荐UP主加分
    if RECOMMENDED_UP_MIDS and owner_mid in RECOMMENDED_UP_MIDS:
        score += 15
    else:
        # 基于播放量的间接可信度（高播放UP主更可信）
        if view >= 100000:
            score += 8
        elif view >= 50000:
            score += 5
        elif view >= 10000:
            score += 3
        # 描述中引用一手信源加分
        source_hints = ["官方", "原版", "翻译", "字幕组", "转载", "来源"]
        for sh in source_hints:
            if sh in combined_text:
                score += 4
                break

    # ---- D3 内容相关性 (0-25分) ----
    # 基础分（不在黑名单就给）
    score += 15
    # 加分关键词命中
    bonus_hit = 0
    for bw in TITLE_BONUS_WORDS:
        if bw in title:
            bonus_hit += 1
    score += min(10, bonus_hit * 5)

    # ---- D4 内容深度 (0-15分) ----
    duration_min = duration_sec / 60
    if category == "interview":
        if duration_min >= 30:
            score += 15
        elif duration_min >= 15:
            score += 10
        elif duration_min >= 10:
            score += 6
        elif duration_min >= 5:
            score += 3
    else:  # news
        if duration_min >= 10:
            score += 15
        elif duration_min >= 5:
            score += 10
        elif duration_min >= 3:
            score += 6
        elif duration_min >= 2:
            score += 3

    # ---- D5 传播力 (0-15分) ----
    # 播放量log缩放
    play_score = min(8, math.log10(max(view, 1)) * 1.2)
    score += play_score
    # 点赞率
    like_rate = like / view
    if like_rate >= 0.08:
        score += 4
    elif like_rate >= 0.05:
        score += 3
    elif like_rate >= 0.03:
        score += 2
    # 评论数
    if reply >= 100:
        score += 3
    elif reply >= 30:
        score += 1

    # ---- D6 受众价值 (0-10分) ----
    # 弹幕密度（弹幕数/分钟）
    danmaku_density = danmaku_count / max(duration_min, 1)
    if danmaku_density >= 10:
        score += 6
    elif danmaku_density >= 5:
        score += 4
    elif danmaku_density >= 2:
        score += 2
    # 评论数作为互动价值
    if reply >= 200:
        score += 4
    elif reply >= 50:
        score += 2

    return min(int(score), 100)


async def search_videos(keyword, page=1, page_size=20):
    """按关键词搜索视频"""
    result = await search.search_by_type(
        keyword=keyword,
        search_type=SearchObjectType.VIDEO,
        page=page,
        page_size=page_size
    )
    return result


async def get_video_info(bvid, credential):
    """获取视频详情"""
    v = video.Video(bvid=bvid, credential=credential)
    return await v.get_info()


async def get_comments(aid, credential, max_pages=5):
    """获取视频评论（前max_pages页）"""
    comments = []
    try:
        for page in range(1, max_pages + 1):
            result = await comment.get_comments(
                oid=aid,
                type_=comment.CommentResourceType.VIDEO,
                page_index=page,
                credential=credential
            )
            if result and 'replies' in result:
                for r in result['replies']:
                    comments.append({
                        'content': r['content']['message'],
                        'like': r['like'],
                        'ctime': r['ctime'],
                        'uname': r['member']['uname']
                    })
                await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
            else:
                break
    except Exception as e:
        logger.warning(f"获取评论失败 aid={aid}: {e}")
    return comments


async def get_danmaku(bvid, credential):
    """获取视频弹幕"""
    danmakus = []
    try:
        v = video.Video(bvid=bvid, credential=credential)
        dms = await v.get_danmakus(page_index=0)
        for dm in dms:
            danmakus.append({
                'text': dm.text,
                'time': dm.dm_time,
                'weight': dm.weight
            })
    except Exception as e:
        logger.warning(f"获取弹幕失败 bvid={bvid}: {e}")
    return danmakus


async def collect_all():
    """完整采集流程（含质量评分和分类）"""
    credential = get_credential()
    all_videos = []
    seen_bvids = set()
    category_counts = {"news": 0, "interview": 0}

    # 加载断点续传
    checkpoint_file = os.path.join(RAW_DIR, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            seen_bvids = set(checkpoint.get('seen_bvids', []))
            all_videos = checkpoint.get('videos', [])
        # 恢复分类计数
        for v in all_videos:
            cat = v.get('category', 'news')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        logger.info(f"断点续传: 已有 {len(seen_bvids)} 个视频 (news:{category_counts['news']}, interview:{category_counts['interview']})")

    for keyword in SEARCH_KEYWORDS:
        # 检查是否两类都已满
        if (category_counts['news'] >= TARGET_COUNT_PER_CATEGORY and
                category_counts['interview'] >= TARGET_COUNT_PER_CATEGORY):
            logger.info("两类视频均已达到目标数量")
            break

        logger.info(f"搜索关键词: {keyword}")
        for page in range(1, 11):  # 每个关键词最多10页
            try:
                result = await search_videos(keyword, page=page)
                if not result or 'result' not in result:
                    break

                video_list = result.get('result', [])
                if not video_list:
                    break

                for item in video_list:
                    bvid = item.get('bvid', '')
                    title = item.get('title', '')
                    # 清除HTML标签
                    clean_title = re.sub(r'<[^>]+>', '', title)

                    if not bvid or bvid in seen_bvids:
                        continue

                    # 黑名单过滤
                    if is_blacklisted(clean_title):
                        logger.debug(f"黑名单过滤: {clean_title[:30]}")
                        continue

                    # 分类识别
                    category = classify_video(keyword, clean_title)

                    # 该分类已满则跳过
                    if category_counts.get(category, 0) >= TARGET_COUNT_PER_CATEGORY:
                        continue

                    # 差异化时长门槛
                    min_dur = INTERVIEW_MIN_DURATION_SEC if category == "interview" else MIN_DURATION_SEC

                    # 筛选条件
                    play = int(item.get('play', 0) or 0)
                    duration = parse_duration(item.get('duration', 0))
                    if play < MIN_PLAY_COUNT or duration < min_dur:
                        continue

                    seen_bvids.add(bvid)
                    logger.info(f"[{len(all_videos)+1}] {bvid} [{category}] - {clean_title[:40]}... 播放:{play}")

                    # 获取详细信息
                    try:
                        info = await get_video_info(bvid, credential)
                        await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

                        # 获取评论
                        aid = info.get('aid', 0)
                        comments = await get_comments(aid, credential, max_pages=3)

                        # 获取弹幕
                        danmakus = await get_danmaku(bvid, credential)

                        video_data = {
                            'bvid': bvid,
                            'title': info.get('title', ''),
                            'desc': info.get('desc', ''),
                            'owner': info.get('owner', {}).get('name', ''),
                            'owner_mid': info.get('owner', {}).get('mid', 0),
                            'view': info.get('stat', {}).get('view', 0),
                            'danmaku_count': info.get('stat', {}).get('danmaku', 0),
                            'reply_count': info.get('stat', {}).get('reply', 0),
                            'like': info.get('stat', {}).get('like', 0),
                            'duration': info.get('duration', 0),
                            'pubdate': info.get('pubdate', 0),
                            'tag': info.get('tag', ''),
                            'keyword': keyword,
                            'category': category,
                            'comments': comments,
                            'danmakus': danmakus,
                            'collect_time': datetime.now().isoformat()
                        }

                        # 计算质量评分
                        quality_score = compute_quality_score(video_data, category)
                        video_data['quality_score'] = quality_score
                        logger.info(f"  质量评分: {quality_score}/100")

                        # 低于门槛则跳过
                        if quality_score < QUALITY_SCORE_THRESHOLD:
                            logger.debug(f"  评分不足({quality_score}<{QUALITY_SCORE_THRESHOLD})，跳过")
                            continue

                        all_videos.append(video_data)
                        category_counts[category] = category_counts.get(category, 0) + 1

                    except Exception as e:
                        logger.error(f"处理视频 {bvid} 失败: {e}")

                    # 控制总量
                    if len(all_videos) >= MAX_VIDEOS:
                        break

                    # 每50个视频保存一次断点
                    if len(all_videos) % 50 == 0 and len(all_videos) > 0:
                        checkpoint = {'seen_bvids': list(seen_bvids), 'videos': all_videos}
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump(checkpoint, f, ensure_ascii=False)
                        logger.info(f"断点保存: {len(all_videos)} 个视频 (news:{category_counts['news']}, interview:{category_counts['interview']})")

                await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

                if len(all_videos) >= MAX_VIDEOS:
                    break

            except Exception as e:
                logger.error(f"搜索 {keyword} 第{page}页失败: {e}")
                await asyncio.sleep(5)

        if len(all_videos) >= MAX_VIDEOS:
            break

    # 按质量评分降序排列
    all_videos.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

    # 保存结果
    output_file = os.path.join(RAW_DIR, "bilibili_videos.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_videos, f, ensure_ascii=False, indent=2)
    logger.info(f"采集完成: 共 {len(all_videos)} 个视频 (news:{category_counts['news']}, interview:{category_counts['interview']})")
    logger.info(f"保存到 {output_file}")

    return all_videos


if __name__ == "__main__":
    asyncio.run(collect_all())
