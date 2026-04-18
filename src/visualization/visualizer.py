"""
可视化模块 - 生成所有论文图表
图（表）名：中英文双语，5号黑体
图中文字：5号宋体
"""
import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from loguru import logger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ============================================================
# 字体配置：论文规范
# 5号字 ≈ 10.5pt
# 标题用黑体(SimHei)，正文/标签用宋体(SimSun)
# ============================================================
FONT_TITLE = fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10.5)   # 5号黑体
FONT_BODY = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=10.5)    # 5号宋体
FONT_BODY_SMALL = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=9) # 小5号宋体

plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_ax_font(ax):
    """统一设置坐标轴标签和刻度为宋体"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_BODY_SMALL)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontproperties=FONT_BODY)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontproperties=FONT_BODY)


def plot_wordcloud_comparison(prop_tokens_all, recv_tokens_all, asr_tokens_all=None, output_dir=OUTPUT_DIR):
    """三端词云对比"""
    from wordcloud import WordCloud

    ncols = 3 if asr_tokens_all else 2
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
    if ncols == 2:
        axes = list(axes)

    panels = [
        (axes[0], prop_tokens_all, "传播端（标题+描述）", "Propagator (Title+Desc)"),
        (axes[1], recv_tokens_all, "接收端（评论+弹幕）", "Receiver (Comments+Danmaku)"),
    ]
    if asr_tokens_all:
        panels.append((axes[2], asr_tokens_all, "ASR音频转写端", "ASR Transcript"))

    for ax, tokens, cn, en in panels:
        text = " ".join(tokens) if tokens else "无数据"
        wc = WordCloud(
            font_path="C:/Windows/Fonts/simhei.ttf",
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
        ).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"{cn}\n{en}", fontproperties=FONT_TITLE)
        ax.axis('off')

    fig.suptitle("三端词云对比\n  Three-way Word Cloud Comparison",
                 fontproperties=FONT_TITLE, y=0.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filepath = os.path.join(output_dir, "wordcloud_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"词云图已保存: {filepath}")


def plot_knowledge_density_distribution(kd_data, output_dir=OUTPUT_DIR):
    """答辩版：左右并列柱状图，独立比例尺，每柱精确百分比标注"""
    color_map = {'news-传播端': '#4e79a7', 'news-接收端': '#a0cbe8',
                 'interview-传播端': '#f28e2b', 'interview-接收端': '#ffbe7d'}
    width = 0.3
    positions = {
        'news':      {'prop': 0.0, 'recv': 0.35},
        'interview': {'prop': 1.2, 'recv': 1.55},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(top=0.78, bottom=0.1, left=0.08, right=0.97, wspace=0.3)

    all_handles, all_labels = [], []
    for cat in ['news', 'interview']:
        cat_data = [d for d in kd_data if d.get('category') == cat]
        if not cat_data:
            continue

        prop_kd = np.mean([d['propagator_knowledge_density'] for d in cat_data if d['propagator_knowledge_density'] > 0] or [0])
        recv_kd = np.mean([d['receiver_knowledge_density']   for d in cat_data if d['receiver_knowledge_density']   > 0] or [0])
        prop_ld = np.mean([d['propagator_lexical_density']   for d in cat_data] or [0])
        recv_ld = np.mean([d['receiver_lexical_density']     for d in cat_data] or [0])

        pp = positions[cat]['prop']
        rp = positions[cat]['recv']

        # 左：知识密度
        b1 = axes[0].bar(pp, prop_kd, width, color=color_map[f'{cat}-传播端'], label=f'{cat}-传播端')
        b2 = axes[0].bar(rp, recv_kd, width, color=color_map[f'{cat}-接收端'], label=f'{cat}-接收端')
        axes[0].text(pp, prop_kd + 0.001, f'{prop_kd:.1%}', ha='center', va='bottom', fontproperties=FONT_BODY_SMALL)
        axes[0].text(rp, recv_kd + 0.001, f'{recv_kd:.1%}', ha='center', va='bottom', fontproperties=FONT_BODY_SMALL)

        # 右：词汇密度
        axes[1].bar(pp, prop_ld, width, color=color_map[f'{cat}-传播端'])
        axes[1].bar(rp, recv_ld, width, color=color_map[f'{cat}-接收端'])
        axes[1].text(pp, prop_ld + 0.003, f'{prop_ld:.1%}', ha='center', va='bottom', fontproperties=FONT_BODY_SMALL)
        axes[1].text(rp, recv_ld + 0.003, f'{recv_ld:.1%}', ha='center', va='bottom', fontproperties=FONT_BODY_SMALL)

        all_handles += [b1, b2]
        all_labels  += [f'{cat}-传播端', f'{cat}-接收端']

    xticks  = [(positions[c]['prop'] + positions[c]['recv']) / 2 for c in ['news', 'interview']]
    xlabels = ['News', 'Interview']

    # 左：知识密度，比例尺贴近数据，衰减差异一目了然
    axes[0].set_title("知识密度对比\nKnowledge Density Comparison", fontproperties=FONT_TITLE, pad=8)
    axes[0].set_ylabel('知识密度 Knowledge Density', fontproperties=FONT_BODY)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xlabels, fontproperties=FONT_BODY)
    axes[0].set_ylim(0, 0.07)
    axes[0].set_xlim(-0.3, 1.9)
    set_ax_font(axes[0])

    # 右：词汇密度，截断y轴避免柱子过高
    axes[1].set_title("词汇密度对比\nLexical Density Comparison", fontproperties=FONT_TITLE, pad=8)
    axes[1].set_ylabel('词汇密度 Lexical Density', fontproperties=FONT_BODY)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xlabels, fontproperties=FONT_BODY)
    axes[1].set_ylim(0.3, 0.58)
    axes[1].set_xlim(-0.3, 1.9)
    set_ax_font(axes[1])

    fig.legend(all_handles, all_labels, loc='upper center', ncol=4,
               prop=FONT_BODY_SMALL, bbox_to_anchor=(0.5, 0.90),
               frameon=True, edgecolor='#cccccc')
    fig.suptitle("传播端与接收端密度对比\nDensity Comparison: Propagator vs Receiver",
                 fontproperties=FONT_TITLE, y=0.99)

    filepath = os.path.join(output_dir, "knowledge_density_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"密度对比图已保存: {filepath}")


def plot_video_type_distribution(kd_data, output_dir=OUTPUT_DIR):
    """视频类型分布饼图"""
    type_counts = Counter(d['video_type'] for d in kd_data)

    fig, ax = plt.subplots(figsize=(8, 8))
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)],
        startangle=90, textprops={'fontproperties': FONT_BODY}
    )
    for at in autotexts:
        at.set_fontproperties(FONT_BODY_SMALL)

    ax.set_title("视频内容类型分布\n Video Content Type Distribution",
                 fontproperties=FONT_TITLE)

    filepath = os.path.join(output_dir, "video_type_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"类型分布图已保存: {filepath}")


def plot_sentiment_distribution(annotations, output_dir=OUTPUT_DIR):
    """三端情感/立场分布图"""
    sides = [
        ('propagator_annotation', '传播端', 'Propagator'),
        ('receiver_annotation', '接收端', 'Receiver'),
        ('asr_annotation', 'ASR转写端', 'ASR Transcript'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (key, cn, en) in enumerate(sides):
        sentiments = []
        stances = []
        for ann in annotations:
            d = ann.get(key)
            if d and isinstance(d, dict):
                sentiments.append(d.get('overall_sentiment', 'neutral'))
                stances.append(d.get('stance', 'neutral'))

        # 情感行
        ax = axes[0][col]
        counts = Counter(sentiments)
        labels = ['positive', 'neutral', 'negative']
        values = [counts.get(l, 0) for l in labels]
        colors = ['#59a14f', '#4e79a7', '#e15759']
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"{cn}情感分布\n{en} Sentiment", fontproperties=FONT_TITLE)
        ax.set_ylabel('数量 Count', fontproperties=FONT_BODY)
        set_ax_font(ax)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(val), ha='center', fontproperties=FONT_BODY_SMALL)

        # 立场行
        ax = axes[1][col]
        counts = Counter(stances)
        labels = ['support', 'neutral', 'oppose']
        values = [counts.get(l, 0) for l in labels]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"{cn}立场分布\n{en} Stance", fontproperties=FONT_TITLE)
        ax.set_ylabel('数量 Count', fontproperties=FONT_BODY)
        set_ax_font(ax)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(val), ha='center', fontproperties=FONT_BODY_SMALL)

    fig.suptitle("三端情感与立场分布\n Three-way Sentiment and Stance Distribution",
                 fontproperties=FONT_TITLE, y=0.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filepath = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"情感分布图已保存: {filepath}")


def plot_topic_evolution(topic_model, docs, timestamps, output_dir=OUTPUT_DIR):
    """主题演化趋势图"""
    try:
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=BERTOPIC_NR_BINS)

        fig, ax = plt.subplots(figsize=(14, 7))
        top_topics = topics_over_time[topics_over_time['Topic'] != -1].groupby('Topic')['Frequency'].sum().nlargest(8).index

        for topic_id in top_topics:
            topic_data = topics_over_time[topics_over_time['Topic'] == topic_id]
            words = topic_model.get_topic(topic_id)
            label = "_".join([w[0] for w in words[:3]]) if words else f"Topic_{topic_id}"
            ax.plot(topic_data['Timestamp'], topic_data['Frequency'], label=label, linewidth=2)

        ax.set_title(" AI话题主题演化趋势\n Topic Evolution Over Time",
                     fontproperties=FONT_TITLE)
        ax.set_xlabel('时间 Time', fontproperties=FONT_BODY)
        ax.set_ylabel('主题频率 Topic Frequency', fontproperties=FONT_BODY)
        ax.legend(fontsize=8, loc='upper left', prop=FONT_BODY_SMALL)
        ax.tick_params(axis='x', rotation=45)
        set_ax_font(ax)

        filepath = os.path.join(output_dir, "topic_evolution.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"主题演化图已保存: {filepath}")
    except Exception as e:
        logger.warning(f"主题演化图生成失败: {e}")


# def plot_kd_comparison_radar(kd_data, output_dir=OUTPUT_DIR):
    """传播端vs接收端知识密度对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.18

    color_map = {'news-传播端': '#4e79a7', 'news-接收端': '#a0cbe8',
                 'interview-传播端': '#f28e2b', 'interview-接收端': '#ffbe7d'}
    offset = 0
    for cat in ['news', 'interview']:
        cat_data = [d for d in kd_data if d.get('category') == cat]
        if not cat_data:
            continue
        prop_vals = [
            np.mean([d['propagator_knowledge_density'] for d in cat_data]),
            np.mean([d['propagator_lexical_density'] for d in cat_data]),
        ]
        recv_vals = [
            np.mean([d['receiver_knowledge_density'] for d in cat_data]),
            np.mean([d['receiver_lexical_density'] for d in cat_data]),
        ]
        ax.bar(x + offset * width, prop_vals, width,
               label=f'{cat}-传播端 Propagator', color=color_map[f'{cat}-传播端'])
        ax.bar(x + (offset + 1) * width, recv_vals, width,
               label=f'{cat}-接收端 Receiver', color=color_map[f'{cat}-接收端'])
        offset += 2

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(['知识密度\nKnowledge Density', '词汇密度\nLexical Density'],
                       fontproperties=FONT_BODY)
    ax.set_title(" 传播端与接收端密度对比\n Density Comparison: Propagator vs Receiver",
                 fontproperties=FONT_TITLE)
    ax.legend(prop=FONT_BODY_SMALL)
    set_ax_font(ax)

    filepath = os.path.join(output_dir, "kd_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"密度对比图已保存: {filepath}")

def plot_kd_comparison_radar(kd_data, output_dir=OUTPUT_DIR):
    """传播端vs接收端密度对比（答辩版：左右子图 + 独立比例尺 + 数值标注 + 图例不遮标题）"""
    # 增加顶部空间给图例和总标题
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(top=0.78, bottom=0.1, left=0.08, right=0.97, wspace=0.3)

    color_map = {'news-传播端': '#4e79a7', 'news-接收端': '#a0cbe8',
                 'interview-传播端': '#f28e2b', 'interview-接收端': '#ffbe7d'}
    width = 0.3
    # 两组各两根柱，组间距1.2
    positions = {
        'news':      {'prop': 0.0, 'recv': 0.35},
        'interview': {'prop': 1.2, 'recv': 1.55},
    }

    all_handles, all_labels = [], []
    for cat in ['news', 'interview']:
        cat_data = [d for d in kd_data if d.get('category') == cat]
        if not cat_data:
            continue

        prop_kd = np.mean([d['propagator_knowledge_density'] for d in cat_data])
        prop_ld = np.mean([d['propagator_lexical_density'] for d in cat_data])
        recv_kd = np.mean([d['receiver_knowledge_density'] for d in cat_data])
        recv_ld = np.mean([d['receiver_lexical_density'] for d in cat_data])

        pp = positions[cat]['prop']
        rp = positions[cat]['recv']

        for ax, pv, rv, pad in [
            (axes[0], prop_kd, recv_kd, 0.001),
            (axes[1], prop_ld, recv_ld, 0.003),
        ]:
            b1 = ax.bar(pp, pv, width, color=color_map[f'{cat}-传播端'],
                        label=f'{cat}-传播端')
            b2 = ax.bar(rp, rv, width, color=color_map[f'{cat}-接收端'],
                        label=f'{cat}-接收端')
            ax.text(pp, pv + pad, f'{pv:.1%}', ha='center', va='bottom',
                    fontproperties=FONT_BODY_SMALL)
            ax.text(rp, rv + pad, f'{rv:.1%}', ha='center', va='bottom',
                    fontproperties=FONT_BODY_SMALL)
            if ax is axes[0]:
                all_handles += [b1, b2]
                all_labels  += [f'{cat}-传播端', f'{cat}-接收端']

    xticks  = [(positions[c]['prop'] + positions[c]['recv']) / 2 for c in ['news', 'interview']]
    xlabels = ['News', 'Interview']

    # 左侧：知识密度，比例尺贴近数据，衰减差异一目了然
    axes[0].set_title("知识密度对比\nKnowledge Density Comparison", fontproperties=FONT_TITLE, pad=8)
    axes[0].set_ylabel('知识密度 Knowledge Density', fontproperties=FONT_BODY)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xlabels, fontproperties=FONT_BODY)
    axes[0].set_ylim(0, 0.07)
    axes[0].set_xlim(-0.3, 1.9)
    set_ax_font(axes[0])

    # 右侧：词汇密度，截断y轴避免柱子过高
    axes[1].set_title("词汇密度对比\nLexical Density Comparison", fontproperties=FONT_TITLE, pad=8)
    axes[1].set_ylabel('词汇密度 Lexical Density', fontproperties=FONT_BODY)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xlabels, fontproperties=FONT_BODY)
    axes[1].set_ylim(0.3, 0.58)
    axes[1].set_xlim(-0.3, 1.9)
    set_ax_font(axes[1])

    # 图例放在两子图正上方，总标题再往上，互不遮挡
    fig.legend(all_handles, all_labels, loc='upper center', ncol=4,
               prop=FONT_BODY_SMALL, bbox_to_anchor=(0.5, 0.90),
               frameon=True, edgecolor='#cccccc')

    fig.suptitle("传播端与接收端密度对比\nDensity Comparison: Propagator vs Receiver",
                 fontproperties=FONT_TITLE, y=0.99)

    filepath = os.path.join(output_dir, "kd_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"密度对比图已保存: {filepath}")



def plot_entity_frequency(annotations, output_dir=OUTPUT_DIR, top_n=20):
    """三端实体频率Top-N柱状图"""
    sides = [
        ('propagator_annotation', '传播端Top实体', 'Propagator Top Entities', 'steelblue'),
        ('receiver_annotation', '接收端Top实体', 'Receiver Top Entities', 'coral'),
        ('asr_annotation', 'ASR转写端Top实体', 'ASR Transcript Top Entities', '#59a14f'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax, (key, cn, en, color) in zip(axes, sides):
        entities = Counter()
        for ann in annotations:
            d = ann.get(key, {})
            if d and isinstance(d, dict) and 'entities' in d:
                for ent in d['entities']:
                    entities[ent['name']] += 1

        top = entities.most_common(top_n)
        if top:
            names, values = zip(*top)
            ax.barh(range(len(names)), values, color=color, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontproperties=FONT_BODY_SMALL)
            ax.invert_yaxis()
        ax.set_title(f"{cn}\n{en}", fontproperties=FONT_TITLE)
        ax.set_xlabel('出现频次 Frequency', fontproperties=FONT_BODY)
        set_ax_font(ax)

    fig.suptitle(" 三端命名实体频率分布\n Three-way Named Entity Frequency Distribution",
                 fontproperties=FONT_TITLE, y=0.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filepath = os.path.join(output_dir, "entity_frequency.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"实体频率图已保存: {filepath}")


def run_all_visualizations():
    """生成全部可视化图表"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载预处理数据
    proc_file = os.path.join(PROCESSED_DIR, "processed_videos.json")
    if os.path.exists(proc_file):
        data = load_json(proc_file)

        # 三端词云对比
        prop_tokens_all = []
        recv_tokens_all = []
        asr_tokens_all = []
        for d in data:
            prop_tokens_all.extend(d.get('propagator_tokens', []))
            recv_tokens_all.extend(d.get('receiver_tokens', []))
            asr_tokens_all.extend(d.get('asr_tokens', []))
        plot_wordcloud_comparison(prop_tokens_all, recv_tokens_all,
                                 asr_tokens_all if asr_tokens_all else None)
    else:
        logger.warning("未找到预处理数据，跳过词云")

    # 知识密度图
    kd_file = os.path.join(RESULTS_DIR, "knowledge_density.json")
    if os.path.exists(kd_file):
        kd_data = load_json(kd_file)['details']
        plot_knowledge_density_distribution(kd_data)
        plot_video_type_distribution(kd_data)
        plot_kd_comparison_radar(kd_data)
    else:
        logger.warning("未找到知识密度数据")

    # 情感分布图
    ann_file = os.path.join(RESULTS_DIR, "annotations.json")
    if os.path.exists(ann_file):
        annotations = load_json(ann_file)
        plot_sentiment_distribution(annotations)
        plot_entity_frequency(annotations)
    else:
        logger.warning("未找到标注数据")

    logger.info(f"全部可视化图表已生成到 {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all_visualizations()
