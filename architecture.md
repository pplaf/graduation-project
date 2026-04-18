# Project Architecture

## Overview
基于文本挖掘的视频平台AI话题传播分析 — B站数据采集、ASR转写、NLP分析全流程项目。

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ASR | FunASR + Paraformer-Large + VAD + CT-Punc |
| NLP分词 | jieba + 自定义AI术语词典 |
| 主题建模 | BERTopic (主力) + gensim LDA (baseline) |
| NER/情感/立场 | 智谱 GLM-4-Flash API (免费) |
| 知识密度 | jieba词性标注 + AI术语词典 |
| 语义网络 | NetworkX + python-louvain |
| 可视化 | pyecharts + matplotlib + plotly |
| 数据采集 | bilibili-api-python + yt-dlp + ffmpeg |
| GPU | RTX 3050 Laptop 4GB (CUDA 12.3) |

## Directory Structure
```
D:\studycode\study0402\
├── CLAUDE.md              # Agent工作流指令
├── task.json              # 任务定义
├── progress.txt           # 进度日志
├── init.sh                # 初始化脚本
├── architecture.md        # 本文件
├── requirements.txt       # Python依赖
├── .gitignore             # 忽略敏感文件
├── src/
│   ├── config.py          # 配置（API Key、Cookie等）
│   ├── main.py            # 主入口
│   ├── collect/           # 数据采集模块
│   │   ├── bilibili_crawler.py   # B站爬虫（视频/评论/弹幕）
│   │   └── audio_downloader.py   # 音频下载（yt-dlp）
│   ├── asr/               # 语音识别模块
│   │   └── transcriber.py        # FunASR Paraformer转写
│   ├── preprocess/        # 文本预处理模块
│   │   ├── cleaner.py            # 数据清洗
│   │   ├── tokenizer.py          # 分词+去停用词
│   │   └── ai_terms.txt          # AI术语词典
│   ├── analysis/          # 分析建模模块
│   │   ├── topic_model.py        # BERTopic + LDA主题建模
│   │   ├── llm_annotator.py      # GLM API 标注（NER/情感/立场）
│   │   ├── knowledge_density.py  # 知识密度计算
│   │   └── semantic_network.py   # 语义网络构建
│   └── visualization/     # 可视化模块
│       ├── wordcloud_chart.py    # 词云图
│       ├── network_chart.py      # 网络图
│       └── trend_chart.py        # 趋势图+分布图
├── data/
│   ├── raw/               # 原始数据
│   ├── audio/             # 下载的音频文件
│   ├── processed/         # 清洗后数据
│   └── results/           # 分析结果
├── output/                # 可视化输出（HTML/图片）
└── logs/                  # 日志
```

## Data Flow
```
B站搜索API → 视频元数据列表 → yt-dlp下载音频 → FunASR ASR转写
     ↓                                                          ↓
  评论+弹幕采集                                          转写文本
     ↓                                                          ↓
     └──────────── 文本预处理（jieba分词+清洗）─────────────────┘
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
             BERTopic主题建模   GLM标注(NER/情感)  知识密度计算
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                           可视化输出 + 论文数据
```

## Environment Variables
```env
# 智谱 API（GLM-4-Flash 免费）
ZHIPU_API_KEY=your_key_here

# B站 Cookie（从浏览器F12获取）
BILIBILI_SESSDATA=your_sessdata
BILIBILI_BILI_JCT=your_bili_jct
BILIBILI_BUVID3=your_buvid3
BILIBILI_DEDEUSERID=your_dedeuserid
```

## Key Constraints
- GPU显存仅4GB，FunASR batch_size_s不超过300
- 智谱GLM-4-Flash免费但有速率限制，需控制并发
- B站反爬：请求间隔1-3秒，每日采集量不超过2000次
- 参考文献：论文中需要写明每个工具的版本和引用
