# 项目总结：基于文本挖掘的视频平台AI话题传播分析

## 一、项目概况

| 项目     | 内容                                                         |
| -------- | ------------------------------------------------------------ |
| 论文题目 | 《基于文本挖掘的视频平台AI话题传播分析》                     |
| 数据来源 | B站（哔哩哔哩）                                              |
| 时间范围 | 2022年11月 ~ 2026年3月                                       |
| 分析维度 | 三端：传播端（标题+描述）、接收端（评论+弹幕）、ASR音频转写端 |
| 项目路径 | `D:\studycode\study0402\`                                    |

---

## 二、数据规模

| 指标         | 数值                              |
| ------------ | --------------------------------- |
| 采集视频     | 800个（news:400 + interview:400） |
| 总播放量     | 1.26亿                            |
| 总评论       | 30,758条                          |
| 总弹幕       | 283,837条                         |
| ASR转写      | 693个视频（86.6%覆盖率）          |
| 音频文件     | 83GB（794个m4a）                  |
| 传播端词数   | 57,257                            |
| 接收端词数   | 1,219,874                         |
| ASR端词数    | 4,385,229                         |
| LLM标注      | 传播端800 + 接收端794 + ASR 671   |
| BERTopic主题 | 传播端5 + 接收端2 + ASR 12        |
| LDA主题      | 传播端k=5 + 接收端k=19 + ASR k=5  |

---

## 三、项目目录结构

```
D:\studycode\study0402\
├── src/                          # 源代码
│   ├── config.py                 # 全局配置（API Key、采集参数、路径）
│   ├── collect/
│   │   └── bilibili_crawler.py   # B站爬虫（六维质量评分+分类采集）
│   ├── asr/
│   │   ├── audio_downloader.py   # yt-dlp音频下载
│   │   └── transcriber.py        # FunASR Paraformer ASR转写（CPU模式）
│   ├── preprocess/
│   │   └── text_processor.py     # 文本清洗+jieba分词+三端独立分词
│   ├── analysis/
│   │   ├── topic_modeling.py     # BERTopic三套+LDA三套主题建模
│   │   ├── llm_annotator.py      # GLM-4-Flash 10线程并发NER+情感+立场标注
│   │   ├── knowledge_density.py  # 三端知识密度/词汇密度计算
│   │   ├── semantic_network.py   # 三端共现网络+Louvain社区发现
│   │   └── cross_analysis.py     # 三端对比分析+雷达图
│   └── visualization/
│       └── visualizer.py         # 论文图表生成（中英双语标题，5号黑体/宋体）
├── data/
│   ├── raw/                      # 原始采集数据（59MB）
│   │   └── bilibili_videos.json  # 800个视频元数据+评论+弹幕
│   ├── audio/                    # 音频文件（83GB）
│   │   └── *.m4a / *.wav         # 794个视频音频
│   ├── processed/                # 预处理数据（139MB）
│   │   ├── processed_videos.json # 三端分词结果
│   │   └── asr_results.json      # ASR转写文本
│   └── results/                  # 分析结果（1.5GB）
│       ├── annotations.json      # LLM三端标注（NER+情感+立场）
│       ├── topic_modeling_summary.json  # BERTopic+LDA汇总
│       ├── bertopic_*_summary.json     # 三套BERTopic主题详情
│       ├── bertopic_*_over_time.json   # 动态主题追踪
│       ├── lda_results.json      # LDA三端结果
│       ├── knowledge_density.json # 三端知识密度
│       ├── semantic_network.json  # 三端语义网络指标
│       ├── cross_analysis.json    # 三端对比分析
│       └── final_report.json      # 最终整合报告
├── output/                       # 论文图表（8.7MB）
│   ├── wordcloud_comparison.png       # 图1 三端词云对比
│   ├── knowledge_density_distribution.png  # 图2 三端知识密度分布
│   ├── video_type_distribution.png    # 图3 视频类型饼图
│   ├── sentiment_distribution.png     # 图4 三端情感/立场分布
│   ├── kd_comparison.png              # 图6 密度对比柱状图
│   ├── entity_frequency.png           # 图7 三端实体频率
│   ├── network_propagator.png         # 图8 传播端语义网络
│   ├── network_receiver.png           # 图9 接收端语义网络
│   ├── network_asr.png                # 图10 ASR端语义网络
│   ├── radar_comparison.png           # 图11 三端综合雷达图
│   └── project_report.md             # 完整技术报告
├── logs/                         # 运行日志（92MB）
├── run_all.py                    # 一键全流程Pipeline
├── run_bertopic_batch.py         # BERTopic分批运行（避免内存溢出）
├── run_incremental.py            # 增量更新Pipeline
├── run_pipeline.py               # 主控Pipeline
├── task.json                     # 任务清单
└── 关注的up.md                    # 39位推荐UP主列表
```

---

## 四、技术栈

| 技术                                             | 用途                               |
| ------------------------------------------------ | ---------------------------------- |
| bilibili-api-python                              | B站视频搜索、详情、评论、弹幕采集  |
| yt-dlp                                           | 视频音频下载                       |
| FunASR Paraformer-Large (CPU)                    | 语音转文字（944MB模型）            |
| imageio-ffmpeg                                   | m4a→wav音频格式转换                |
| jieba + 自定义AI词典(188词)                      | 中文分词                           |
| BERTopic + paraphrase-multilingual-MiniLM-L12-v2 | 主题建模（CPU，hf-mirror.com镜像） |
| Gensim LDA                                       | 主题建模baseline对照               |
| GLM-4-Flash × 5个API Key × 10线程                | NER+情感+立场并发标注              |
| NetworkX + Louvain                               | 语义共现网络+社区发现              |
| Matplotlib + WordCloud                           | 论文图表（中英双语，5号黑体/宋体） |

---

## 五、完整流程

```
Step 1  数据采集
        ├── 36个关键词搜索B站视频
        ├── 六维质量评分体系（时效性+来源可信度+相关性+深度+传播力+受众价值）
        ├── 黑名单过滤（教程/卖课）+ 推荐UP主加分（39位）
        ├── 分类采集：news 400 + interview 400
        └── 输出: bilibili_videos.json (800个视频+评论+弹幕)

Step 2  音频下载 + ASR转写
        ├── yt-dlp下载794个m4a音频（83GB）
        ├── imageio-ffmpeg转换为16kHz WAV
        ├── FunASR Paraformer CPU模式批量转写
        └── 输出: asr_results.json (693条转写文本，438万词)

Step 3  文本预处理
        ├── 三端分离：传播端(标题+描述) / 接收端(评论+弹幕) / ASR端(转写文本)
        ├── jieba分词 + 188个AI术语词典
        ├── 去停用词（通用+B站专属）+ 词性过滤
        └── 输出: processed_videos.json (三端独立分词)

Step 4  主题建模
        ├── BERTopic三套独立模型（传播端5主题 / 接收端2主题 / ASR 12主题）
        ├── topics_over_time动态主题追踪
        ├── LDA三套对照（传播端k=5 / 接收端k=19 / ASR k=5）
        ├── Coherence Score曲线选最优k
        └── 输出: topic_modeling_summary.json + bertopic_*_summary.json

Step 5  LLM标注
        ├── GLM-4-Flash多任务Prompt（NER+情感+立场一次完成）
        ├── 5个API Key轮询 + 10线程并发
        ├── ASR端3万字分段采样（开头1万+中间1万+结尾1万）
        ├── 7层JSON解析容错 + 缺失字段自动补全
        └── 输出: annotations.json (传播端800+接收端794+ASR 671)

Step 6  知识密度分析
        ├── 知识密度 = AI术语命中数 / 总词数
        ├── 词汇密度 = 实词数 / 总词数
        ├── 三端对比：传播端0.048 > 接收端0.015 > ASR 0.008
        └── 输出: knowledge_density.json

Step 7  语义网络
        ├── 三端独立共现网络（top80高频词，min共现2次）
        ├── Louvain社区发现
        ├── 度中心性 + 介数中心性
        ├── 标点/数字/URL过滤
        └── 输出: semantic_network.json + 3张网络图

Step 8  可视化
        ├── 10张论文图表（300dpi PNG）
        ├── 中英双语标题（5号黑体）+ 图中文字（5号宋体）
        └── 输出: output/*.png

Step 9  三端对比分析
        ├── 实体频率对比（传播端5413 / 接收端10013 / ASR 3430）
        ├── 情感分布对比（传播端58%正面 / 接收端48%正面）
        ├── 知识密度对比 + 雷达图
        └── 输出: cross_analysis.json + radar_comparison.png
```

---

## 六、核心发现

1. **信息衰减**：知识密度 传播端(4.8%) > 接收端(1.5%) > ASR(0.8%)，专业术语逐层稀释
2. **情感鸿沟**：传播端58%正面/10%负面，接收端48%正面/24%负面，用户对AI更谨慎
3. **ASR揭示深层内容**：ASR端发现12个BERTopic主题（最多），口述内容远比标题描述丰富
4. **关注点错位**：传播端聚焦OpenAI/Claude等国际产品，接收端更多提及华为/腾讯等国内企业
5. **BERTopic vs LDA**：BERTopic在短文本（弹幕评论）上表现更优，LDA在长文本上可解释性更强

---

## 七、运行方式

```bash
# 一键全流程
python run_all.py

# BERTopic分批（避免内存溢出）
python run_bertopic_batch.py all

# 增量更新（ASR完成后补跑）
python run_incremental.py

# 单独重跑某步骤
python -m src.analysis.topic_modeling
python -m src.analysis.llm_annotator
python -m src.visualization.visualizer
```

