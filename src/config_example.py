"""
项目配置文件模板 - 复制为 config.py 并填入真实值
"""
import os

# ============================================================
# 智谱 GLM-4-Flash API
# ============================================================
ZHIPU_API_KEY = "your_api_key_here"
ZHIPU_API_KEYS = [
    "your_api_key_1",
    "your_api_key_2",
]
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_MODEL = "glm-4-flash"

# ============================================================
# B站 Cookie（从浏览器F12获取）
# ============================================================
BILIBILI_SESSDATA = "your_sessdata_here"
BILIBILI_BILI_JCT = "your_bili_jct_here"
BILIBILI_BUVID3 = "your_buvid3_here"
BILIBILI_DEDEUSERID = "your_dedeuserid_here"

# ============================================================
# 数据采集参数
# ============================================================
NEWS_KEYWORDS = ["AI新闻", "AI快报", "ChatGPT发布"]
INTERVIEW_KEYWORDS = ["Sam Altman访谈", "黄仁勋访谈"]
SEARCH_KEYWORDS = NEWS_KEYWORDS + INTERVIEW_KEYWORDS

TITLE_BLACKLIST = ["教程", "卖课", "零基础"]
TITLE_BONUS_WORDS = ["访谈", "发布", "深度"]

TIME_RANGE_START = "2022-11-01"
TIME_RANGE_END = "2026-04-01"
MIN_PLAY_COUNT = 5000
MIN_DURATION_SEC = 120
INTERVIEW_MIN_DURATION_SEC = 300
MAX_VIDEOS = 800
TARGET_COUNT_PER_CATEGORY = 400
QUALITY_SCORE_THRESHOLD = 50

RECOMMENDED_UP_MIDS = []  # 推荐UP主MID列表

REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0
BATCH_SIZE = 100

# ============================================================
# ASR 参数
# ============================================================
ASR_MODEL = "paraformer-zh"
VAD_MODEL = "fsmn-vad"
PUNC_MODEL = "ct-punc"
ASR_DEVICE = "cpu"
ASR_BATCH_SIZE_S = 300

# ============================================================
# LLM标注参数
# ============================================================
LLM_BATCH_SIZE = 20
LLM_SLEEP_INTERVAL = 0.5

# ============================================================
# 主题建模参数
# ============================================================
BERTOPIC_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
BERTOPIC_MIN_CLUSTER_SIZE = 15
BERTOPIC_NR_BINS = 20

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for d in [RAW_DIR, AUDIO_DIR, PROCESSED_DIR, RESULTS_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
