# 1. 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- 2. 攻克 NVIDIA 依赖（第一层：CUDNN） ---
RUN pip install --no-cache-dir --upgrade --retries 50 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=2000 \
    nvidia-cudnn-cu12==9.1.0.70

# --- 3. 攻克 NVIDIA 依赖（第二层：CUBLAS） ---
RUN pip install --no-cache-dir --upgrade --retries 50 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=2000 \
    nvidia-cublas-cu12==12.1.3.1

# --- 4. 【关键改动】让 Torch 自己决定配套依赖，防止冲突 ---
RUN pip install --no-cache-dir --retries 100 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=5000 \
    torch==2.1.0


# --- 5. 安装剩余的第三方库 ---
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 50 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=2000 \
    -r requirements.txt

# --- 6. 复制项目代码 ---
COPY . .

# 启动命令
CMD ["streamlit", "run", "main.py"]



