FROM python:3.11-slim

# 先裝依賴
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式
COPY . .

# 建一個有家目錄的使用者，並把 /app 與快取目錄權限交給他
RUN useradd -m -u 1000 appuser \
 && mkdir -p /home/appuser/.cache/huggingface \
 && chown -R appuser:appuser /home/appuser /app

# 指定家目錄與 HF 快取位置（之後 HF/transformers/datasets 都會寫到這）
ENV HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache \
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub \
    HF_DATASETS_CACHE=/home/appuser/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers

USER appuser


ENTRYPOINT ["python", "score_measure.py"]


CMD ["--task", "QA", "--model", "BERT"]