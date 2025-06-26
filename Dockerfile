# Use Python 3.11 slim como base para menor tamanho
FROM python:3.11-slim

# Definir variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema (incluindo para EasyOCR e OpenCV)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libglib2.0-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libfontconfig1 \
    libice6 \
    libxt6 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements específicos do Docker
COPY requirements-docker.txt .

# Instalar dependências Python completas (incluindo IA/ML)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copiar código da aplicação
COPY . .

# Criar diretório temp
RUN mkdir -p temp

# Definir variáveis de ambiente para cache unificado (compatível com volumes)
ENV TORCH_HOME=/app/.cache/torch
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV EASYOCR_MODULE_PATH=/app/.cache/easyocr
ENV EASYOCR_DOWNLOAD_PATH=/app/.cache/easyocr

# Criar diretórios de cache
RUN mkdir -p /app/.cache && \
    mkdir -p $TORCH_HOME $TRANSFORMERS_CACHE $HF_HOME $HF_DATASETS_CACHE $EASYOCR_MODULE_PATH

# Pre-carregar modelos para otimizar inicialização
# TEMPORARIAMENTE DESABILITADO para debugging - modelos serão baixados no runtime
# RUN echo "🔄 Baixando modelos EasyOCR..." && \
#     python -c "import easyocr; reader = easyocr.Reader(['pt', 'en'], gpu=False); print('✅ EasyOCR modelos baixados')" || echo "❌ Falha no download EasyOCR"

# RUN echo "🔄 Baixando modelos TrOCR..." && \
#     python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten'); print('✅ TrOCR modelos baixados')" || echo "❌ Falha no download TrOCR"

# Expor porta
EXPOSE 8000

# Comando de saúde
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 