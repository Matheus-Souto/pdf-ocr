# 🐳 Deploy Docker - API OCR Híbrida com EasyOCR

Este guia mostra como fazer deploy da API OCR com **todas as funcionalidades avançadas** incluindo EasyOCR, TrOCR e consensus de múltiplos engines.

## 🚀 **Funcionalidades Incluídas no Docker:**

- ✅ **Tesseract OCR** (básico)
- ✅ **EasyOCR** (rede neural)
- ✅ **TrOCR** (transformer-based)
- ✅ **Endpoint híbrido** `/extract-text-hybrid/`
- ✅ **Seleção de engine** (auto, tesseract, easyocr, trocr, consensus)
- ✅ **Layout detection** inteligente
- ✅ **Perspective correction** automática
- ✅ **Poppler** pré-instalado

## 📋 **Pré-requisitos:**

- Docker instalado
- 4GB+ de RAM (recomendado para IA/ML)
- 2GB+ de espaço em disco

## 🛠️ **Build e Execução:**

### 1. Build da Imagem

```bash
# Fazer build da imagem Docker
docker build -t pdf-ocr-hybrid .
```

### 2. Executar Container

```bash
# Executar com mapeamento de porta
docker run -d \
  -p 8000:8000 \
  --name pdf-ocr-api \
  pdf-ocr-hybrid
```

### 3. Verificar Status

```bash
# Verificar se está rodando
docker ps

# Ver logs
docker logs pdf-ocr-api

# Testar health check
curl http://localhost:8000/health
```

## 🧪 **Testando o Endpoint Híbrido:**

### Via Swagger UI

1. Acesse: `http://localhost:8000/docs`
2. Vá para `/extract-text-hybrid/`
3. Configure os parâmetros:
   - **file**: Seu PDF
   - **enhancement_level**: `ultra`
   - **use_ai_engines**: `true`
   - **engine_preference**: `auto`

### Via cURL

```bash
curl -X POST "http://localhost:8000/extract-text-hybrid/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@documento.pdf" \
  -F "enhancement_level=ultra" \
  -F "use_ai_engines=true" \
  -F "engine_preference=auto"
```

### Via Python

```python
import requests

url = "http://localhost:8000/extract-text-hybrid/"
files = {"file": open("documento.pdf", "rb")}
data = {
    "enhancement_level": "ultra",
    "use_ai_engines": True,
    "engine_preference": "auto"
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Páginas processadas: {result['total_paginas']}")
print(f"Engines utilizados: {result['estatisticas_globais']['engines_utilizados']}")
```

## ⚙️ **Configurações de Engine:**

### `engine_preference` Options:

- **`auto`**: Consenso automático entre todos os engines (padrão)
- **`tesseract`**: Apenas Tesseract (mais rápido)
- **`easyocr`**: Apenas EasyOCR (melhor para textos manuscritos)
- **`trocr`**: Apenas TrOCR (melhor para documentos históricos)
- **`consensus`**: Força análise de consenso

### `enhancement_level` Options:

- **`conservative`**: Processamento mínimo
- **`medium`**: Balanceado (padrão)
- **`aggressive`**: Processamento intensivo
- **`ultra`**: Máxima qualidade

## 📊 **Monitoramento:**

### Health Check

```bash
# Verificar status da API
curl http://localhost:8000/health
```

### Logs em Tempo Real

```bash
# Acompanhar logs
docker logs -f pdf-ocr-api
```

### Recursos do Container

```bash
# Ver uso de recursos
docker stats pdf-ocr-api
```

## 🚀 **Deploy em Produção:**

### Docker Compose

```yaml
version: '3.8'
services:
  pdf-ocr-api:
    build: .
    ports:
      - '8000:8000'
    environment:
      - TORCH_HOME=/app/.torch
      - TRANSFORMERS_CACHE=/app/.transformers_cache
    volumes:
      - ./temp:/app/temp
    restart: unless-stopped
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:8000/health']
      interval: 30s
      timeout: 10s
      retries: 3
```

### Variáveis de Ambiente

```bash
# Otimizações para produção
docker run -d \
  -p 8000:8000 \
  -e TORCH_HOME=/app/.torch \
  -e TRANSFORMERS_CACHE=/app/.transformers_cache \
  -e EASYOCR_MODULE_PATH=/app/.easyocr \
  --name pdf-ocr-api \
  pdf-ocr-hybrid
```

## 📈 **Performance:**

### Tamanho da Imagem

- **Base**: ~1.5GB (apenas Tesseract)
- **Híbrida**: ~3.5GB (com EasyOCR + TrOCR)

### Tempo de Inicialização

- **Primeira inicialização**: 30-60s (download de modelos)
- **Inicializações seguintes**: 10-15s

### Performance por Engine

- **Tesseract**: Mais rápido (~2-5s por página)
- **EasyOCR**: Moderado (~5-10s por página)
- **TrOCR**: Mais lento (~10-20s por página)
- **Consensus**: Mais lento, maior precisão

## 🔧 **Otimizações:**

### Pre-download de Modelos (Opcional)

Descomente as linhas no Dockerfile para pre-baixar modelos:

```dockerfile
RUN python -c "import easyocr; easyocr.Reader(['pt', 'en'], gpu=False)"
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')"
```

**⚠️ Aviso**: Isso aumentará a imagem para ~5GB.

### Volume para Cache

```bash
docker run -d \
  -p 8000:8000 \
  -v ocr_cache:/app/.torch \
  -v ocr_cache:/app/.transformers_cache \
  --name pdf-ocr-api \
  pdf-ocr-hybrid
```

## 🆘 **Troubleshooting:**

### Container não Inicia

```bash
# Ver logs de erro
docker logs pdf-ocr-api

# Verificar recursos
docker system df
free -h
```

### Modelos não Baixam

```bash
# Executar manualmente
docker exec -it pdf-ocr-api python -c "import easyocr; easyocr.Reader(['pt', 'en'], gpu=False)"
```

### Performance Lenta

- Aumentar RAM disponível para Docker
- Usar SSD para armazenamento
- Considerar apenas Tesseract para volume alto

## ✅ **Resultado Final:**

- **API Híbrida** com 4 engines de OCR
- **Auto-seleção** do melhor resultado
- **JSON estruturado** página por página
- **Deploy simples** com Docker
- **Pronto para produção**

🎉 **Sua API OCR híbrida está pronta para extrair texto com máxima precisão!**
