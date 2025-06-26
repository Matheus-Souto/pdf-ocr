# 🚨 PROBLEMA: TIMEOUT EM PDFS GRANDES

## 📋 **Diagnóstico**

**Situação**: PDF de 44 páginas processa até página 9 e reinicia

### **Causas Identificadas:**

1. ⏱️ **Timeout do Proxy Reverso** (Nginx/Traefik do Easypanel)
2. ⏱️ **Timeout do FastAPI/Uvicorn** (padrão ~30s)
3. 💾 **Acúmulo de memória** ao processar múltiplas páginas
4. 🔗 **Keep-alive insuficiente** para processos longos

## 🛠️ **Soluções Implementadas**

### **1. Timeout do Uvicorn Aumentado**

```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    timeout_keep_alive=300,  # 5 minutos
    timeout_graceful_shutdown=30
)
```

### **2. Headers HTTP para Timeout Longo**

```python
headers={
    "X-Accel-Buffering": "no",  # Nginx: desabilitar buffering
    "Keep-Alive": "timeout=600, max=1000"  # Keep-alive por 10 minutos
}
```

### **3. Garbage Collection Agressivo**

```python
# Limpeza após cada página
gc.collect()

# Limpeza intensiva a cada 5 páginas
if (page_num + 1) % 5 == 0:
    gc.collect()
    gc.collect()  # Dupla limpeza
```

### **4. Silenciar Warnings do PyTorch**

```python
warnings.filterwarnings("ignore", message=".*NNPACK.*")
```

## ⚙️ **Configurações Recomendadas no Easypanel**

### **Nginx/Traefik Settings:**

```nginx
proxy_read_timeout 600s;
proxy_connect_timeout 600s;
proxy_send_timeout 600s;
```

### **Container Resources:**

- **CPU**: 2-4 cores
- **RAM**: 4-8GB (para EasyOCR + PDFs grandes)
- **Timeout**: 10+ minutos

### **Volume para Cache:**

- **Nome**: `pdf-ocr-cache`
- **Caminho**: `/app/.cache`
- **Tamanho**: 5GB

## 🎯 **Resultados Esperados**

✅ **Processar PDFs de 50+ páginas** sem timeout
✅ **Memória estável** com garbage collection
✅ **Cache persistente** de modelos IA
✅ **Fallback robusto** para Tesseract se necessário

## 🔧 **Debugging de Timeouts**

1. **Verificar logs** do container para crashes
2. **Monitorar RAM** durante processamento
3. **Testar com PDFs menores** (5-10 páginas) primeiro
4. **Confirmar cache** de modelos funcionando

## 🚀 **Deploy Atualizado**

```bash
# Rebuild com otimizações
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verificar logs
docker-compose logs -f
```

## 📊 **Métricas de Performance**

| Páginas | Tempo Esperado | RAM Máxima |
| ------- | -------------- | ---------- |
| 1-10    | 30s-2min       | 2GB        |
| 11-25   | 2-5min         | 3GB        |
| 26-50   | 5-10min        | 4GB        |
| 51+     | 10-20min       | 6GB+       |

**Nota**: Tempos podem variar dependendo da complexidade do conteúdo e resolução das páginas.
