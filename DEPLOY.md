# 🐳 Guia de Deploy - EasyPanel com Docker

Este guia mostra como fazer deploy da API de OCR para PDFs usando **EasyPanel** via **GitHub**.

## 📋 Pré-requisitos

- ✅ Conta no **EasyPanel**
- ✅ Repositório no **GitHub** (público ou privado)
- ✅ Projeto configurado com Docker

## 🚀 Passos para Deploy

### 1. **Preparar Repositório GitHub**

Certifique-se de que seu repositório contém:

```
pdf-ocr/
├── Dockerfile              # ✅ Criado
├── docker-compose.yml      # ✅ Criado
├── .dockerignore           # ✅ Criado
├── requirements.txt        # ✅ Atualizado (sem python-magic-bin)
├── main.py                 # ✅ API principal
├── config.py               # ✅ Configurações
└── temp/                   # ✅ Diretório temporário
```

### 2. **Configurar no EasyPanel**

#### 🔗 **Conectar GitHub:**

1. Acesse seu painel EasyPanel
2. Clique em **"Create New App"**
3. Selecione **"From GitHub Repository"**
4. Autorize acesso ao seu repositório
5. Selecione o repositório `pdf-ocr`

#### ⚙️ **Configurações da Aplicação:**

```yaml
# Configurações recomendadas
Name: pdf-ocr-api
Type: Web Service
Build Method: Dockerfile
Port: 8000
Health Check: /health
```

#### 🔧 **Variáveis de Ambiente:**

```bash
PYTHONPATH=/app
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

#### 📦 **Recursos Recomendados:**

```yaml
CPU: 1-2 vCPUs
RAM: 2-4 GB (OCR requer memória)
Storage: 10-20 GB
```

### 3. **Configurações de Build**

O EasyPanel detectará automaticamente o `Dockerfile` e usará as seguintes configurações:

```dockerfile
# Build Context: ./
# Build Args: Nenhum necessário
# Expose Port: 8000
# Health Check: curl -f http://localhost:8000/health
```

### 4. **Deploy Automático**

#### 🔄 **Auto-Deploy via GitHub:**

- ✅ **Push para main/master** → Deploy automático
- ✅ **Pull Requests** → Preview deployments (opcional)
- ✅ **Rollback** → Reversão para versão anterior

### 5. **Monitoramento**

#### 📊 **Health Checks:**

```bash
# Endpoint de saúde
GET https://sua-app.easypanel.app/health

# Resposta esperada:
{
    "status": "OK",
    "message": "API funcionando corretamente"
}
```

#### 📈 **Logs de Aplicação:**

- Acesse **Logs** no painel EasyPanel
- Monitore performance de OCR
- Verifique uso de memória

### 6. **Teste da API Deployada**

#### 🌐 **Swagger UI:**

```
https://sua-app.easypanel.app/docs
```

#### 🧪 **Teste rápido via curl:**

```bash
# Health check
curl https://sua-app.easypanel.app/health

# Upload de PDF para OCR
curl -X POST "https://sua-app.easypanel.app/convert-pdf/" \
     -H "accept: application/pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@exemplo.pdf" \
     --output "resultado_ocr.pdf"
```

## 🔧 Troubleshooting

### ❌ **Problemas Comuns:**

#### **Build falha - Dependências não encontradas:**

```bash
# ✅ CORRIGIDO: Removido python-magic-bin (Windows only)
# Agora usa apenas dependências multiplataforma:
- fastapi, uvicorn, pytesseract
- PyMuPDF, Pillow, requests, httpx
```

#### **Build falha - Tesseract não encontrado:**

```bash
# Solução: Verificar se o Dockerfile tem:
RUN apt-get install tesseract-ocr tesseract-ocr-por
```

#### **Container usa muita memória:**

```yaml
# Ajustar recursos no EasyPanel:
Memory Limit: 4GB
CPU Limit: 2 cores
```

#### **Upload de arquivos grandes falha:**

```python
# Adicionar no main.py:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)
```

#### **Health check falha:**

```bash
# Verificar se curl está instalado no container
# Verificar se porta 8000 está exposta
# Verificar se /health endpoint responde
```

## ✅ **Dependências Otimizadas para Docker:**

```txt
# requirements.txt (corrigido para Linux)
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pytesseract==0.3.10
PyMuPDF==1.23.8
Pillow==10.1.0
requests==2.32.4
httpx==0.25.2
```

**🔥 Principais correções aplicadas:**

- ❌ Removido `python-magic-bin` (específico Windows)
- ✅ Mantidas apenas dependências multiplataforma
- ✅ Dockerfile otimizado para Linux containers
- ✅ Build mais rápido e confiável

## 🏗️ **Estrutura de Deploy**

```
EasyPanel Dashboard
├── 📱 Apps
│   └── pdf-ocr-api
│       ├── ⚙️ Settings
│       ├── 📊 Metrics
│       ├── 📝 Logs
│       ├── 🔄 Deployments
│       └── 🌐 Domains
├── 🗄️ Databases (se necessário)
└── 📁 File Storage (opcional)
```

## 🎯 **Otimizações Recomendadas**

### **Para Produção:**

- ✅ Configure **domínio customizado**
- ✅ Ative **SSL/HTTPS automático**
- ✅ Configure **backups automáticos**
- ✅ Monitore **uso de recursos**
- ✅ Configure **alertas** de erro
- ✅ Implemente **rate limiting**

### **Para Performance:**

- ✅ Use **cache** para PDFs processados
- ✅ Configure **load balancing** se necessário
- ✅ Otimize **resolução de imagens**
- ✅ Implemente **processamento assíncrono**

## 📞 **Suporte**

- 📧 **EasyPanel:** Suporte via painel
- 🐛 **Issues:** GitHub Issues do projeto
- 📚 **Docs:** Documentação da API em `/docs`

---

**🎉 Pronto! Sua API de OCR está rodando em produção com deploy automático via GitHub!**
