# 🚀 Guia Rápido - API de OCR para PDFs

## ⚡ Instalação em 3 passos (Windows)

### 1. 📥 Instalar Tesseract OCR

```powershell
# Baixe e instale em: https://github.com/UB-Mannheim/tesseract/wiki
# OU use o Chocolatey (se tiver):
choco install tesseract
```

### 2. 🔧 Configurar projeto

```powershell
# Executar o script de configuração automática
python setup_venv.py
```

### 3. 🚀 Iniciar API

```powershell
# Ativar ambiente virtual e executar
.\ativar_venv.ps1
python main.py
```

**✅ Pronto! API rodando em: http://localhost:8000**

---

## ⚡ Instalação em 3 passos (Linux/macOS)

### 1. 📥 Instalar Tesseract OCR

```bash
# Ubuntu/Debian:
sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-por

# macOS:
brew install tesseract tesseract-lang
```

### 2. 🔧 Configurar projeto

```bash
# Executar o script de configuração automática
python setup_venv.py
```

### 3. 🚀 Iniciar API

```bash
# Ativar ambiente virtual e executar
./ativar_venv.sh
python main.py
```

**✅ Pronto! API rodando em: http://localhost:8000**

---

## 🧪 Testar a API

### Via Browser (Mais fácil)

1. Acesse: http://localhost:8000/docs
2. Clique em "Try it out" no endpoint `/convert-pdf/`
3. Faça upload do seu PDF
4. Clique "Execute"
5. Baixe o PDF convertido

### Via Linha de Comando

```bash
# Ativar ambiente virtual primeiro (se não estiver ativo)
# Windows: .\ativar_venv.ps1
# Linux/macOS: ./ativar_venv.sh

# Testar se a API está funcionando
python test_client.py --health

# Converter um PDF
python test_client.py --convert meu_arquivo.pdf

# Extrair texto apenas
python test_client.py --extract meu_arquivo.pdf --save-json
```

---

## 🆘 Problemas Comuns

### ❌ "TesseractNotFoundError"

**Solução:**

1. Certifique-se de que o Tesseract está instalado
2. No Windows, descomente esta linha no `main.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### ❌ "No module named 'fastapi'"

**Solução:**

1. Certifique-se de que o ambiente virtual está ativo
2. Reexecute: `python setup_venv.py`

### ❌ API não inicia

**Solução:**

1. Verifique se a porta 8000 está livre
2. Ou altere a porta no `main.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`

### ❌ Erro "TypeError: 'coroutine' object is not callable"

**✅ CORRIGIDO!** Este erro do BackgroundTask foi resolvido na versão atual.

**Correção aplicada:**

- Substituído `BackgroundTask` por `BackgroundTasks`
- Uso correto de `background_tasks.add_task()`
- Cleanup automático de arquivos temporários funcionando

---

## 📁 Estrutura do Projeto

```
pdf-ocr/
├── main.py              # API principal
├── setup_venv.py        # Configuração automática
├── test_client.py       # Cliente de teste
├── requirements.txt     # Dependências
├── README.md           # Documentação completa
├── venv/               # Ambiente virtual (criado automaticamente)
├── temp/               # Arquivos temporários
├── ativar_venv.ps1     # Script ativação Windows PowerShell
├── ativar_venv.cmd     # Script ativação Windows CMD
└── ativar_venv.sh      # Script ativação Linux/macOS
```

---

## 💡 Dicas

- **Sempre use o ambiente virtual** para evitar conflitos
- **Verifique se tem `(venv)` no prompt** antes de executar comandos
- **Para PDFs grandes**: reduza a resolução no `main.py` (linha 45)
- **Para melhor qualidade OCR**: aumente a resolução no `main.py`
- **Interface web**: http://localhost:8000/docs é mais fácil para testes
- **✅ Erro corrigido**: O problema do BackgroundTask foi completamente resolvido

---

## 🔄 Comandos de Manutenção

```bash
# Desativar ambiente virtual
deactivate

# Reativar ambiente virtual
# Windows: .\ativar_venv.ps1
# Linux/macOS: ./ativar_venv.sh

# Atualizar dependências
pip install --upgrade -r requirements.txt

# Ver dependências instaladas
pip list

# Testar API rapidamente
python test_client.py --health
```

---

**📞 Precisa de ajuda?** Consulte o [README completo](README.md) ou abra uma issue!
