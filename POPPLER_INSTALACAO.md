# 🔧 Guia de Instalação do Poppler para Windows

## Problema

Erro: `Unable to get page count. Is poppler installed and in PATH?`

## Solução Completa

### Passo 1: Download do Poppler

1. Acesse: https://github.com/oschwartz10612/poppler-windows/releases/
2. Baixe a versão mais recente: **Release 24.08.0-0** (última versão)
3. Baixe o arquivo: `poppler-24.08.0_x86_64.zip`

### Passo 2: Instalação

1. **Extrair o arquivo:**

   - Extraia para: `C:\poppler-24.08.0\`
   - Estrutura deve ficar: `C:\poppler-24.08.0\Library\bin\`, `C:\poppler-24.08.0\Library\lib\`, etc.

2. **Adicionar ao PATH:**
   - Pressione `Win + R`, digite `sysdm.cpl`
   - Clique em "Variáveis de Ambiente"
   - Em "Variáveis do Sistema", encontre "Path"
   - Clique em "Editar" > "Novo"
   - Adicione: `C:\poppler-24.08.0\Library\bin`
   - Clique "OK" em todas as janelas

### Passo 3: Verificação

Abra um **novo** Command Prompt e digite:

```bash
pdftoppm -h
```

Se aparecer a ajuda do comando, a instalação está correta!

### Passo 4: Reiniciar o Sistema

**IMPORTANTE:** Reinicie o PowerShell/Command Prompt ou o computador para as alterações do PATH terem efeito.

## Solução Alternativa: Usando Conda

Se você tem conda instalado:

```bash
conda install -c conda-forge poppler
```

## Solução de Código (Se o PATH não funcionar)

Adicione o caminho diretamente no código Python:

```python
from pdf2image import convert_from_path

# Especificar o caminho do poppler diretamente
poppler_path = r"C:\poppler-24.08.0\Library\bin"
images = convert_from_path(pdf_path, poppler_path=poppler_path)
```

## Verificação Final

Após a instalação, teste com:

```bash
python -c "from pdf2image import convert_from_path; print('Poppler funcionando!')"
```

## Links Úteis

- **Download oficial:** https://github.com/oschwartz10612/poppler-windows/releases/
- **Documentação:** https://pdf2image.readthedocs.io/en/latest/installation.html

---

**Nota:** Certifique-se de baixar a versão x86_64 para sistemas de 64 bits (padrão no Windows 10/11).
**IMPORTANTE:** O caminho correto é `C:\poppler-24.08.0\Library\bin` (com Library no meio).
