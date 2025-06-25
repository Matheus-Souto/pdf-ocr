# API de OCR para PDFs

Esta API FastAPI converte PDFs não pesquisáveis (que contêm apenas imagens) em PDFs pesquisáveis usando tecnologia OCR (Optical Character Recognition).

## 🚀 Funcionalidades

- **Conversão PDF para PDF pesquisável**: Converte PDFs com imagens em PDFs onde o texto pode ser pesquisado e selecionado
- **Extração de texto**: Extrai todo o texto de PDFs não pesquisáveis usando OCR
- **Suporte ao português**: Configurado para reconhecimento de texto em português
- **API RESTful**: Interface simples e intuitiva

## 📋 Pré-requisitos

### 1. Python 3.8+

Certifique-se de ter Python 3.8 ou superior instalado.

### 2. Tesseract OCR

Você precisa instalar o Tesseract OCR no seu sistema:

**Windows:**

1. Baixe o instalador em: https://github.com/UB-Mannheim/tesseract/wiki
2. Instale seguindo as instruções
3. Adicione o Tesseract ao PATH do sistema ou descomente e ajuste a linha no `main.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-por
```

**macOS:**

```bash
brew install tesseract tesseract-lang
```

## 🛠️ Instalação (Recomendado - com Ambiente Virtual)

### Opção 1: Configuração Automática com Ambiente Virtual

1. **Execute o script de configuração:**

```bash
python setup_venv.py
```

2. **Ative o ambiente virtual:**

**Windows (PowerShell):**

```powershell
.\ativar_venv.ps1
```

**Windows (CMD):**

```cmd
ativar_venv.cmd
```

**Linux/macOS:**

```bash
./ativar_venv.sh
```

### Opção 2: Configuração Manual com Ambiente Virtual

1. **Criar ambiente virtual:**

```bash
python -m venv venv
```

2. **Ativar o ambiente virtual:**

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

3. **Instalar dependências:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Opção 3: Instalação Direta (sem ambiente virtual)

1. Clone ou baixe este projeto
2. Navegue até o diretório do projeto
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Como executar

### Com Ambiente Virtual (Recomendado)

1. **Ative o ambiente virtual** (se ainda não estiver ativo):

   - Windows PowerShell: `.\ativar_venv.ps1`
   - Windows CMD: `ativar_venv.cmd`
   - Linux/macOS: `./ativar_venv.sh`

2. **Execute a API:**

```bash
python main.py
```

### Sem Ambiente Virtual

```bash
python main.py
```

A API estará disponível em: `http://localhost:8000`

## 📖 Documentação da API

### Endpoints disponíveis:

#### 1. Página inicial

- **GET** `/`
- Retorna informações básicas sobre a API

#### 2. Converter PDF para pesquisável

- **POST** `/convert-pdf/`
- **Parâmetros**: Arquivo PDF (form-data)
- **Resposta**: PDF convertido para download
- **Descrição**: Converte um PDF não pesquisável em um PDF pesquisável

#### 3. Extrair texto

- **POST** `/extract-text/`
- **Parâmetros**: Arquivo PDF (form-data)
- **Resposta**: JSON com texto extraído de cada página
- **Descrição**: Extrai apenas o texto do PDF sem gerar novo arquivo

#### 4. Health check

- **GET** `/health`
- **Resposta**: Status da API

## 🔧 Como usar

### 1. Via interface automática (Swagger)

Acesse `http://localhost:8000/docs` para uma interface interativa da API.

### 2. Via curl

**Converter PDF:**

```bash
curl -X POST "http://localhost:8000/convert-pdf/" \
     -H "accept: application/pdf" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@seu_arquivo.pdf" \
     --output "arquivo_convertido.pdf"
```

**Extrair texto:**

```bash
curl -X POST "http://localhost:8000/extract-text/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@seu_arquivo.pdf"
```

### 3. Via Python (requests)

```python
import requests

# Converter PDF
with open('seu_arquivo.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/convert-pdf/', files=files)

    if response.status_code == 200:
        with open('arquivo_convertido.pdf', 'wb') as output:
            output.write(response.content)

# Extrair texto
with open('seu_arquivo.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/extract-text/', files=files)

    if response.status_code == 200:
        resultado = response.json()
        print(resultado)
```

### 4. Via cliente de teste incluído

**Com ambiente virtual ativo:**

```bash
python test_client.py --health
python test_client.py --convert meu_arquivo.pdf
python test_client.py --extract meu_arquivo.pdf --save-json
```

## 🐍 Gerenciamento do Ambiente Virtual

### Comandos Úteis:

**Verificar se o ambiente virtual está ativo:**

- No prompt/terminal, você verá `(venv)` no início da linha

**Desativar o ambiente virtual:**

```bash
deactivate
```

**Reativar o ambiente virtual:**

- Windows PowerShell: `.\ativar_venv.ps1`
- Windows CMD: `ativar_venv.cmd`
- Linux/macOS: `./ativar_venv.sh`

**Instalar novas dependências:**

```bash
pip install nome_do_pacote
pip freeze > requirements.txt  # Atualizar arquivo de requisitos
```

## ⚙️ Configurações

### Ajustar qualidade do OCR

No arquivo `main.py`, você pode ajustar a matriz de resolução para melhorar a qualidade do OCR:

```python
# Linha 45 e 105
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Aumenta resolução (2x)
```

Valores maiores = melhor qualidade mas processamento mais lento.

### Idioma do OCR

Para alterar o idioma de reconhecimento, modifique a linha:

```python
# Linha 51 e 111
texto_ocr = pytesseract.image_to_string(image, lang='por')  # 'por' = português
```

Outros idiomas disponíveis: 'eng' (inglês), 'spa' (espanhol), etc.

## 🐛 Solução de problemas

### Erro: TesseractNotFoundError

- Certifique-se de que o Tesseract está instalado
- Configure o caminho correto no `main.py`
- Adicione o Tesseract ao PATH do sistema

### Erro de memória com PDFs grandes

- Reduza a matriz de resolução de `Matrix(2, 2)` para `Matrix(1.5, 1.5)`
- Processe PDFs em lotes menores

### Qualidade do OCR baixa

- Aumente a resolução: `Matrix(3, 3)` ou maior
- Certifique-se de que as imagens no PDF têm boa qualidade
- Considere pré-processar as imagens (contraste, brilho, etc.)

### Problemas com ambiente virtual

- Certifique-se de que o ambiente virtual está ativo antes de executar comandos
- Re-execute `python setup_venv.py` se houver problemas na configuração
- Use `deactivate` e reative o ambiente virtual se houver conflitos

## 📝 Notas importantes

- **Use sempre o ambiente virtual** para evitar conflitos de dependências
- A API processa PDFs página por página
- Arquivos temporários são automaticamente deletados após o processamento
- O texto é inserido de forma invisível sobre as imagens originais
- A API está otimizada para texto em português

## 🤝 Contribuições

Sinta-se à vontade para contribuir com melhorias, correções de bugs ou novas funcionalidades!
