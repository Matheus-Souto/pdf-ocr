from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import tempfile
import uuid
from typing import List
import uvicorn

app = FastAPI(
    title="PDF OCR API",
    description="API para converter PDFs não pesquisáveis em PDFs pesquisáveis usando OCR",
    version="1.0.0"
)

# Configurar o caminho do Tesseract (ajuste conforme sua instalação)
# Para Windows, descomente e ajuste o caminho:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Criar diretório temporário se não existir
os.makedirs("temp", exist_ok=True)

def cleanup_temp_file(file_path: str):
    """
    Função síncrona para deletar arquivo temporário após enviar resposta
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Arquivo temporário removido: {file_path}")
    except Exception as e:
        print(f"Erro ao remover arquivo temporário {file_path}: {e}")

@app.get("/")
async def read_root():
    return {"message": "API de OCR para PDFs - Envie um PDF não pesquisável e receba um PDF pesquisável!"}

@app.post("/convert-pdf/")
async def convert_pdf_to_searchable(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Converte um PDF não pesquisável (com imagens) em um PDF pesquisável usando OCR.
    """
    
    # Validar tipo do arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    try:
        # Ler o arquivo PDF
        pdf_content = await file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Criar um novo PDF para armazenar o resultado
        output_pdf = fitz.open()
        
        # Processar cada página
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Converter página para imagem
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Aumenta resolução
            img_data = pix.tobytes("png")
            
            # Converter para PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Extrair texto usando OCR
            texto_ocr = pytesseract.image_to_string(image, lang='por')
            
            # Criar nova página no PDF de saída
            nova_pagina = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
            
            # Inserir a imagem original como fundo
            nova_pagina.insert_image(nova_pagina.rect, stream=img_data)
            
            # Adicionar texto invisível para permitir pesquisa
            if texto_ocr.strip():
                # Inserir texto invisível sobre a imagem
                # O texto será invisível mas pesquisável
                text_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                nova_pagina.insert_textbox(
                    text_rect,
                    texto_ocr,
                    fontsize=8,
                    color=(1, 1, 1),  # Texto branco (invisível)
                    overlay=False
                )
        
        # Salvar PDF convertido em arquivo temporário
        temp_id = str(uuid.uuid4())
        output_path = os.path.join("temp", f"output_{temp_id}.pdf")
        output_pdf.save(output_path)
        
        # Fechar documentos
        pdf_document.close()
        output_pdf.close()
        
        # Adicionar tarefa em background para limpar arquivo temporário
        background_tasks.add_task(cleanup_temp_file, output_path)
        
        # Retornar o arquivo convertido
        return FileResponse(
            path=output_path,
            filename=f"ocr_{file.filename}",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar PDF: {str(e)}")

@app.post("/extract-text/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    """
    Extrai apenas o texto de um PDF não pesquisável usando OCR.
    """
    
    # Validar tipo do arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    try:
        # Ler o arquivo PDF
        pdf_content = await file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        extracted_text = []
        
        # Processar cada página
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Converter página para imagem
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Converter para PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Extrair texto usando OCR
            texto_ocr = pytesseract.image_to_string(image, lang='por')
            
            extracted_text.append({
                "pagina": page_num + 1,
                "texto": texto_ocr.strip()
            })
        
        pdf_document.close()
        
        return {
            "filename": file.filename,
            "total_paginas": len(extracted_text),
            "texto_extraido": extracted_text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar se a API está funcionando
    """
    return {"status": "OK", "message": "API funcionando corretamente"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 