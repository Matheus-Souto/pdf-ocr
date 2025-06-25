from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import tempfile
import uuid
from typing import List
import uvicorn
import json
import asyncio

app = FastAPI(
    title="PDF OCR API",
    description="API para converter PDFs não pesquisáveis em PDFs pesquisáveis usando OCR",
    version="1.0.0"
)

# Montar arquivos estáticos (para servir exemplo_progress.html)
app.mount("/static", StaticFiles(directory="."), name="static")

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

@app.post("/extract-text-progress/")
async def extract_text_with_progress(file: UploadFile = File(...)):
    """
    Extrai texto de PDF com indicador de progresso em tempo real usando Server-Sent Events.
    """
    
    # Validar tipo do arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    async def generate_progress():
        try:
            # Ler o arquivo PDF
            pdf_content = await file.read()
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = pdf_document.page_count
            
            # Enviar informações iniciais
            initial_data = {
                "tipo": "info",
                "timestamp": str(asyncio.get_event_loop().time()),
                "arquivo": {
                    "nome": file.filename,
                    "tamanho_mb": round(len(pdf_content) / (1024 * 1024), 2)
                },
                "processamento": {
                    "total_paginas": total_pages,
                    "progresso_atual": 0,
                    "status": "Iniciando processamento"
                }
            }
            yield f"data: {json.dumps(initial_data, ensure_ascii=False)}\n\n"
            
            extracted_text = []
            total_characters = 0
            
            # Processar cada página
            for page_num in range(total_pages):
                # Calcular progresso
                progress_percent = int((page_num / total_pages) * 100)
                
                # Enviar progresso atual
                progress_data = {
                    "tipo": "progresso",
                    "timestamp": str(asyncio.get_event_loop().time()),
                    "processamento": {
                        "pagina_atual": page_num + 1,
                        "total_paginas": total_pages,
                        "progresso_percent": progress_percent,
                        "status": f"Processando página {page_num + 1} de {total_pages}",
                        "etapa": "ocr_em_andamento"
                    },
                    "estatisticas": {
                        "paginas_concluidas": page_num,
                        "caracteres_extraidos": total_characters,
                        "tempo_decorrido": round(asyncio.get_event_loop().time(), 2)
                    }
                }
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                
                page = pdf_document[page_num]
                
                # Converter página para imagem
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                # Converter para PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Extrair texto usando OCR
                texto_ocr = pytesseract.image_to_string(image, lang='por')
                page_text = texto_ocr.strip()
                
                extracted_text.append({
                    "pagina": page_num + 1,
                    "texto": page_text,
                    "estatisticas": {
                        "caracteres": len(page_text),
                        "linhas": len(page_text.split('\n')) if page_text else 0,
                        "palavras": len(page_text.split()) if page_text else 0
                    }
                })
                
                total_characters += len(page_text)
                
                # Enviar resultado da página processada
                page_completed_data = {
                    "tipo": "pagina_concluida",
                    "timestamp": str(asyncio.get_event_loop().time()),
                    "pagina": {
                        "numero": page_num + 1,
                        "processamento_completo": True
                    },
                    "resultado": {
                        "caracteres_pagina": len(page_text),
                        "linhas_pagina": len(page_text.split('\n')) if page_text else 0,
                        "palavras_pagina": len(page_text.split()) if page_text else 0,
                        "tem_texto": bool(page_text)
                    },
                    "estatisticas_gerais": {
                        "total_caracteres": total_characters,
                        "paginas_processadas": page_num + 1,
                        "progresso_percent": int(((page_num + 1) / total_pages) * 100)
                    }
                }
                yield f"data: {json.dumps(page_completed_data, ensure_ascii=False)}\n\n"
                
                # Pequena pausa para permitir que o cliente processe
                await asyncio.sleep(0.1)
            
            pdf_document.close()
            
            # Calcular estatísticas finais
            paginas_com_texto = sum(1 for item in extracted_text if item["texto"])
            total_palavras = sum(item["estatisticas"]["palavras"] for item in extracted_text)
            total_linhas = sum(item["estatisticas"]["linhas"] for item in extracted_text)
            
            # Enviar resultado final
            final_result = {
                "tipo": "concluido",
                "timestamp": str(asyncio.get_event_loop().time()),
                "arquivo": {
                    "nome": file.filename,
                    "tamanho_mb": round(len(pdf_content) / (1024 * 1024), 2)
                },
                "processamento": {
                    "status": "concluido_com_sucesso",
                    "total_paginas": len(extracted_text),
                    "progresso_percent": 100,
                    "tempo_total": round(asyncio.get_event_loop().time(), 2)
                },
                "estatisticas": {
                    "total_caracteres": total_characters,
                    "total_palavras": total_palavras,
                    "total_linhas": total_linhas,
                    "paginas_com_texto": paginas_com_texto,
                    "paginas_vazias": len(extracted_text) - paginas_com_texto,
                    "media_caracteres_por_pagina": round(total_characters / len(extracted_text), 2) if extracted_text else 0
                },
                "resultados": {
                    "texto_por_pagina": extracted_text,
                    "resumo_conteudo": {
                        "primeira_pagina_preview": extracted_text[0]["texto"][:200] + "..." if extracted_text and extracted_text[0]["texto"] else "Sem texto",
                        "pagina_mais_longa": max(extracted_text, key=lambda x: x["estatisticas"]["caracteres"])["pagina"] if extracted_text else None,
                        "pagina_mais_curta": min(extracted_text, key=lambda x: x["estatisticas"]["caracteres"])["pagina"] if extracted_text else None
                    }
                }
            }
            
            yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            # Enviar erro formatado
            error_data = {
                "tipo": "erro",
                "timestamp": str(asyncio.get_event_loop().time()),
                "erro": {
                    "mensagem": f"Erro ao extrair texto: {str(e)}",
                    "tipo_erro": type(e).__name__,
                    "detalhes": "Verifique se o arquivo PDF é válido e não está corrompido"
                },
                "processamento": {
                    "status": "falhou",
                    "progresso_percent": -1
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar se a API está funcionando
    """
    return {"status": "OK", "message": "API funcionando corretamente"}

@app.get("/exemplo_progress.html")
async def get_example_page():
    """
    Serve a página de exemplo para teste do endpoint de progresso.
    """
    return FileResponse("exemplo_progress.html", media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 