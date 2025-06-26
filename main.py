from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import os
import tempfile
import uuid
from typing import List, Tuple, Dict, Optional
import uvicorn
import json
import asyncio
import cv2
import numpy as np
import datetime
import time
import re
from scipy import ndimage
from skimage import morphology, filters, measure
from collections import Counter
import math
import shutil
from pdf2image import convert_from_path
from uuid import uuid4
from difflib import SequenceMatcher
import gc

# Imports opcionais de AI/ML
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch não disponível - funcionalidades TrOCR desabilitadas")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers não disponível - funcionalidades TrOCR desabilitadas")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR não disponível - funcionalidades EasyOCR desabilitadas")

app = FastAPI(
    title="PDF OCR API Avançada",
    description="API para converter PDFs não pesquisáveis em PDFs pesquisáveis usando OCR com máxima precisão e técnicas avançadas",
    version="2.0.0"
)

# Montar arquivos estáticos (para servir exemplo_progress.html)
app.mount("/static", StaticFiles(directory="."), name="static")

# Configurar o caminho do Tesseract (ajuste conforme sua instalação)
# Para Windows, descomente e ajuste o caminho:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Variáveis globais para engines (serão inicializadas no startup)
easyocr_reader = None
trocr_processor = None
trocr_model = None

# Flag para indicar se engines estão sendo carregadas
engines_loading = False

@app.on_event("startup")
async def startup_event():
    """Inicializa as engines de OCR apenas uma vez durante o startup."""
    global easyocr_reader, trocr_processor, trocr_model
    
    print("🚀 INICIANDO SISTEMA PDF OCR API...")
    print(f"🔧 EASYOCR_AVAILABLE: {EASYOCR_AVAILABLE}")
    print(f"🔧 TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
    print(f"🔧 TORCH_AVAILABLE: {TORCH_AVAILABLE}")
    
    # FORÇAR configuração de cache unificado
    print("🔧 FORÇANDO CONFIGURAÇÃO DE CACHE UNIFICADO...")
    os.environ['TORCH_HOME'] = '/app/.cache/torch'
    os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/transformers'
    os.environ['HF_HOME'] = '/app/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/app/.cache/huggingface/datasets'
    os.environ['EASYOCR_MODULE_PATH'] = '/app/.cache/easyocr'
    os.environ['EASYOCR_DOWNLOAD_PATH'] = '/app/.cache/easyocr'
    
    # Criar diretórios se não existirem
    cache_dirs = [
        '/app/.cache',
        '/app/.cache/torch',
        '/app/.cache/transformers', 
        '/app/.cache/huggingface',
        '/app/.cache/huggingface/datasets',
        '/app/.cache/easyocr'
    ]
    
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  ✅ Criado/verificado: {cache_dir}")
    
    # Mostrar configurações de cache APÓS correção
    print("📁 CONFIGURAÇÕES DE CACHE (CORRIGIDAS):")
    print(f"  🗂️ TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    print(f"  🗂️ TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
    print(f"  🗂️ HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"  🗂️ EASYOCR_MODULE_PATH: {os.environ.get('EASYOCR_MODULE_PATH')}")
    
    # Verificar estado dos diretórios
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            print(f"  📂 {cache_dir}: {len(files)} arquivos")
        else:
            print(f"  ❌ {cache_dir}: não existe")
    
    # Inicialização LAZY - não carregar engines no startup para evitar timeout
    print("⚡ Inicialização LAZY ativada - engines serão carregadas sob demanda")
    print("✅ API pronta para receber requisições!")
    
    # # Inicializar EasyOCR
    # if EASYOCR_AVAILABLE:
    #     try:
    #         print("⏳ Carregando EasyOCR...")
    #         easyocr_reader = easyocr.Reader(['pt', 'en'], gpu=False)  # Português e Inglês
    #         print("✅ EasyOCR inicializado com sucesso")
    #     except Exception as e:
    #         easyocr_reader = None
    #         print(f"❌ EasyOCR não pôde ser inicializado: {e}")
    # else:
    #     easyocr_reader = None
    #     print("⚠️ EasyOCR não disponível - pacote não instalado")

    # # Inicializar TrOCR
    # if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
    #     try:
    #         print("⏳ Carregando TrOCR...")
    #         trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    #         trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    #         print("✅ TrOCR inicializado com sucesso")
    #     except Exception as e:
    #         trocr_processor = None
    #         trocr_model = None
    #         print(f"❌ TrOCR não pôde ser inicializado: {e}")
    # else:
    #     trocr_processor = None
    #     trocr_model = None
    #     print("⚠️ TrOCR não disponível - pacotes PyTorch/Transformers não instalados")

# Criar diretório temporário se não existir
os.makedirs("temp", exist_ok=True)

def detect_text_orientation(image: np.ndarray) -> float:
    """
    Detecta a orientação do texto na imagem usando análise de gradientes.
    
    Returns:
        float: Ângulo de rotação necessário em graus
    """
    try:
        # Converter para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar filtro para realçar bordas de texto
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detectar linhas usando transformada de Hough
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for line in lines[:20]:  # Analisar apenas as primeiras 20 linhas
                # Corrigir o desempacotamento - cv2.HoughLines retorna array com shape (n, 1, 2)
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            
            if angles:
                # Usar o ângulo mais comum
                angle_counts = Counter([round(a, 1) for a in angles])
                most_common_angle = angle_counts.most_common(1)[0][0]
                return most_common_angle
    
    except Exception as e:
        print(f"Erro na detecção de orientação: {e}")
    
    return 0.0

def correct_skew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Corrige a inclinação da imagem.
    
    Args:
        image: Imagem a ser corrigida
        angle: Ângulo de rotação em graus
    
    Returns:
        Imagem corrigida
    """
    if abs(angle) < 0.5:  # Não corrigir se o ângulo for muito pequeno
        return image
    
    # Obter dimensões da imagem
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Criar matriz de rotação
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calcular novas dimensões após rotação
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Ajustar matriz de rotação para centralizar a imagem
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Aplicar rotação
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detecta regiões de texto na imagem usando análise morfológica.
    
    Returns:
        Lista de retângulos (x, y, w, h) contendo texto
    """
    try:
        # Converter para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar threshold adaptativo
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Criar kernel para operações morfológicas
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detectar linhas horizontais e verticais
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
        
        # Combinar linhas para encontrar regiões de texto
        combined = cv2.add(horizontal_lines, vertical_lines)
        
        # Dilatação para conectar componentes próximos
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(combined, kernel_dilate, iterations=3)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por tamanho e razão de aspecto
        text_regions = []
        min_area = 500  # Área mínima
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > min_area and w > 50 and h > 20:
                text_regions.append((x, y, w, h))
        
        # Ordenar regiões por posição (top-to-bottom, left-to-right)
        text_regions.sort(key=lambda region: (region[1], region[0]))
        
        return text_regions
    
    except Exception as e:
        print(f"Erro na detecção de regiões de texto: {e}")
        # Retornar a imagem inteira como uma única região
        h, w = image.shape[:2]
        return [(0, 0, w, h)]

def enhance_text_contrast(image: np.ndarray) -> np.ndarray:
    """
    Melhora o contraste especificamente para texto usando técnicas avançadas.
    """
    try:
        # Converter para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Equalização de histograma adaptativa por regiões
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Aplicar filtro de realce de borda suave
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 3. Normalização gamma adaptativa
        gamma = 1.2
        gamma_corrected = np.power(sharpened / 255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        # 4. Redução de ruído preservando bordas
        denoised = cv2.bilateralFilter(gamma_corrected, 9, 80, 80)
        
        return denoised
    
    except Exception as e:
        print(f"Erro no enhancement de contraste: {e}")
        return image

def remove_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Remove artefatos comuns como pontos, linhas e ruídos.
    """
    try:
        # Converter para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Remover pontos pequenos (ruído salt-and-pepper)
        kernel_small = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_small)
        
        # 2. Detectar e remover linhas horizontais/verticais que não são texto
        binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Kernel para detectar linhas horizontais longas
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Kernel para detectar linhas verticais longas
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combinar linhas detectadas
        lines_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Remover linhas da imagem original (inverter máscara)
        lines_mask_inv = cv2.bitwise_not(lines_mask)
        cleaned = cv2.bitwise_and(opened, lines_mask_inv)
        
        # 3. Aplicar fechamento para conectar letras quebradas
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        return closed
    
    except Exception as e:
        print(f"Erro na remoção de artefatos: {e}")
        return image

def preprocess_image(image: np.ndarray, enhancement_level: str = "medium") -> np.ndarray:
    """
    Pré-processa imagem com técnicas avançadas baseadas no nível de melhoria especificado.
    Agora inclui: detecção de layout, correção de perspectiva, e realce adaptativo.
    """
    try:
        original_image = image.copy()
        
        # 1. **CORREÇÃO DE PERSPECTIVA AUTOMÁTICA**
        print("🔄 Aplicando correção de perspectiva...")
        corrected_image = auto_perspective_correction(original_image)
        
        # 2. **DETECÇÃO INTELIGENTE DE LAYOUT**
        print("🔍 Analisando layout do documento...")
        layout_info = intelligent_layout_detection(corrected_image)
        print(f"📄 Layout detectado: {layout_info['type']} ({layout_info['column_count']} colunas)")
        
        # 3. **REALCE ADAPTATIVO BASEADO NA QUALIDADE**
        print("✨ Aplicando realce adaptativo...")
        enhanced_image = adaptive_image_enhancement(corrected_image, enhancement_level)
        
        # 4. **PROCESSAMENTO ESPECÍFICO POR NÍVEL**
        if enhancement_level == "conservative":
            return apply_conservative_processing(enhanced_image)
        elif enhancement_level == "medium":
            return apply_medium_processing(enhanced_image)
        elif enhancement_level == "aggressive":
            return apply_aggressive_processing(enhanced_image, layout_info)
        elif enhancement_level == "ultra":
            return apply_ultra_processing(enhanced_image, layout_info)
        else:
            return enhanced_image
            
    except Exception as e:
        print(f"❌ Erro no pré-processamento: {e}")
        return image

def apply_conservative_processing(image: np.ndarray) -> np.ndarray:
    """Processamento conservador com mínima alteração da imagem."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apenas suavização muito leve
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Threshold adaptativo suave
        processed = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        
        return processed
        
    except Exception as e:
        print(f"Erro no processamento conservador: {e}")
        return image

def apply_medium_processing(image: np.ndarray) -> np.ndarray:
    """Processamento médio balanceado."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Denoising moderado
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        
        # Equalização de histograma adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Threshold adaptativo
        processed = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6
        )
        
        return processed
        
    except Exception as e:
        print(f"Erro no processamento médio: {e}")
        return image

def apply_aggressive_processing(image: np.ndarray, layout_info: Dict) -> np.ndarray:
    """Processamento agressivo com otimizações baseadas no layout."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Denoising mais forte
        denoised = cv2.fastNlMeansDenoising(gray, h=20)
        
        # Filtro bilateral para preservar bordas
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Sharpening adaptativo
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(bilateral, -1, kernel)
        
        # Normalização
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold adaptativo otimizado para o tipo de layout
        if layout_info.get('has_tables', False):
            # Para tabelas, usar threshold mais rígido
            processed = cv2.adaptiveThreshold(
                normalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4
            )
        else:
            # Para texto normal
            processed = cv2.adaptiveThreshold(
                normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
            )
        
        return processed
        
    except Exception as e:
        print(f"Erro no processamento agressivo: {e}")
        return image

def apply_ultra_processing(image: np.ndarray, layout_info: Dict) -> np.ndarray:
    """
    Processamento ultra com todas as técnicas avançadas.
    OTIMIZADO para reduzir ruído ao máximo.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # **1. DENOISING MULTI-ESTÁGIO**
        # Primeiro estágio: denoising suave
        denoised1 = cv2.fastNlMeansDenoising(gray, h=12)
        
        # **2. CORREÇÃO DE ORIENTAÇÃO CONSERVADORA**
        angle = detect_skew_angle(denoised1)
        if abs(angle) > 1.0:  # Só corrigir se significativo
            print(f"🔄 Corrigindo orientação: {angle:.2f}°")
            denoised1 = rotate_image(denoised1, angle)
        
        # **3. FILTRO BILATERAL PARA PRESERVAR BORDAS**
        bilateral = cv2.bilateralFilter(denoised1, 7, 50, 50)
        
        # **4. SHARPENING MUITO SUAVE**
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(bilateral, -1, kernel)
        
        # **5. NORMALIZAÇÃO CUIDADOSA**
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        # **6. THRESHOLD ADAPTATIVO OTIMIZADO**
        processed = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        
        # **7. LIMPEZA PÓS-PROCESSAMENTO**
        # Remover pontos isolados pequenos
        kernel_clean = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_clean)
        
        return cleaned
        
    except Exception as e:
        print(f"❌ Erro no processamento ultra: {e}")
        return image

def extract_text_with_multiple_configs(image):
    """
    Tenta extrair texto usando múltiplas configurações do Tesseract
    e retorna o melhor resultado baseado em heurísticas avançadas.
    """
    
    # Configurações otimizadas do Tesseract para diferentes tipos de documento
    configs = [
        # Configuração principal otimizada para português - melhor qualidade
        '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ0123456789.,;:!?()[]{} -c tessedit_enable_dict_correction=1 -c preserve_interword_spaces=1',
        
        # Para documentos de alta qualidade com texto limpo
        '--oem 3 --psm 6 -c tessedit_enable_dict_correction=1 -c preserve_interword_spaces=1 -c textord_really_old_xheight=0',
        
        # Para documentos com layout simples
        '--oem 3 --psm 4 -c preserve_interword_spaces=1 -c tessedit_enable_dict_correction=1',
        
        # Para blocos de texto únicos
        '--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_enable_dict_correction=1 -c textord_heavy_nr=0',
        
        # Para documentos digitalizados com boa qualidade
        '--oem 3 --psm 3 -c tessedit_enable_dict_correction=1 -c textord_really_old_xheight=1',
        
        # Para linhas de texto individuais
        '--oem 3 --psm 7 -c tessedit_enable_dict_correction=1 -c preserve_interword_spaces=1',
        
        # Configuração mais conservadora para textos difíceis
        '--oem 1 --psm 6 -c tessedit_enable_dict_correction=1 -c preserve_interword_spaces=1',
        
        # Para documentos com layout complexo
        '--oem 3 --psm 1 -c tessedit_enable_dict_correction=1 -c textord_heavy_nr=1',
    ]
    
    results = []
    
    for config in configs:
        try:
            # Extrair texto
            text = pytesseract.image_to_string(image, lang='por', config=config)
            
            # Pós-processar o texto
            text = post_process_text(text)
            
            # Extrair dados de confiança
            data = pytesseract.image_to_data(image, lang='por', config=config, output_type=pytesseract.Output.DICT)
            
            # Calcular qualidade do resultado (heurística avançada)
            confidence = calculate_average_confidence(data)
            quality_score = calculate_text_quality(text, confidence)
            
            results.append({
                'text': text,
                'confidence': confidence,
                'quality': quality_score,
                'config': config,
                'data': data,
                'engine': 'Tesseract'
            })
            
        except Exception as e:
            continue
    
    # Retornar o melhor resultado
    if results:
        best_result = max(results, key=lambda x: x['quality'])
        return best_result
    else:
        return {
            'text': "",
            'confidence': 0,
            'quality': 0,
            'config': "default",
            'data': {},
            'engine': 'Tesseract'
        }

def calculate_text_quality(text: str, confidence: float, layout_info: Dict = None) -> float:
    """
    Calcula a qualidade do texto extraído com base em múltiplos fatores.
    Agora inclui análise baseada no layout detectado.
    
    Args:
        text: Texto extraído
        confidence: Confiança média do OCR
        layout_info: Informações sobre o layout do documento
    
    Returns:
        float: Pontuação de qualidade (0-100)
    """
    if not text or not text.strip():
        return 0
    
    try:
        score = 0
        text = text.strip()
        
        # **1. CONFIANÇA DO OCR (Peso: 35%)**
        confidence_score = min(confidence, 100) * 0.35
        score += confidence_score
        
        # **2. ANÁLISE ESTRUTURAL DO TEXTO (Peso: 25%)**
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        words = text.split()
        
        # Pontuação por número de palavras
        word_count_score = min(len(words) * 1.2, 60)  # Máximo 60 pontos
        score += word_count_score * 0.25
        
        # **3. QUALIDADE DOS CARACTERES (Peso: 20%)**
        # Análise do comprimento médio das palavras
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 2 <= avg_word_length <= 12:
                char_quality_score = 20
            elif avg_word_length > 15:
                char_quality_score = 5  # Penaliza palavras muito longas (possível ruído)
            else:
                char_quality_score = 10
        else:
            char_quality_score = 0
        
        score += char_quality_score
        
        # **4. DETECÇÃO DE RUÍDO (Peso: -10% a -30%)**
        # Caracteres isolados e ruído
        isolated_chars = len(re.findall(r'\b[|!@#$%^&*=+~`]\b', text))
        noise_penalty = min(isolated_chars * 12, 30)  # Penalidade aumentada
        score -= noise_penalty
        
        # Sequências repetitivas suspeitas
        repetitive_patterns = len(re.findall(r'([|!@#$%^&*=+~-])\1{1,}', text))
        repetitive_penalty = min(repetitive_patterns * 8, 20)
        score -= repetitive_penalty
        
        # **5. PALAVRAS PORTUGUESAS COMUNS (Peso: 15%)**
        portuguese_words = {
            'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'suas', 'numa', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'pelas', 'são', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'onde', 'bem', 'te', 'dela', 'tu', 'antes', 'vem', 'porque', 'nada', 'dizer', 'cada', 'grande', 'estado', 'fazer', 'governo', 'ainda', 'sobre', 'nacional', 'trabalho', 'caso', 'grupo', 'durante', 'público', 'primeiro', 'tempo', 'ano', 'anos', 'acordo', 'geral', 'parte', 'lugar', 'vida', 'dia', 'forma', 'área', 'momento', 'desenvolvimento', 'processo', 'sistema', 'política', 'empresa', 'pessoa', 'programa', 'problema', 'projeto', 'serviço', 'mercado', 'recursos', 'social', 'informação', 'dados', 'valor'
        }
        
        word_matches = 0
        valid_words = [word.lower().strip('.,;:!?()[]{}"\'-') for word in words if len(word) > 2]
        
        for word in valid_words:
            if word in portuguese_words:
                word_matches += 1
        
        if valid_words:
            portuguese_ratio = word_matches / len(valid_words)
            portuguese_score = min(portuguese_ratio * 45, 45)  # Aumentado para 45
        else:
            portuguese_score = 0
        
        score += portuguese_score
        
        # **6. ANÁLISE DE LAYOUT ESPECÍFICO (Peso: 10%)**
        layout_score = 0
        if layout_info:
            # Bonus para textos que respeitam o layout detectado
            if layout_info.get('type') == 'table_mixed' and '|' in text:
                layout_score += 8  # Bonus para texto que parece tabela
            elif layout_info.get('column_count', 1) > 1:
                # Para múltiplas colunas, verificar se o texto tem estrutura adequada
                if len(lines) > 5:  # Textos com múltiplas linhas são esperados
                    layout_score += 6
            elif layout_info.get('type') == 'single_column':
                # Para coluna única, premiar textos coesos
                if len(lines) > 0 and all(len(line.split()) > 2 for line in lines[:5]):
                    layout_score += 7
            
            # Bonus para textos com cabeçalhos detectados apropriadamente
            if layout_info.get('header_regions') and any(line.isupper() or line.istitle() for line in lines[:3]):
                layout_score += 5
        
        score += layout_score
        
        # **7. PROPORÇÃO DE LETRAS (Peso: 10%)**
        if text:
            letter_count = sum(1 for c in text if c.isalpha())
            total_chars = len(text.replace(' ', '').replace('\n', ''))
            
            if total_chars > 0:
                letter_ratio = letter_count / total_chars
                if letter_ratio >= 0.7:  # Ideal: pelo menos 70% letras
                    ratio_score = 15
                elif letter_ratio >= 0.5:
                    ratio_score = 10
                elif letter_ratio >= 0.3:
                    ratio_score = 5
                else:
                    ratio_score = 0
                    score -= 25  # Penalidade forte para textos com poucas letras
            else:
                ratio_score = 0
        else:
            ratio_score = 0
        
        score += ratio_score
        
        # **8. FORMATAÇÃO E ESTRUTURA (Peso: 5%)**
        # Maiúsculas no início de frases
        sentences = re.split(r'[.!?]+', text)
        proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        caps_score = min(proper_caps * 3, 10)
        
        # Parágrafos bem formados
        paragraphs = text.split('\n\n')
        paragraph_score = min(len(paragraphs) * 2, 6)
        
        formatting_score = caps_score + paragraph_score
        score += formatting_score
        
        # **9. PENALIZAÇÃO POR LINHAS MUITO CURTAS**
        if lines:
            short_lines = sum(1 for line in lines if len(line.split()) <= 2)
            if short_lines / len(lines) > 0.6:  # Mais de 60% das linhas são muito curtas
                score -= 15
        
        # **10. LIMITAR SCORE ENTRE 0 E 100**
        final_score = max(0, min(100, score))
        
        return final_score
        
    except Exception as e:
        print(f"Erro no cálculo de qualidade: {e}")
        return 0

def post_process_text(text: str) -> str:
    """
    Aplica pós-processamento ao texto extraído para corrigir erros comuns.
    """
    if not text:
        return text
    
    processed_text = text
    
    # 1. Remover caracteres isolados suspeitos (muito comuns em OCR com ruído)
    # Remover caracteres únicos que não fazem sentido
    processed_text = re.sub(r'\b[|!@#$%^&*=+~`<>{}[\]\\]+\b', ' ', processed_text)
    
    # 2. Corrigir sequências de caracteres especiais repetitivos
    processed_text = re.sub(r'[|_\-=]{3,}', ' ', processed_text)  # Múltiplos traços
    processed_text = re.sub(r'[.]{3,}', '...', processed_text)  # Múltiplos pontos
    
    # 3. Correções de caracteres comuns mal interpretados
    corrections = {
        # OCR confusions comuns em contexto
        '—': '-',
        '–': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',
        '°': 'o',  # Grau vs o minúsculo
        '¨': '"',  # Trema vs aspas
        '´': "'",  # Acento vs apóstrofe
    }
    
    # Aplicar correções básicas
    for wrong, correct in corrections.items():
        processed_text = processed_text.replace(wrong, correct)
    
    # 4. Correções contextuais mais inteligentes
    # Corrigir números mal interpretados apenas em contexto de palavras
    processed_text = re.sub(r'\b0(?=[a-zA-ZÀ-ÿ])', 'O', processed_text)  # 0 seguido de letra -> O
    processed_text = re.sub(r'(?<=[a-zA-ZÀ-ÿ])0\b', 'o', processed_text)  # 0 precedido de letra -> o
    processed_text = re.sub(r'\b1(?=[a-zA-ZÀ-ÿ])', 'I', processed_text)  # 1 seguido de letra -> I
    processed_text = re.sub(r'(?<=[a-zA-ZÀ-ÿ])1(?=[a-zA-ZÀ-ÿ])', 'l', processed_text)  # 1 entre letras -> l
    
    # 5. Corrigir letras mal interpretadas como números em contexto
    processed_text = re.sub(r'(?<=[a-zA-ZÀ-ÿ])5(?=[a-zA-ZÀ-ÿ])', 'S', processed_text)  # 5 entre letras -> S
    processed_text = re.sub(r'(?<=[a-zA-ZÀ-ÿ])8(?=[a-zA-ZÀ-ÿ])', 'B', processed_text)  # 8 entre letras -> B
    
    # 6. Remover linhas que são principalmente ruído
    lines = processed_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
            
        # Calcular proporção de caracteres alfabéticos
        alpha_chars = sum(1 for c in line if c.isalpha())
        total_chars = len(line)
        
        # Manter linha se tiver proporção razoável de letras ou for muito curta
        if total_chars <= 3 or (total_chars > 0 and alpha_chars / total_chars >= 0.3):
            # Remover caracteres isolados suspeitos no início/fim da linha
            line = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', line).strip()
            if line:
                cleaned_lines.append(line)
    
    processed_text = '\n'.join(cleaned_lines)
    
    # 7. Limpar espaçamentos múltiplos
    processed_text = re.sub(r' +', ' ', processed_text)
    
    # 8. Corrigir quebras de linha problemáticas
    # Remover quebras de linha no meio de palavras (hifenização)
    processed_text = re.sub(r'(\w)-\n(\w)', r'\1\2', processed_text)
    
    # 9. Normalizar pontuação
    processed_text = re.sub(r'\s+([,.;:!?])', r'\1', processed_text)  # Remove espaço antes de pontuação
    processed_text = re.sub(r'([,.;:!?])(?![,.\s])', r'\1 ', processed_text)  # Adiciona espaço após pontuação
    
    # 10. Remover linhas muito curtas que são provavelmente ruído
    lines = processed_text.split('\n')
    final_lines = []
    
    for line in lines:
        line = line.strip()
        # Manter apenas linhas com pelo menos 2 caracteres ou que sejam números
        if len(line) >= 2 or (len(line) == 1 and line.isdigit()):
            final_lines.append(line)
    
    return '\n'.join(final_lines).strip()

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
async def convert_pdf_to_searchable(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    enhancement_level: str = "medium",
    resolution_scale: float = 2.0
):
    """
    Converte um PDF não pesquisável (com imagens) em um PDF pesquisável usando OCR.
    
    Args:
        file: Arquivo PDF para conversão
        enhancement_level: "basic", "medium", "aggressive", "ultra" - Nível de pré-processamento
        resolution_scale: Fator de escala para resolução (1.0-4.0, padrão 2.0)
    """
    
    # Validar tipo do arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    # Validar parâmetros
    if enhancement_level not in ["basic", "medium", "aggressive", "ultra"]:
        raise HTTPException(status_code=400, detail="enhancement_level deve ser 'basic', 'medium', 'aggressive' ou 'ultra'")
    
    if not 1.0 <= resolution_scale <= 4.0:
        raise HTTPException(status_code=400, detail="resolution_scale deve estar entre 1.0 e 4.0")
    
    try:
        # Ler o arquivo PDF
        pdf_content = await file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Criar um novo PDF para armazenar o resultado
        output_pdf = fitz.open()
        
        # Processar cada página
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Converter página para imagem com resolução ajustada
            pix = page.get_pixmap(matrix=fitz.Matrix(resolution_scale, resolution_scale))
            img_data = pix.tobytes("png")
            
            # Converter para PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Aplicar pré-processamento
            processed_image = preprocess_image(image, enhancement_level)
            
            # Extrair texto usando OCR
            result = extract_text_with_multiple_configs(processed_image)
            texto_ocr = result['text']
            config = result['config']
            ocr_data = result['data']
            
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
            filename=f"ocr_{enhancement_level}_{file.filename}",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar PDF: {str(e)}")

@app.post("/extract-text/")
async def extract_text_from_pdf(
    file: UploadFile = File(...),
    enhancement_level: str = "medium",
    resolution_scale: float = 2.0
):
    """
    Extrai apenas o texto de um PDF não pesquisável usando OCR.
    
    Args:
        file: Arquivo PDF para extração
        enhancement_level: "basic", "medium", "aggressive", "ultra" - Nível de pré-processamento
        resolution_scale: Fator de escala para resolução (1.0-4.0, padrão 2.0)
    """
    
    # Validar tipo do arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    # Validar parâmetros
    if enhancement_level not in ["basic", "medium", "aggressive", "ultra"]:
        raise HTTPException(status_code=400, detail="enhancement_level deve ser 'basic', 'medium', 'aggressive' ou 'ultra'")
    
    if not 1.0 <= resolution_scale <= 4.0:
        raise HTTPException(status_code=400, detail="resolution_scale deve estar entre 1.0 e 4.0")
    
    try:
        # Ler o arquivo PDF
        pdf_content = await file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        extracted_text = []
        
        # Processar cada página
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Converter página para imagem com resolução ajustada
            pix = page.get_pixmap(matrix=fitz.Matrix(resolution_scale, resolution_scale))
            img_data = pix.tobytes("png")
            
            # Converter para PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Aplicar pré-processamento
            processed_image = preprocess_image(image, enhancement_level)
            
            # Extrair texto usando OCR
            result = extract_text_with_multiple_configs(processed_image)
            texto_ocr = result['text']
            config = result['config']
            ocr_data = result['data']
            
            extracted_text.append({
                "pagina": page_num + 1,
                "texto": texto_ocr.strip(),
                "configuracao": config,
                "parametros": {
                    "enhancement_level": enhancement_level,
                    "resolution_scale": resolution_scale
                }
            })
        
        pdf_document.close()
        
        return {
            "filename": file.filename,
            "total_paginas": len(extracted_text),
            "texto_extraido": extracted_text,
            "configuracao_global": {
                "enhancement_level": enhancement_level,
                "resolution_scale": resolution_scale
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair texto: {str(e)}")

@app.post("/extract-text-progress/")
async def extract_text_with_progress(
    file: UploadFile = File(...),
    enhancement_level: str = "medium",
    resolution_scale: float = 2.0
):
    """
    Extrai texto de PDF com indicador de progresso em tempo real usando Server-Sent Events.
    
    Args:
        file: Arquivo PDF para extração
        enhancement_level: "basic", "medium", "aggressive", "ultra" - Nível de pré-processamento
        resolution_scale: Fator de escala para resolução (1.0-4.0, padrão 2.0)
    """
    
    # Validar tipo do arquivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    # Validar parâmetros
    if enhancement_level not in ["basic", "medium", "aggressive", "ultra"]:
        raise HTTPException(status_code=400, detail="enhancement_level deve ser 'basic', 'medium', 'aggressive' ou 'ultra'")
    
    if not 1.0 <= resolution_scale <= 4.0:
        raise HTTPException(status_code=400, detail="resolution_scale deve estar entre 1.0 e 4.0")
    
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
            
            all_text = []
            start_time = time.time()
            
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
                        "caracteres_extraidos": 0,
                        "tempo_decorrido": round(asyncio.get_event_loop().time(), 2)
                    }
                }
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                
                page = pdf_document[page_num]
                
                # Converter página para imagem com resolução ajustada
                pix = page.get_pixmap(matrix=fitz.Matrix(resolution_scale, resolution_scale))
                img_data = pix.tobytes("png")
                
                # Converter para PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Aplicar pré-processamento
                processed_image = preprocess_image(image, enhancement_level)
                
                # Extrair texto usando OCR
                result = extract_text_with_multiple_configs(processed_image)
                texto_ocr = result['text']
                config = result['config']
                ocr_data = result['data']
                
                # Enviar informações da página processada
                page_info = {
                    "tipo": "pagina_concluida",
                    "timestamp": datetime.now().isoformat(),
                    "arquivo": {
                        "nome": file.filename,
                        "pagina_atual": page_num + 1,
                        "total_paginas": total_pages
                    },
                    "processamento": {
                        "porcentagem": int((page_num + 1) / total_pages * 100),
                        "tempo_decorrido": f"{time.time() - start_time:.2f}s",
                        "configuracao_ocr": config,
                        "parametros": {
                            "enhancement_level": enhancement_level,
                            "resolution_scale": resolution_scale
                        }
                    },
                    "resultados": {
                        "pagina": page_num + 1,
                        "texto": texto_ocr.strip(),
                        "estatisticas": {
                            "caracteres": len(texto_ocr.strip()),
                            "linhas": len(texto_ocr.strip().split('\n')) if texto_ocr.strip() else 0,
                            "palavras": len(texto_ocr.strip().split()) if texto_ocr.strip() else 0
                        }
                    }
                }
                
                all_text.append(page_info["resultados"])
                yield f"data: {json.dumps(page_info, ensure_ascii=False)}\n\n"
                
            # Calcular estatísticas finais
            total_characters = sum(page["estatisticas"]["caracteres"] for page in all_text)
            total_words = sum(page["estatisticas"]["palavras"] for page in all_text)
            total_lines = sum(page["estatisticas"]["linhas"] for page in all_text)
            
            # Encontrar página com mais e menos texto
            pages_with_text = [page for page in all_text if page["estatisticas"]["caracteres"] > 0]
            longest_page = max(pages_with_text, key=lambda x: x["estatisticas"]["caracteres"]) if pages_with_text else None
            shortest_page = min(pages_with_text, key=lambda x: x["estatisticas"]["caracteres"]) if pages_with_text else None
            
            # Enviar resultado final
            final_result = {
                "tipo": "concluido",
                "timestamp": datetime.now().isoformat(),
                "arquivo": {
                    "nome": file.filename,
                    "total_paginas": total_pages,
                    "tempo_total": f"{time.time() - start_time:.2f}s"
                },
                "processamento": {
                    "porcentagem": 100,
                    "configuracao_global": {
                        "enhancement_level": enhancement_level,
                        "resolution_scale": resolution_scale
                    },
                    "sucesso": True
                },
                "estatisticas": {
                    "resumo": {
                        "total_caracteres": total_characters,
                        "total_palavras": total_words,
                        "total_linhas": total_lines,
                        "paginas_com_texto": len(pages_with_text),
                        "paginas_vazias": total_pages - len(pages_with_text)
                    },
                    "pagina_mais_longa": longest_page,
                    "pagina_mais_curta": shortest_page
                },
                "resultados": {
                    "texto_completo": "\n\n".join([page["texto"] for page in all_text if page["texto"]]),
                    "paginas": all_text
                }
            }
            
            yield f"data: {json.dumps(final_result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            error_result = {
                "tipo": "erro",
                "timestamp": datetime.now().isoformat(),
                "erro": {
                    "message": str(e),
                    "type": type(e).__name__
                },
                "processamento": {
                    "porcentagem": 0,
                    "sucesso": False
                }
            }
            yield f"data: {json.dumps(error_result, ensure_ascii=False)}\n\n"
    
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
    global engines_loading
    
    # Se engines estão carregando, ainda considerar saudável
    if engines_loading:
        return {
            "status": "OK", 
            "message": "API funcionando - engines de IA carregando em background",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    return {
        "status": "OK", 
        "message": "API funcionando corretamente",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/exemplo_progress.html")
async def get_example_page():
    """
    Serve a página de exemplo para teste do endpoint de progresso.
    """
    return FileResponse("exemplo_progress.html", media_type="text/html")

def extract_text_with_ai_engines(image):
    """
    Extrai texto usando múltiplos engines de IA com análise de consenso.
    """
    print("\n🚀 INICIANDO EXTRAÇÃO COM MÚLTIPLOS ENGINES DE IA")
    
    results = []
    layout_info = intelligent_layout_detection(image)
    
    print(f"📊 Layout detectado: {layout_info}")
    
    # Engine 1: Tesseract com múltiplas configurações
    print("\n🔍 ENGINE 1: TESSERACT")
    try:
        tesseract_result = extract_text_with_multiple_configs(image)
        if tesseract_result and tesseract_result.get('text', '').strip():
            results.append({
                'text': tesseract_result['text'],
                'confidence': tesseract_result['confidence'],
                'engine': 'Tesseract',
                'method': tesseract_result.get('method', 'multi_config')
            })
            print(f"✅ Tesseract - Confiança: {tesseract_result['confidence']:.1f}%")
            print(f"📝 Texto extraído (primeiros 200 chars): {tesseract_result['text'][:200]}...")
        else:
            print("❌ Tesseract - Falhou na extração")
    except Exception as e:
        print(f"❌ Tesseract - Erro: {e}")
    
    # Engine 2: EasyOCR
    print("\n🔍 ENGINE 2: EASYOCR")
    try:
        easyocr_results = easyocr_reader.readtext(image)
        if easyocr_results:
            easyocr_text = ' '.join([result[1] for result in easyocr_results])
            easyocr_confidence = sum([result[2] for result in easyocr_results]) / len(easyocr_results) * 100
            results.append({
                'text': easyocr_text,
                'confidence': easyocr_confidence,
                'engine': 'EasyOCR',
                'method': 'neural_network'
            })
            print(f"✅ EasyOCR - Confiança: {easyocr_confidence:.1f}%")
            print(f"📝 Texto extraído (primeiros 200 chars): {easyocr_text[:200]}...")
        else:
            print("❌ EasyOCR - Nenhum texto detectado")
    except Exception as e:
        print(f"❌ EasyOCR - Erro: {e}")
    
    # Engine 3: TrOCR (Transformer OCR)
    try:
        print("\n🔍 ENGINE 3: TrOCR")
        from PIL import Image as PILImage
        
        # Converter para PIL Image adequadamente
        if isinstance(image, np.ndarray):
            # Se é numpy array, converter para PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        elif hasattr(image, 'convert'):  # Já é PIL Image
            pil_image = image
        else:
            # Tentar converter de qualquer outro tipo
            pil_image = PILImage.fromarray(np.array(image))
        
        # Garantir que está em RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pixel_values = trocr_processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        trocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if trocr_text.strip():
            # TrOCR não fornece confiança direta, estimamos baseado no tamanho e qualidade
            trocr_confidence = min(95.0, len(trocr_text.strip()) * 2.5)
            results.append({
                'text': trocr_text,
                'confidence': trocr_confidence,
                'engine': 'TrOCR',
                'method': 'transformer'
            })
            print(f"✅ TrOCR - Confiança estimada: {trocr_confidence:.1f}%")
            print(f"📝 Texto extraído (primeiros 200 chars): {trocr_text[:200]}...")
        else:
            print("❌ TrOCR - Nenhum texto detectado")
    except Exception as e:
        print(f"❌ TrOCR - Erro: {e}")
    
    # Análise de consenso
    print(f"\n📊 ANÁLISE DE CONSENSO - {len(results)} engines executados com sucesso")
    
    if not results:
        print("❌ Nenhum engine conseguiu extrair texto")
        return {
            'text': '',
            'confidence': 0.0,
            'engine': 'none',
            'method': 'failed'
        }
    
    print("\n🏆 INICIANDO ANÁLISE AVANÇADA DE CONSENSO...")
    consensus_result = analyze_consensus_advanced(results, layout_info)
    
    if consensus_result:
        print(f"\n🎯 RESULTADO FINAL DO CONSENSO:")
        print(f"🏅 Melhor engine: {consensus_result.get('engine', 'unknown')}")
        print(f"📊 Score de consenso: {consensus_result.get('consensus_score', 0):.2f}")
        print(f"🎯 Confiança final: {consensus_result.get('confidence', 0):.1f}%")
        print(f"📝 Texto final (primeiros 300 chars): {consensus_result.get('text', '')[:300]}...")
        
        return {
            'text': consensus_result['text'],
            'confidence': consensus_result['confidence'],
            'engine': consensus_result.get('engine', 'consensus'),
            'method': consensus_result.get('method', 'advanced_analysis')
        }
    else:
        print("❌ Análise de consenso falhou, usando melhor resultado individual")
        best_result = max(results, key=lambda x: x['confidence'])
        return {
            'text': best_result['text'],
            'confidence': best_result['confidence'],
            'engine': best_result['engine'],
            'method': best_result.get('method', 'fallback')
        }

def analyze_consensus_advanced(results: List[Dict], layout_info: Dict) -> Dict:
    """
    Análise avançada de consenso entre múltiplos engines de OCR.
    Considera qualidade, confiança, similaridade e contexto do layout.
    """
    try:
        if not results:
            return {'text': '', 'confidence': 0, 'engine': 'none'}
        
        if len(results) == 1:
            return results[0]
        
        print(f"🔍 Analisando consenso entre {len(results)} resultados...")
        
        # **1. ANÁLISE DE SIMILARIDADE ENTRE TEXTOS**
        similarity_matrix = []
        for i, result1 in enumerate(results):
            similarities = []
            for j, result2 in enumerate(results):
                if i != j:
                    sim = calculate_text_similarity_advanced(result1['text'], result2['text'])
                    similarities.append(sim)
                else:
                    similarities.append(1.0)
            similarity_matrix.append(similarities)
        
        # **2. PONTUAÇÃO PONDERADA PARA CADA RESULTADO**
        scored_results = []
        
        for i, result in enumerate(results):
            score = 0
            
            # Peso base da qualidade do texto (40%)
            quality_score = result.get('quality', 0) * 0.4
            score += quality_score
            
            # Peso da confiança do engine (30%)
            confidence_score = result.get('confidence', 0) * 0.3
            score += confidence_score
            
            # Peso do consenso com outros engines (20%)
            avg_similarity = sum(similarity_matrix[i]) / len(similarity_matrix[i])
            consensus_score = avg_similarity * 20
            score += consensus_score
            
            # Bonus baseado no tipo de engine (10%)
            engine_bonus = 0
            if 'Tesseract' in result.get('engine', ''):
                engine_bonus = 8  # Tesseract é confiável para português
            elif 'TrOCR' in result.get('engine', ''):
                engine_bonus = 10  # Transformers são muito bons
            elif 'EasyOCR' in result.get('engine', ''):
                engine_bonus = 9  # Neural networks são eficazes
            
            score += engine_bonus
            
            # Bonus baseado no layout (ajuste fino)
            layout_bonus = 0
            if layout_info.get('has_tables', False) and 'psm 4' in result.get('engine', ''):
                layout_bonus = 5  # Bonus para PSM 4 em tabelas
            elif layout_info.get('column_count', 1) > 1 and 'psm 3' in result.get('engine', ''):
                layout_bonus = 5  # Bonus para PSM 3 em múltiplas colunas
            
            score += layout_bonus
            
            # Penalização por texto muito curto ou muito longo
            text_length = len(result['text'].strip())
            if text_length < 10:
                score -= 10  # Penaliza textos muito curtos
            elif text_length > 10000:
                score -= 5   # Penaliza textos excessivamente longos
            
            scored_results.append({
                **result,
                'consensus_score': score,
                'avg_similarity': avg_similarity,
                'breakdown': {
                    'quality': quality_score,
                    'confidence': confidence_score,
                    'consensus': consensus_score,
                    'engine_bonus': engine_bonus,
                    'layout_bonus': layout_bonus
                }
            })
        
        # **3. ORDENAR POR PONTUAÇÃO E SELECIONAR O MELHOR**
        scored_results.sort(key=lambda x: x['consensus_score'], reverse=True)
        
        best_result = scored_results[0]
        
        print(f"🏆 Melhor resultado: {best_result['engine']} (Score: {best_result['consensus_score']:.1f})")
        print(f"📊 Breakdown: Q={best_result['breakdown']['quality']:.1f} | C={best_result['breakdown']['confidence']:.1f} | S={best_result['breakdown']['consensus']:.1f}")
        
        # **4. VERIFICAÇÃO DE QUALIDADE MÍNIMA**
        if best_result['consensus_score'] < 30:  # Threshold mínimo
            print("⚠️ Todos os resultados têm qualidade baixa, aplicando fallback...")
            
            # Tentar combinar os melhores elementos
            combined_text = ""
            for result in scored_results[:2]:  # Pegar os 2 melhores
                if len(result['text'].strip()) > len(combined_text.strip()):
                    combined_text = result['text']
            
            return {
                'text': combined_text,
                'confidence': max(r['confidence'] for r in results),
                'engine': 'consensus_fallback',
                'method': 'emergency_consensus'
            }
        
        # **5. APRIMORAMENTO DO MELHOR RESULTADO**
        enhanced_text = enhance_consensus_result(best_result['text'], scored_results)
        
        return {
            'text': enhanced_text,
            'confidence': best_result['confidence'],
            'engine': best_result['engine'],
            'method': 'advanced_consensus',
            'score': best_result['consensus_score']
        }
        
    except Exception as e:
        print(f"❌ Erro na análise de consenso: {e}")
        # Fallback: retornar o resultado com maior confiança
        return max(results, key=lambda x: x.get('confidence', 0))

def calculate_text_similarity_advanced(text1: str, text2: str) -> float:
    """
    Calcula similaridade avançada entre dois textos usando múltiplas métricas.
    """
    try:
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Normalizar textos
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # **1. SIMILARIDADE DE CARACTERES (Jaccard)**
        set1 = set(t1)
        set2 = set(t2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        char_similarity = intersection / union if union > 0 else 0
        
        # **2. SIMILARIDADE DE PALAVRAS**
        words1 = set(t1.split())
        words2 = set(t2.split())
        word_intersection = len(words1.intersection(words2))
        word_union = len(words1.union(words2))
        word_similarity = word_intersection / word_union if word_union > 0 else 0
        
        # **3. SIMILARIDADE DE SEQUÊNCIA (RATIO)**
        sequence_similarity = SequenceMatcher(None, t1, t2).ratio()
        
        # **4. SIMILARIDADE DE COMPRIMENTO**
        len1, len2 = len(t1), len(t2)
        length_similarity = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # **5. COMBINAÇÃO PONDERADA**
        final_similarity = (
            char_similarity * 0.2 +
            word_similarity * 0.4 +
            sequence_similarity * 0.3 +
            length_similarity * 0.1
        )
        
        return final_similarity
        
    except Exception as e:
        print(f"Erro no cálculo de similaridade: {e}")
        return 0.0

def enhance_consensus_result(best_text: str, all_results: List[Dict]) -> str:
    """
    Aprimora o melhor resultado usando informações dos outros engines.
    """
    try:
        if len(all_results) <= 1:
            return best_text
        
        # Coletar palavras de alta confiança de todos os resultados
        high_confidence_words = set()
        
        for result in all_results:
            words = result['text'].split()
            # Adicionar palavras de engines com alta confiança
            if result.get('confidence', 0) > 70:
                high_confidence_words.update(words)
        
        # Verificar se podemos melhorar o texto principal
        enhanced_text = best_text
        
        # Verificação básica de integridade
        if len(enhanced_text.strip()) < 10:
            # Se o melhor resultado é muito curto, tentar o segundo melhor
            for result in all_results[1:]:
                if len(result['text'].strip()) > len(enhanced_text.strip()):
                    enhanced_text = result['text']
                    break
        
        return enhanced_text
        
    except Exception as e:
        print(f"Erro no aprimoramento do resultado: {e}")
        return best_text

def analyze_consensus(results):
    """
    Analisa consenso entre diferentes motores de OCR para melhorar precisão.
    """
    if len(results) < 2:
        return None
    
    # Extrair textos e normalizar
    texts = [result['text'].lower().strip() for result in results]
    
    # Calcular similaridade entre textos
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = calculate_text_similarity(texts[i], texts[j])
            similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # Se há alto consenso (similaridade > 0.7), combinar resultados
    if avg_similarity > 0.7:
        # Usar o resultado com maior qualidade, mas considerar consenso
        best_result = results[0].copy()
        best_result['consensus_score'] = avg_similarity
        best_result['engine'] = f"consensus_{best_result['engine']}"
        return best_result
    
    # Se baixo consenso, usar votação ponderada por confiança
    elif len(results) >= 3:
        weighted_texts = []
        for result in results:
            weight = result['confidence'] * result['quality'] / 100
            weighted_texts.append((result['text'], weight, result))
        
        # Escolher texto com maior peso combinado
        best_weighted = max(weighted_texts, key=lambda x: x[1])
        final_result = best_weighted[2].copy()
        final_result['engine'] = f"weighted_{final_result['engine']}"
        return final_result
    
    return None

def calculate_text_similarity(text1, text2):
    """
    Calcula similaridade entre dois textos usando múltiplas métricas.
    """
    if not text1 or not text2:
        return 0
    
    # 1. Similaridade de palavras (Jaccard)
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard = intersection / union if union > 0 else 0
    
    # 2. Similaridade de caracteres (razão)
    char_similarity = 1 - (abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1))
    
    # 3. Combinar métricas
    final_similarity = (jaccard * 0.7) + (char_similarity * 0.3)
    
    return final_similarity

def get_avg_confidence(ocr_data):
    """
    Extrai confiança média dos dados do OCR.
    """
    if not ocr_data or 'conf' not in ocr_data:
        return 0
    
    confidences = [float(conf) for conf in ocr_data['conf'] if float(conf) > 0]
    return sum(confidences) / len(confidences) if confidences else 0

def intelligent_layout_detection(image) -> Dict:
    """
    Detecta layout do documento usando análise avançada e IA.
    Retorna informações sobre colunas, tabelas, cabeçalhos, etc.
    """
    try:
        # Converter PIL Image para numpy array se necessário
        if hasattr(image, 'convert'):  # É um objeto PIL
            image = np.array(image.convert('RGB'))
        
        # Converter para numpy array se necessário
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Detecção de bordas para estrutura
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Detecção de linhas horizontais e verticais
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        
        # 3. Análise de estrutura de colunas
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        # 4. Detectar tipo de layout
        layout_info = {
            'type': 'single_column',
            'has_tables': False,
            'column_count': 1,
            'text_regions': [],
            'table_regions': [],
            'header_regions': [],
            'confidence': 0.8
        }
        
        # Análise de colunas baseada em espaços em branco
        height, width = gray.shape
        
        # Projeção horizontal para detectar colunas
        horizontal_projection = np.sum(gray == 255, axis=0)
        
        # Encontrar vales (espaços entre colunas)
        valley_threshold = height * 0.7
        valleys = []
        
        for i in range(1, len(horizontal_projection) - 1):
            if (horizontal_projection[i] > valley_threshold and 
                horizontal_projection[i-1] < valley_threshold and 
                horizontal_projection[i+1] < valley_threshold):
                valleys.append(i)
        
        if len(valleys) >= 1:
            layout_info['type'] = 'multi_column'
            layout_info['column_count'] = len(valleys) + 1
        
        # Detectar tabelas baseado em grade de linhas
        if h_lines is not None and v_lines is not None and len(h_lines) > 3 and len(v_lines) > 3:
            layout_info['has_tables'] = True
            layout_info['type'] = 'table_mixed'
        
        # Detectar regiões de cabeçalho (parte superior da página)
        header_region = gray[:int(height * 0.15), :]
        header_density = np.sum(header_region < 200) / header_region.size
        
        if header_density > 0.1:  # Tem conteúdo no cabeçalho
            layout_info['header_regions'].append({
                'bbox': (0, 0, width, int(height * 0.15)),
                'type': 'header'
            })
        
        return layout_info
        
    except Exception as e:
        print(f"Erro na detecção de layout: {e}")
        return {
            'type': 'single_column',
            'has_tables': False,
            'column_count': 1,
            'text_regions': [],
            'confidence': 0.5
        }

def auto_perspective_correction(image: np.ndarray) -> np.ndarray:
    """
    Correção automática de perspectiva usando detecção de bordas da página.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Blur e threshold para encontrar contornos
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 2. Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Encontrar o maior contorno (presumivelmente a página)
        if not contours:
            return image
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 4. Aproximar contorno para um quadrilátero
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 5. Se encontramos um quadrilátero, corrigir perspectiva
        if len(approx) == 4:
            # Ordenar pontos: top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            
            # Calcular dimensões da imagem de destino
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Destino da transformação
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            # Calcular matriz de transformação e aplicar
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            return warped
        
        return image
        
    except Exception as e:
        print(f"Erro na correção de perspectiva: {e}")
        return image

def order_points(pts):
    """
    Ordena pontos para correção de perspectiva.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left: menor soma, Bottom-right: maior soma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right: menor diferença, Bottom-left: maior diferença
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def adaptive_image_enhancement(image: np.ndarray, enhancement_level: str = "medium") -> np.ndarray:
    """
    Realce adaptativo baseado na qualidade e características da imagem.
    """
    try:
        # Analisar qualidade da imagem
        quality_metrics = analyze_image_quality(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        enhanced = gray.copy()
        
        # Aplicar realces baseados na qualidade detectada
        if quality_metrics['is_low_contrast']:
            # Melhorar contraste para imagens com baixo contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
        
        if quality_metrics['is_noisy']:
            # Reduzir ruído
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        if quality_metrics['is_blurry']:
            # Aplicar sharpening para imagens borradas
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        if quality_metrics['is_low_resolution']:
            # Upscaling para imagens de baixa resolução
            enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Ajustes finais baseados no nível de realce
        if enhancement_level in ["aggressive", "ultra"]:
            # Normalização de intensidade
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
            # Correção gamma adaptativa
            gamma = 1.2 if quality_metrics['is_dark'] else 0.8
            enhanced = adjust_gamma(enhanced, gamma)
        
        return enhanced
        
    except Exception as e:
        print(f"Erro no realce adaptativo: {e}")
        return image

def analyze_image_quality(image: np.ndarray) -> Dict:
    """
    Analisa qualidade da imagem para determinar realces necessários.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    metrics = {
        'is_low_contrast': False,
        'is_noisy': False,
        'is_blurry': False,
        'is_low_resolution': False,
        'is_dark': False,
        'overall_quality': 'good'
    }
    
    # 1. Análise de contraste
    contrast = gray.std()
    if contrast < 30:
        metrics['is_low_contrast'] = True
    
    # 2. Análise de ruído (usando Laplaciano)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        metrics['is_blurry'] = True
    
    # 3. Análise de resolução
    height, width = gray.shape
    if height < 300 or width < 300:
        metrics['is_low_resolution'] = True
    
    # 4. Análise de brilho
    mean_brightness = gray.mean()
    if mean_brightness < 80:
        metrics['is_dark'] = True
    
    # 5. Detecção de ruído usando gradientes
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    noise_level = np.sqrt(grad_x**2 + grad_y**2).mean()
    
    if noise_level > 50:
        metrics['is_noisy'] = True
    
    # 6. Qualidade geral
    quality_score = 0
    if not metrics['is_low_contrast']: quality_score += 1
    if not metrics['is_noisy']: quality_score += 1
    if not metrics['is_blurry']: quality_score += 1
    if not metrics['is_low_resolution']: quality_score += 1
    if not metrics['is_dark']: quality_score += 1
    
    if quality_score >= 4:
        metrics['overall_quality'] = 'excellent'
    elif quality_score >= 3:
        metrics['overall_quality'] = 'good'
    elif quality_score >= 2:
        metrics['overall_quality'] = 'fair'
    else:
        metrics['overall_quality'] = 'poor'
    
    return metrics

def adjust_gamma(image, gamma=1.0):
    """
    Ajusta gamma da imagem.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

@app.post("/extract-text-hybrid/")
async def extract_text_hybrid_endpoint(
    file: UploadFile = File(...),
    enhancement_level: str = Form("ultra"),
    use_ai_engines: bool = Form(True),
    engine_preference: str = Form("auto")
):
    """
    Endpoint para extração de texto híbrida com tecnologias avançadas.
    
    - **file**: Arquivo PDF para processamento
    - **enhancement_level**: Nível de melhoria da imagem (conservative, medium, aggressive, ultra)
    - **use_ai_engines**: Usar múltiplos motores de IA (EasyOCR, TrOCR)
    - **engine_preference**: Escolher engine específico:
        - "auto": Análise de consenso automática (padrão)
        - "tesseract": Usar apenas Tesseract
        - "easyocr": Usar apenas EasyOCR
        - "trocr": Usar apenas TrOCR
        - "consensus": Forçar análise de consenso entre todos
    
    Retorna texto extraído estruturado página por página com estatísticas detalhadas.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")
    
    # Validar engine_preference
    valid_engines = ["auto", "tesseract", "easyocr", "trocr", "consensus"]
    if engine_preference not in valid_engines:
        raise HTTPException(
            status_code=400, 
            detail=f"engine_preference deve ser um de: {', '.join(valid_engines)}"
        )
    
    input_pdf_path = None
    start_time = time.time()
    
    try:
        print(f"🚀 INICIANDO SISTEMA HÍBRIDO DE OCR")
        print(f"📁 Arquivo: {file.filename}")
        print(f"⚙️ Nível: {enhancement_level}")
        print(f"🤖 IA Engines: {'Ativados' if use_ai_engines else 'Desativados'}")
        print(f"🎯 Engine preferido: {engine_preference}")
        
        # Salvar arquivo temporário
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        input_pdf_path = os.path.join(temp_dir, f"input_{file_id}.pdf")
        
        with open(input_pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📄 Convertendo PDF para imagens...")
        
        # Lista de possíveis caminhos do Poppler para testar
        poppler_paths = [
            None,  # Primeiro tenta o PATH do sistema
            r"C:\poppler-24.08.0\Library\bin",  # Caminho correto para a versão conda
            r"C:\poppler-24.08.0\bin",
            r"C:\poppler\bin", 
            r"C:\Program Files\poppler\bin",
            r"C:\tools\poppler\bin"
        ]
        
        for poppler_path in poppler_paths:
            try:
                if poppler_path:
                    print(f"🔧 Tentando poppler em: {poppler_path}")
                    images = convert_from_path(input_pdf_path, dpi=300, fmt='jpeg', poppler_path=poppler_path)
                else:
                    print(f"🔧 Tentando poppler do PATH do sistema...")
                    images = convert_from_path(input_pdf_path, dpi=300, fmt='jpeg')
                print(f"✅ Poppler funcionando! Páginas encontradas: {len(images)}")
                break
            except Exception as e:
                if poppler_path:
                    print(f"❌ Falha com poppler em {poppler_path}: {str(e)}")
                else:
                    print(f"❌ Falha com poppler do PATH: {str(e)}")
                continue
        
        if images is None:
            raise HTTPException(
                status_code=500, 
                detail="❌ Erro crítico: Poppler não encontrado! Consulte POPPLER_INSTALACAO.md para instruções de instalação."
            )
        
        extracted_text = []
        total_pages = len(images)
        engines_used = set()
        total_confidence = 0
        confidence_count = 0
        
        print(f"📊 Processando {total_pages} páginas...")
        
        for page_num, image in enumerate(images):
            print(f"📄 Processando página {page_num + 1}/{total_pages}...")
            
            # Escolher método de extração baseado na preferência
            if engine_preference == "tesseract":
                # Usar apenas Tesseract
                result = extract_text_with_multiple_configs(image)
                engine_used = 'Tesseract'
                method_used = result.get('config', 'multiple_configs')
                
            elif engine_preference == "easyocr":
                # Usar apenas EasyOCR
                result = extract_text_with_easyocr_only(image)
                engine_used = 'EasyOCR'
                method_used = 'easyocr_only'
                
            elif engine_preference == "trocr":
                # Usar apenas TrOCR
                result = extract_text_with_trocr_only(image)
                engine_used = 'TrOCR'
                method_used = 'trocr_only'
                
            elif engine_preference in ["auto", "consensus"]:
                # Usar sistema híbrido com múltiplos motores
                if use_ai_engines:
                    result = extract_text_with_ai_engines(image)
                    engine_used = result.get('engine', 'Consensus')
                    method_used = result.get('method', 'ai_consensus')
                else:
                    # Se AI engines estão desativados, usar apenas Tesseract
                    result = extract_text_with_multiple_configs(image)
                    engine_used = 'Tesseract'
                    method_used = result.get('config', 'multiple_configs')
            
            engines_used.add(engine_used)
            
            # Extrair texto e confiança
            texto_extraido = result.get('text', '').strip()
            confidence = result.get('confidence', 0)
            
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
            
            # Estruturar informações da página
            page_info = {
                "pagina": page_num + 1,
                "texto": texto_extraido,
                "engine_usado": engine_used,
                "metodo": method_used,
                "confianca": round(confidence, 2) if confidence else 0,
                "estatisticas": {
                    "caracteres": len(texto_extraido),
                    "linhas": len(texto_extraido.split('\n')) if texto_extraido else 0,
                    "palavras": len(texto_extraido.split()) if texto_extraido else 0
                },
                "parametros": {
                    "enhancement_level": enhancement_level,
                    "engine_preference": engine_preference,
                    "use_ai_engines": use_ai_engines
                }
            }
            
            extracted_text.append(page_info)
        
        # Calcular estatísticas finais
        avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else 0
        processing_time = time.time() - start_time
        
        # Contar caracteres totais
        total_characters = sum(page['estatisticas']['caracteres'] for page in extracted_text)
        total_words = sum(page['estatisticas']['palavras'] for page in extracted_text)
        
        print(f"✅ PROCESSAMENTO CONCLUÍDO")
        print(f"📊 Total de caracteres extraídos: {total_characters}")
        print(f"📊 Engines utilizados: {list(engines_used)}")
        
        return {
            "filename": file.filename,
            "total_paginas": total_pages,
            "texto_extraido": extracted_text,
            "configuracao_global": {
                "enhancement_level": enhancement_level,
                "engine_preference": engine_preference,
                "use_ai_engines": use_ai_engines
            },
            "estatisticas_globais": {
                "paginas_processadas": total_pages,
                "engines_utilizados": list(engines_used),
                "confianca_media": round(avg_confidence, 2),
                "tempo_processamento_segundos": round(processing_time, 2),
                "total_caracteres": total_characters,
                "total_palavras": total_words,
                "paginas_com_texto": len([p for p in extracted_text if p['texto']])
            },
            "sucesso": True
        }
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        print(f"❌ Erro crítico no sistema híbrido: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Erro crítico no sistema híbrido: {str(e)}")
    
    finally:
        # Limpeza
        if input_pdf_path and os.path.exists(input_pdf_path):
            try:
                os.remove(input_pdf_path)
            except:
                pass

def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detecta o ângulo de inclinação do texto na imagem.
    """
    try:
        # Converter para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detectar bordas
        edges = cv2.Canny(thresh, 50, 150)
        
        # Detectar linhas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            if angles:
                # Usar mediana dos ângulos
                return np.median(angles)
        
        return 0.0
        
    except Exception as e:
        print(f"Erro na detecção de skew: {e}")
        return 0.0

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotaciona a imagem pelo ângulo especificado.
    """
    try:
        if abs(angle) < 0.5:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Criar matriz de rotação
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Aplicar rotação
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
        
    except Exception as e:
        print(f"Erro na rotação: {e}")
        return image

def calculate_average_confidence(data: Dict) -> float:
    """
    Calcula a confiança média dos dados do OCR.
    """
    try:
        if not data or 'conf' not in data:
            return 0.0
        
        confidences = [float(conf) for conf in data['conf'] if float(conf) > 0]
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
        
    except Exception as e:
        print(f"Erro no cálculo de confiança: {e}")
        return 0.0

def extract_text_with_easyocr_only(image):
    """
    Extrai texto usando apenas EasyOCR.
    """
    global easyocr_reader
    try:
        print("\n🔍 USANDO APENAS EASYOCR")
        
        # Inicialização LAZY do EasyOCR
        if not easyocr_reader and EASYOCR_AVAILABLE:
            try:
                global engines_loading
                engines_loading = True
                print("⏳ Inicializando EasyOCR sob demanda...")
                
                # Log dos diretórios antes da inicialização
                print("📁 Estado do cache ANTES da inicialização:")
                import easyocr
                print(f"  🗂️ EasyOCR module path: {easyocr.__file__}")
                
                # Verificar onde o EasyOCR salva por padrão
                easyocr_default_path = os.path.expanduser('~/.EasyOCR')
                print(f"  🗂️ EasyOCR default path: {easyocr_default_path}")
                
                if os.path.exists(easyocr_default_path):
                    files = os.listdir(easyocr_default_path)
                    print(f"  📂 {easyocr_default_path}: {len(files)} arquivos")
                
                # Garantir que o EasyOCR use nosso cache
                easyocr_cache_dir = '/app/.cache/easyocr'
                os.makedirs(easyocr_cache_dir, exist_ok=True)
                
                print(f"🔧 Forçando EasyOCR a usar: {easyocr_cache_dir}")
                easyocr_reader = easyocr.Reader(['pt', 'en'], gpu=False, model_storage_directory=easyocr_cache_dir)
                
                # Log dos diretórios APÓS a inicialização
                print("📁 Estado do cache APÓS a inicialização:")
                for cache_dir in ['/app/.cache', '/app/.cache/easyocr', easyocr_default_path]:
                    if os.path.exists(cache_dir):
                        files = os.listdir(cache_dir)
                        print(f"  📂 {cache_dir}: {len(files)} arquivos")
                        if files:
                            print(f"    📄 Primeiros arquivos: {files[:3]}")
                
                print("✅ EasyOCR inicializado com sucesso")
                engines_loading = False
            except Exception as e:
                engines_loading = False
                print(f"❌ Erro ao inicializar EasyOCR: {e}")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'EasyOCR',
                    'method': 'initialization_error'
                }
        
        if not easyocr_reader:
            print("❌ EasyOCR não está disponível")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'EasyOCR',
                'method': 'unavailable'
            }
        
        # Converter imagem para formato adequado
        try:
            if hasattr(image, 'convert'):  # PIL Image
                image_array = np.array(image)
            else:
                image_array = image
            
            # Reduzir tamanho da imagem se muito grande (economizar memória)
            height, width = image_array.shape[:2]
            max_dimension = 2000  # Limitar a 2000px
            
            if height > max_dimension or width > max_dimension:
                print(f"🔧 Redimensionando imagem de {width}x{height} para economizar memória...")
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                print(f"✅ Nova dimensão: {new_width}x{new_height}")
            
            print("🔄 Executando EasyOCR.readtext()...")
            start_time = time.time()
            easyocr_results = easyocr_reader.readtext(image_array)
            processing_time = time.time() - start_time
            print(f"✅ EasyOCR processou {len(easyocr_results) if easyocr_results else 0} regiões em {processing_time:.2f}s")
        except Exception as e:
            print(f"❌ Erro durante readtext: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'EasyOCR',
                'method': 'execution_error'
            }
        
        if easyocr_results:
            easyocr_text = ' '.join([result[1] for result in easyocr_results])
            easyocr_confidence = sum([result[2] for result in easyocr_results]) / len(easyocr_results) * 100
            
            print(f"✅ EasyOCR - Confiança: {easyocr_confidence:.1f}%")
            print(f"📝 Texto extraído (primeiros 200 chars): {easyocr_text[:200]}...")
            
            # Limpeza de memória
            del image_array, easyocr_results
            gc.collect()
            
            return {
                'text': easyocr_text,
                'confidence': easyocr_confidence,
                'engine': 'EasyOCR',
                'method': 'neural_network'
            }
        else:
            print("❌ EasyOCR - Nenhum texto detectado")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'EasyOCR',
                'method': 'no_text_detected'
            }
            
    except Exception as e:
        print(f"❌ EasyOCR - Erro: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'engine': 'EasyOCR',
            'method': 'error'
        }

def extract_text_with_trocr_only(image):
    """
    Extrai texto usando apenas TrOCR.
    """
    global trocr_processor, trocr_model
    try:
        print("\n🔍 USANDO APENAS TrOCR")
        
        # Inicialização LAZY do TrOCR
        if (not trocr_processor or not trocr_model) and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                global engines_loading
                engines_loading = True
                print("⏳ Inicializando TrOCR sob demanda...")
                trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                print("✅ TrOCR inicializado com sucesso")
                engines_loading = False
            except Exception as e:
                engines_loading = False
                print(f"❌ Erro ao inicializar TrOCR: {e}")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'TrOCR',
                    'method': 'initialization_error'
                }
        
        if not trocr_processor or not trocr_model:
            print("❌ TrOCR não está disponível")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'TrOCR',
                'method': 'unavailable'
            }
        
        from PIL import Image as PILImage
        
        # Converter para PIL Image adequadamente
        if isinstance(image, np.ndarray):
            # Se é numpy array, converter para PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        elif hasattr(image, 'convert'):  # Já é PIL Image
            pil_image = image
        else:
            # Tentar converter de qualquer outro tipo
            pil_image = PILImage.fromarray(np.array(image))
        
        # Garantir que está em RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pixel_values = trocr_processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        trocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if trocr_text.strip():
            # TrOCR não fornece confiança direta, estimamos baseado no tamanho e qualidade
            trocr_confidence = min(95.0, len(trocr_text.strip()) * 2.5)
            
            print(f"✅ TrOCR - Confiança estimada: {trocr_confidence:.1f}%")
            print(f"📝 Texto extraído (primeiros 200 chars): {trocr_text[:200]}...")
            
            return {
                'text': trocr_text,
                'confidence': trocr_confidence,
                'engine': 'TrOCR',
                'method': 'transformer'
            }
        else:
            print("❌ TrOCR - Nenhum texto detectado")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'TrOCR',
                'method': 'no_text_detected'
            }
            
    except Exception as e:
        print(f"❌ TrOCR - Erro: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'engine': 'TrOCR',
            'method': 'error'
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 