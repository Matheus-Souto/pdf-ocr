#!/usr/bin/env python3
"""
Script de setup para a API de OCR
"""

import os
import sys
import platform
import subprocess
import requests
from pathlib import Path

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário. Versão atual:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Instala as dependências do projeto"""
    print("\n📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False

def check_tesseract():
    """Verifica se o Tesseract está instalado"""
    print("\n🔍 Verificando Tesseract OCR...")
    
    try:
        import pytesseract
        # Tenta executar o Tesseract
        pytesseract.get_tesseract_version()
        print("✅ Tesseract encontrado e funcionando!")
        return True
    except Exception as e:
        print(f"❌ Tesseract não encontrado: {e}")
        return False

def show_tesseract_installation_guide():
    """Mostra o guia de instalação do Tesseract"""
    system = platform.system().lower()
    
    print("\n📖 GUIA DE INSTALAÇÃO DO TESSERACT:")
    print("=" * 50)
    
    if system == "windows":
        print("🪟 WINDOWS:")
        print("1. Baixe o instalador em:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Execute o instalador como administrador")
        print("3. Adicione ao PATH ou descomente a linha no main.py:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        
    elif system == "linux":
        print("🐧 LINUX (Ubuntu/Debian):")
        print("sudo apt update")
        print("sudo apt install tesseract-ocr tesseract-ocr-por")
        print("\n🐧 LINUX (CentOS/RHEL/Fedora):")
        print("sudo yum install tesseract tesseract-langpack-por")
        print("# ou")
        print("sudo dnf install tesseract tesseract-langpack-por")
        
    elif system == "darwin":
        print("🍎 macOS:")
        print("brew install tesseract tesseract-lang")
        
    print("\n💡 DICA: Certifique-se de instalar o pacote de idioma português!")

def create_temp_directory():
    """Cria o diretório temporário"""
    temp_dir = Path("temp")
    if not temp_dir.exists():
        temp_dir.mkdir()
        print("✅ Diretório temporário criado")
    else:
        print("✅ Diretório temporário já existe")

def test_api_dependencies():
    """Testa se todas as dependências da API estão funcionando"""
    print("\n🧪 Testando dependências...")
    
    try:
        import fastapi
        print("✅ FastAPI")
    except ImportError:
        print("❌ FastAPI não encontrado")
        return False
    
    try:
        import fitz
        print("✅ PyMuPDF")
    except ImportError:
        print("❌ PyMuPDF não encontrado")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError:
        print("❌ Pillow não encontrado")
        return False
    
    try:
        import pytesseract
        print("✅ pytesseract")
    except ImportError:
        print("❌ pytesseract não encontrado")
        return False
    
    try:
        import uvicorn
        print("✅ uvicorn")
    except ImportError:
        print("❌ uvicorn não encontrado")
        return False
    
    return True

def main():
    print("🔧 SETUP - API de OCR para PDFs")
    print("=" * 50)
    
    # Verificar versão do Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependências
    if not install_dependencies():
        print("\n❌ Falha na instalação das dependências")
        sys.exit(1)
    
    # Criar diretório temporário
    create_temp_directory()
    
    # Testar dependências
    if not test_api_dependencies():
        print("\n❌ Algumas dependências não foram instaladas corretamente")
        sys.exit(1)
    
    # Verificar Tesseract
    tesseract_ok = check_tesseract()
    
    if not tesseract_ok:
        show_tesseract_installation_guide()
        print("\n⚠️  Instale o Tesseract e execute este script novamente")
    
    print("\n" + "=" * 50)
    if tesseract_ok:
        print("🎉 SETUP COMPLETO!")
        print("Para iniciar a API, execute:")
        print("   python main.py")
        print("\nPara testar a API, execute:")
        print("   python test_client.py --health")
    else:
        print("⚠️  SETUP PARCIALMENTE COMPLETO")
        print("Instale o Tesseract OCR para usar a API")
    
    print("\nDocumentação completa: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 