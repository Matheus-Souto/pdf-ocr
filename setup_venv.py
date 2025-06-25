#!/usr/bin/env python3
"""
Script para configurar ambiente virtual e instalar dependências
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário. Versão atual:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def create_virtual_environment():
    """Cria o ambiente virtual"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Ambiente virtual já existe")
        return True
    
    print("🔨 Criando ambiente virtual...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("✅ Ambiente virtual criado com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao criar ambiente virtual: {e}")
        return False

def get_activation_command():
    """Retorna o comando de ativação do venv para o sistema operacional"""
    system = platform.system().lower()
    
    if system == "windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def get_python_executable():
    """Retorna o caminho do executável Python no venv"""
    system = platform.system().lower()
    
    if system == "windows":
        return Path("venv/Scripts/python.exe")
    else:
        return Path("venv/bin/python")

def install_dependencies_in_venv():
    """Instala as dependências no ambiente virtual"""
    python_exec = get_python_executable()
    
    if not python_exec.exists():
        print("❌ Executável Python não encontrado no venv")
        return False
    
    print("📦 Instalando dependências no ambiente virtual...")
    try:
        subprocess.check_call([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(python_exec), "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def check_tesseract_in_venv():
    """Verifica se o Tesseract está acessível no venv"""
    python_exec = get_python_executable()
    
    print("🔍 Verificando Tesseract OCR...")
    try:
        result = subprocess.run([
            str(python_exec), "-c", 
            "import pytesseract; print(pytesseract.get_tesseract_version())"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Tesseract encontrado: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Tesseract não encontrado: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar Tesseract: {e}")
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
        print("3. Certifique-se de adicionar ao PATH do sistema")
        print("4. Ou descomente e ajuste a linha no main.py:")
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

def create_activation_scripts():
    """Cria scripts de ativação personalizados"""
    system = platform.system().lower()
    
    if system == "windows":
        # Script para Windows (PowerShell e CMD)
        activate_ps1 = """# Ativar ambiente virtual - PowerShell
Write-Host "🚀 Ativando ambiente virtual..." -ForegroundColor Green
& .\\venv\\Scripts\\Activate.ps1
Write-Host "✅ Ambiente virtual ativado!" -ForegroundColor Green
Write-Host "Para executar a API: python main.py" -ForegroundColor Yellow
"""
        
        activate_cmd = """@echo off
echo 🚀 Ativando ambiente virtual...
call venv\\Scripts\\activate.bat
echo ✅ Ambiente virtual ativado!
echo Para executar a API: python main.py
cmd /k
"""
        
        with open("ativar_venv.ps1", "w", encoding="utf-8") as f:
            f.write(activate_ps1)
        
        with open("ativar_venv.cmd", "w", encoding="utf-8") as f:
            f.write(activate_cmd)
            
        print("✅ Scripts de ativação criados:")
        print("   - ativar_venv.ps1 (PowerShell)")
        print("   - ativar_venv.cmd (CMD)")
        
    else:
        # Script para Linux/macOS
        activate_sh = """#!/bin/bash
echo "🚀 Ativando ambiente virtual..."
source venv/bin/activate
echo "✅ Ambiente virtual ativado!"
echo "Para executar a API: python main.py"
exec $SHELL
"""
        
        with open("ativar_venv.sh", "w") as f:
            f.write(activate_sh)
        
        # Tornar executável
        os.chmod("ativar_venv.sh", 0o755)
        
        print("✅ Script de ativação criado: ativar_venv.sh")

def create_temp_directory():
    """Cria o diretório temporário"""
    temp_dir = Path("temp")
    if not temp_dir.exists():
        temp_dir.mkdir()
        print("✅ Diretório temporário criado")
    else:
        print("✅ Diretório temporário já existe")

def show_usage_instructions():
    """Mostra instruções de uso"""
    system = platform.system().lower()
    activation_cmd = get_activation_command()
    
    print("\n" + "=" * 60)
    print("📚 COMO USAR O AMBIENTE VIRTUAL:")
    print("=" * 60)
    
    print("\n1️⃣ ATIVAR O AMBIENTE VIRTUAL:")
    if system == "windows":
        print("   PowerShell: .\\ativar_venv.ps1")
        print("   CMD:        ativar_venv.cmd")
        print("   Manual:     venv\\Scripts\\activate")
    else:
        print("   Automático: ./ativar_venv.sh")
        print("   Manual:     source venv/bin/activate")
    
    print("\n2️⃣ EXECUTAR A API:")
    print("   python main.py")
    
    print("\n3️⃣ TESTAR A API:")
    print("   python test_client.py --health")
    print("   python test_client.py --convert seu_arquivo.pdf")
    
    print("\n4️⃣ DESATIVAR O AMBIENTE VIRTUAL:")
    print("   deactivate")
    
    print("\n5️⃣ ACESSAR DOCUMENTAÇÃO:")
    print("   http://localhost:8000/docs")

def main():
    print("🔧 CONFIGURAÇÃO DE AMBIENTE VIRTUAL - API de OCR")
    print("=" * 60)
    
    # Verificar versão do Python
    if not check_python_version():
        sys.exit(1)
    
    # Criar ambiente virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalar dependências no venv
    if not install_dependencies_in_venv():
        sys.exit(1)
    
    # Criar diretório temporário
    create_temp_directory()
    
    # Criar scripts de ativação
    create_activation_scripts()
    
    # Verificar Tesseract
    tesseract_ok = check_tesseract_in_venv()
    
    if not tesseract_ok:
        show_tesseract_installation_guide()
    
    # Mostrar instruções
    show_usage_instructions()
    
    print("\n" + "=" * 60)
    if tesseract_ok:
        print("🎉 CONFIGURAÇÃO COMPLETA!")
        print("Ambiente virtual configurado e pronto para uso!")
    else:
        print("⚠️  CONFIGURAÇÃO PARCIALMENTE COMPLETA")
        print("Instale o Tesseract OCR e teste novamente")
    
    print("\n💡 PRÓXIMO PASSO:")
    system = platform.system().lower()
    if system == "windows":
        print("Execute: .\\ativar_venv.ps1 (PowerShell) ou ativar_venv.cmd (CMD)")
    else:
        print("Execute: ./ativar_venv.sh")

if __name__ == "__main__":
    main() 