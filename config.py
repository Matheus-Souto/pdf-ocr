import os
from typing import Optional

class Settings:
    """
    Configurações da API de OCR
    """
    
    # Configurações do Tesseract
    TESSERACT_CMD: Optional[str] = None  # Será detectado automaticamente ou definido manualmente
    OCR_LANGUAGE: str = "por"  # Idioma padrão para OCR (português)
    
    # Configurações de qualidade
    OCR_RESOLUTION_MATRIX: float = 2.0  # Multiplicador de resolução (maior = melhor qualidade, mais lento)
    
    # Configurações do servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Configurações de arquivos temporários
    TEMP_DIR: str = "temp"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB máximo
    
    # Configurações de OCR avançadas
    OCR_CONFIG: str = '--oem 3 --psm 6'  # Configurações do Tesseract
    
    def __init__(self):
        # Criar diretório temporário se não existir
        if not os.path.exists(self.TEMP_DIR):
            os.makedirs(self.TEMP_DIR)
    
    def get_tesseract_cmd(self) -> Optional[str]:
        """
        Retorna o caminho do comando Tesseract
        """
        if self.TESSERACT_CMD:
            return self.TESSERACT_CMD
        
        # Tentar detectar automaticamente
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', ''))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return None  # Deixar o pytesseract detectar automaticamente

# Instância global das configurações
settings = Settings() 