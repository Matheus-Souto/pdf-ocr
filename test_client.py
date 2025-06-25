#!/usr/bin/env python3
"""
Cliente de teste para a API de OCR
"""

import requests
import argparse
import os
import json
from typing import Optional

class PDFOCRClient:
    """
    Cliente para testar a API de OCR de PDFs
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> bool:
        """
        Verifica se a API está funcionando
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ API está funcionando!")
                print(f"Resposta: {response.json()}")
                return True
            else:
                print(f"❌ API retornou status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Não foi possível conectar à API. Certifique-se de que está rodando.")
            return False
        except Exception as e:
            print(f"❌ Erro ao verificar API: {e}")
            return False
    
    def convert_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> bool:
        """
        Converte um PDF para pesquisável
        """
        if not os.path.exists(pdf_path):
            print(f"❌ Arquivo não encontrado: {pdf_path}")
            return False
        
        if not output_path:
            name, ext = os.path.splitext(pdf_path)
            output_path = f"{name}_ocr{ext}"
        
        print(f"📄 Convertendo: {pdf_path}")
        print(f"💾 Salvando em: {output_path}")
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/convert-pdf/", files=files)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                
                print("✅ PDF convertido com sucesso!")
                print(f"📁 Arquivo salvo: {output_path}")
                return True
            else:
                print(f"❌ Erro na conversão: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"Detalhes: {error_info}")
                except:
                    print(f"Resposta: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante conversão: {e}")
            return False
    
    def extract_text(self, pdf_path: str, save_json: bool = False) -> bool:
        """
        Extrai texto de um PDF
        """
        if not os.path.exists(pdf_path):
            print(f"❌ Arquivo não encontrado: {pdf_path}")
            return False
        
        print(f"📄 Extraindo texto de: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/extract-text/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Texto extraído com sucesso!")
                print(f"📊 Total de páginas: {result['total_paginas']}")
                print(f"📝 Arquivo: {result['filename']}")
                print("\n" + "="*50)
                
                for page_data in result['texto_extraido']:
                    print(f"\n📄 Página {page_data['pagina']}:")
                    print("-" * 30)
                    if page_data['texto']:
                        print(page_data['texto'][:500])  # Primeiros 500 caracteres
                        if len(page_data['texto']) > 500:
                            print("... (texto truncado)")
                    else:
                        print("(Nenhum texto encontrado nesta página)")
                
                if save_json:
                    json_path = pdf_path.replace('.pdf', '_extracted_text.json')
                    with open(json_path, 'w', encoding='utf-8') as json_file:
                        json.dump(result, json_file, ensure_ascii=False, indent=2)
                    print(f"\n💾 Texto salvo em JSON: {json_path}")
                
                return True
            else:
                print(f"❌ Erro na extração: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"Detalhes: {error_info}")
                except:
                    print(f"Resposta: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante extração: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Cliente de teste para API de OCR")
    parser.add_argument('--url', default='http://localhost:8000', help='URL da API')
    parser.add_argument('--health', action='store_true', help='Verificar saúde da API')
    parser.add_argument('--convert', help='Converter PDF para pesquisável')
    parser.add_argument('--extract', help='Extrair texto do PDF')
    parser.add_argument('--output', help='Arquivo de saída para conversão')
    parser.add_argument('--save-json', action='store_true', help='Salvar texto extraído em JSON')
    
    args = parser.parse_args()
    
    if not any([args.health, args.convert, args.extract]):
        parser.print_help()
        return
    
    client = PDFOCRClient(args.url)
    
    print("🔧 Cliente de Teste - API de OCR para PDFs")
    print("=" * 50)
    
    if args.health:
        client.health_check()
    
    if args.convert:
        client.convert_pdf(args.convert, args.output)
    
    if args.extract:
        client.extract_text(args.extract, args.save_json)

if __name__ == "__main__":
    main() 