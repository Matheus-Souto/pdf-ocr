#!/usr/bin/env python3
"""
Cliente Python para demonstrar o uso do endpoint de progresso da API OCR.
Mostra como receber atualizações de progresso em tempo real.
"""

import requests
import json
import sys
import os
from pathlib import Path

def extract_text_with_progress(api_url: str, pdf_file_path: str, enhancement_level: str = "medium", resolution_scale: float = 2.0):
    """
    Extrai texto de um PDF com indicador de progresso.
    
    Args:
        api_url: URL da API
        pdf_file_path: Caminho para o arquivo PDF
        enhancement_level: Nível de pré-processamento ("basic", "medium", "aggressive", "ultra")
        resolution_scale: Escala de resolução (1.0-4.0)
    """
    
    # Verificar se o arquivo existe
    pdf_path = Path(pdf_file_path)
    if not pdf_path.exists():
        print(f"❌ Erro: Arquivo '{pdf_file_path}' não encontrado.")
        return None
    
    print(f"📄 Processando: {pdf_path.name}")
    print(f"🔧 Configurações:")
    print(f"   • Nível de processamento: {enhancement_level}")
    print(f"   • Escala de resolução: {resolution_scale}x")
    print("─" * 50)
    
    try:
        # Preparar dados para envio
        with open(pdf_path, 'rb') as file:
            files = {'file': file}
            data = {
                'enhancement_level': enhancement_level,
                'resolution_scale': str(resolution_scale)
            }
            
            # Fazer requisição com streaming
            response = requests.post(
                f"{api_url}/extract-text-progress/",
                files=files,
                data=data,
                stream=True
            )
            
            if response.status_code != 200:
                print(f"❌ Erro HTTP {response.status_code}: {response.text}")
                return None
            
            # Processar resposta em streaming
            extracted_data = None
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        json_data = json.loads(line[6:])  # Remove 'data: '
                        extracted_data = process_message(json_data)
                        if extracted_data and json_data.get('tipo') == 'concluido':
                            break
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Erro ao decodificar JSON: {e}")
                        continue
            
            return extracted_data
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de conexão: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return None

def save_extracted_text(data: dict, output_file: str = None):
    """
    Salva o texto extraído em arquivo.
    
    Args:
        data: Dados retornados pela API
        output_file: Arquivo de saída (opcional)
    """
    
    if not data or 'resultados' not in data:
        print("❌ Nenhum dado para salvar")
        return
    
    # Nome do arquivo de saída
    if not output_file:
        filename = data['arquivo']['nome'].replace('.pdf', '_extracted.txt')
        output_file = f"extracted_{filename}"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write(f"# Texto Extraído de: {data['arquivo']['nome']}\n")
            f.write(f"# Tamanho do arquivo: {data['arquivo']['tamanho_mb']}MB\n")
            f.write(f"# Total de páginas: {data['processamento']['total_paginas']}\n")
            f.write(f"# Total de caracteres: {data['estatisticas']['total_caracteres']:,}\n")
            f.write(f"# Total de palavras: {data['estatisticas']['total_palavras']:,}\n")
            f.write(f"# Total de linhas: {data['estatisticas']['total_linhas']:,}\n")
            f.write(f"# Páginas com texto: {data['estatisticas']['paginas_com_texto']}\n")
            f.write(f"# Páginas vazias: {data['estatisticas']['paginas_vazias']}\n")
            f.write(f"# Média caracteres/página: {data['estatisticas']['media_caracteres_por_pagina']:.1f}\n")
            
            if data['resultados']['resumo_conteudo']['pagina_mais_longa']:
                f.write(f"# Página mais longa: {data['resultados']['resumo_conteudo']['pagina_mais_longa']}\n")
            if data['resultados']['resumo_conteudo']['pagina_mais_curta']:
                f.write(f"# Página mais curta: {data['resultados']['resumo_conteudo']['pagina_mais_curta']}\n")
                
            f.write("=" * 80 + "\n\n")
            
            # Conteúdo por página
            for page_data in data['resultados']['texto_por_pagina']:
                f.write(f"=== PÁGINA {page_data['pagina']} ===\n")
                f.write(f"Estatísticas: {page_data['estatisticas']['caracteres']} caracteres, ")
                f.write(f"{page_data['estatisticas']['palavras']} palavras, ")
                f.write(f"{page_data['estatisticas']['linhas']} linhas\n")
                f.write("-" * 40 + "\n")
                f.write(page_data['texto'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"💾 Texto salvo em: {output_file}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")

def main():
    """
    Função principal do cliente com interface de linha de comando melhorada.
    """
    print("🔍 Cliente PDF OCR com Progresso em Tempo Real")
    print("=" * 60)
    print("📌 Uso: python client_progress.py <arquivo.pdf> [enhancement_level] [resolution_scale]")
    print()
    
    if len(sys.argv) < 2:
        print("❌ Erro: Por favor, forneça o caminho do arquivo PDF")
        print()
        print("📋 Exemplos de uso:")
        print("  python client_progress.py documento.pdf")
        print("  python client_progress.py documento.pdf medium 2.0")
        print("  python client_progress.py documento.pdf ultra 3.0")
        print()
        print("📚 Parâmetros disponíveis:")
        print("  enhancement_level: basic, medium, aggressive, ultra (padrão: medium)")
        print("  resolution_scale: 1.0-4.0 (padrão: 2.0)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Configurações padrão
    enhancement_level = "medium"
    resolution_scale = 2.0
    
    # Processar argumentos opcionais
    if len(sys.argv) >= 3:
        enhancement_level = sys.argv[2]
    if len(sys.argv) >= 4:
        try:
            resolution_scale = float(sys.argv[3])
        except ValueError:
            print("❌ Erro: resolution_scale deve ser um número decimal (ex: 2.0)")
            sys.exit(1)
    
    # Validar parâmetros
    if not os.path.exists(pdf_path):
        print(f"❌ Erro: Arquivo '{pdf_path}' não encontrado!")
        sys.exit(1)
    
    if enhancement_level not in ["basic", "medium", "aggressive", "ultra"]:
        print("❌ Erro: enhancement_level deve ser 'basic', 'medium', 'aggressive' ou 'ultra'")
        sys.exit(1)
    
    if not 1.0 <= resolution_scale <= 4.0:
        print("❌ Erro: resolution_scale deve estar entre 1.0 e 4.0")
        sys.exit(1)
    
    # Executar extração
    extracted_data = extract_text_with_progress(API_URL, pdf_path, enhancement_level, resolution_scale)
    
    if extracted_data:
        print("\n" + "=" * 50)
        print("💾 Deseja salvar o texto extraído em um arquivo?")
        save_choice = input("Digite 's' para sim, ou Enter para não: ").lower().strip()
        
        if save_choice == 's':
            output_file = input("Nome do arquivo (ou Enter para nome automático): ").strip()
            if not output_file:
                output_file = None
            
            save_extracted_text(extracted_data, output_file)
        else:
            print("✅ Extração concluída. Texto não foi salvo.")
    else:
        print("❌ Falha na extração do texto.")
        sys.exit(1)

if __name__ == "__main__":
    main() 