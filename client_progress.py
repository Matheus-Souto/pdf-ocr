#!/usr/bin/env python3
"""
Cliente Python para demonstrar o uso do endpoint de progresso da API OCR.
Mostra como receber atualizações de progresso em tempo real.
"""

import requests
import json
import sys
from pathlib import Path

def extract_text_with_progress(api_url: str, pdf_file_path: str):
    """
    Extrai texto de PDF com indicador de progresso em tempo real.
    
    Args:
        api_url: URL base da API (ex: http://localhost:8000)
        pdf_file_path: Caminho para o arquivo PDF
    """
    
    # Verificar se arquivo existe
    pdf_path = Path(pdf_file_path)
    if not pdf_path.exists():
        print(f"❌ Arquivo não encontrado: {pdf_file_path}")
        return None
    
    print(f"📄 Iniciando extração de texto: {pdf_path.name}")
    print("=" * 60)
    
    # Preparar arquivo para upload
    files = {'file': ('document.pdf', open(pdf_path, 'rb'), 'application/pdf')}
    
    try:
        # Fazer requisição com streaming
        response = requests.post(
            f"{api_url}/extract-text-progress/",
            files=files,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ Erro na requisição: {response.status_code}")
            return None
        
        # Processar stream de dados
        extracted_data = None
        total_chars = 0
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    # Parse JSON data
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    
                    # Processar diferentes tipos de mensagem
                    if data['tipo'] == 'info':
                        print(f"📋 Arquivo: {data['arquivo']['nome']} ({data['arquivo']['tamanho_mb']}MB)")
                        print(f"📊 Total de páginas: {data['processamento']['total_paginas']}")
                        print()
                        
                    elif data['tipo'] == 'progresso':
                        # Mostrar barra de progresso
                        progress = data['processamento']['progresso_percent']
                        bar_length = 40
                        filled_length = int(bar_length * progress // 100)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)
                        
                        print(f"\r🔄 [{bar}] {progress}% - {data['processamento']['status']}", end='', flush=True)
                        
                    elif data['tipo'] == 'pagina_concluida':
                        total_chars = data['estatisticas_gerais']['total_caracteres']
                        print(f"\n✅ Página {data['pagina']['numero']} concluída")
                        print(f"   📝 Caracteres: {data['resultado']['caracteres_pagina']}")
                        print(f"   📄 Linhas: {data['resultado']['linhas_pagina']}")
                        print(f"   🔤 Palavras: {data['resultado']['palavras_pagina']}")
                        print(f"   📊 Total geral: {total_chars} caracteres")
                        
                    elif data['tipo'] == 'concluido':
                        print(f"\n\n🎉 Extração concluída!")
                        print("=" * 50)
                        print(f"📄 Arquivo: {data['arquivo']['nome']}")
                        print(f"💾 Tamanho: {data['arquivo']['tamanho_mb']}MB")
                        print(f"📊 Páginas processadas: {data['processamento']['total_paginas']}")
                        print(f"📝 Total de caracteres: {data['estatisticas']['total_caracteres']:,}")
                        print(f"🔤 Total de palavras: {data['estatisticas']['total_palavras']:,}")
                        print(f"📄 Total de linhas: {data['estatisticas']['total_linhas']:,}")
                        print(f"📑 Páginas com texto: {data['estatisticas']['paginas_com_texto']}")
                        print(f"📄 Páginas vazias: {data['estatisticas']['paginas_vazias']}")
                        print(f"📊 Média: {data['estatisticas']['media_caracteres_por_pagina']:.1f} caracteres/página")
                        
                        if data['resultados']['resumo_conteudo']['pagina_mais_longa']:
                            print(f"🏆 Página mais longa: {data['resultados']['resumo_conteudo']['pagina_mais_longa']}")
                        if data['resultados']['resumo_conteudo']['pagina_mais_curta']:
                            print(f"🎯 Página mais curta: {data['resultados']['resumo_conteudo']['pagina_mais_curta']}")
                        
                        extracted_data = data
                        break
                        
                    elif data['tipo'] == 'erro':
                        print(f"\n❌ Erro: {data['erro']['mensagem']}")
                        print(f"🔍 Tipo: {data['erro']['tipo_erro']}")
                        if data['erro']['detalhes']:
                            print(f"💡 Dica: {data['erro']['detalhes']}")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Erro ao processar dados: {e}")
                    continue
        
        return extracted_data
        
    except requests.RequestException as e:
        print(f"❌ Erro na requisição: {e}")
        return None
    finally:
        # Fechar arquivo
        files['file'][1].close()

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
    """Função principal do cliente."""
    
    # Configurações
    API_URL = "http://localhost:8000"
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("📋 Uso: python client_progress.py <arquivo_pdf> [url_api]")
        print("📋 Exemplo: python client_progress.py documento.pdf")
        print("📋 Exemplo: python client_progress.py documento.pdf http://localhost:8000")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else API_URL
    
    print("🚀 Cliente de Extração de Texto com Progresso")
    print("=" * 60)
    print(f"🌐 API: {api_url}")
    print(f"📄 Arquivo: {pdf_file}")
    print()
    
    # Extrair texto
    result = extract_text_with_progress(api_url, pdf_file)
    
    if result:
        print("\n" + "=" * 60)
        
        # Perguntar se quer salvar
        save_choice = input("\n💾 Deseja salvar o texto extraído? (s/N): ").strip().lower()
        
        if save_choice in ['s', 'sim', 'y', 'yes']:
            output_file = input("📁 Nome do arquivo (Enter para automático): ").strip()
            save_extracted_text(result, output_file if output_file else None)
        
        # Mostrar resumo das primeiras páginas
        print("\n📋 Resumo do conteúdo extraído:")
        print("-" * 40)
        
        for i, page_data in enumerate(result['resultados']['texto_por_pagina'][:3]):  # Primeiras 3 páginas
            texto = page_data['texto'][:200]  # Primeiros 200 caracteres
            if len(page_data['texto']) > 200:
                texto += "..."
            
            print(f"\n📄 Página {page_data['pagina']}:")
            print(f"   {len(page_data['texto'])} caracteres")
            print(f"   Prévia: {texto}")
        
        if len(result['resultados']['texto_por_pagina']) > 3:
            print(f"\n   ... e mais {len(result['resultados']['texto_por_pagina']) - 3} páginas")
    
    print("\n✨ Processamento concluído!")

if __name__ == "__main__":
    main() 