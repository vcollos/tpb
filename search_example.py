#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de exemplo para demonstrar como realizar buscas no índice vetorial.
"""

import os
import logging
import argparse
from index.vector_store import search_index

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Função principal para demonstrar buscas no índice vetorial."""
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Busca semântica no índice vetorial de TPB')
    parser.add_argument('--query', required=True, help='Consulta para busca semântica')
    parser.add_argument('--index-path', default='data/index/tpb_index.pkl', help='Caminho para o arquivo de índice')
    parser.add_argument('--top-k', type=int, default=5, help='Número de resultados a retornar')
    parser.add_argument('--output', help='Arquivo para salvar os resultados (opcional)')
    args = parser.parse_args()
    
    # Verificar se o índice existe
    if not os.path.exists(args.index_path):
        logger.error(f"Índice não encontrado: {args.index_path}")
        logger.info("Execute primeiro o script main.py para criar o índice.")
        return
    
    logger.info(f"Realizando busca semântica para: '{args.query}'")
    
    # Realizar a busca
    results = search_index(
        query=args.query,
        index_path=args.index_path,
        top_k=args.top_k
    )
    
    # Exibir resultados
    print("\n" + "="*80)
    print(f"RESULTADOS PARA: '{args.query}'")
    print("="*80)
    
    if not results:
        print("\nNenhum resultado encontrado.")
    else:
        # Preparar resultados para exibição e possível salvamento
        output_lines = []
        
        for i, (doc, score) in enumerate(results, 1):
            # Cabeçalho do resultado
            header = f"\n[{i}] Score: {score:.4f} - {doc.id}"
            print(header)
            output_lines.append(header)
            
            # Metadados
            if doc.metadata:
                metadata_str = "\nMetadados:"
                print(metadata_str)
                output_lines.append(metadata_str)
                
                for key, value in doc.metadata.items():
                    meta_line = f"  - {key}: {value}"
                    print(meta_line)
                    output_lines.append(meta_line)
            
            # Texto (limitado a 300 caracteres para exibição)
            preview = doc.text[:300] + "..." if len(doc.text) > 300 else doc.text
            text_str = f"\nTexto: {preview}"
            print(text_str)
            output_lines.append(text_str)
            
            # Separador
            separator = "\n" + "-"*80
            print(separator)
            output_lines.append(separator)
        
        # Salvar resultados em arquivo se solicitado
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write("\n".join(output_lines))
                logger.info(f"Resultados salvos em: {args.output}")
            except Exception as e:
                logger.error(f"Erro ao salvar resultados: {str(e)}")
    
    # Sugestões de consultas relacionadas
    print("\nSugestões de consultas relacionadas:")
    
    # Estas são apenas sugestões genéricas - em uma implementação mais avançada,
    # poderiam ser geradas dinamicamente com base na consulta original
    related_queries = [
        "tratamentos para transtorno de personalidade borderline",
        "sintomas do transtorno borderline",
        "comorbidades associadas ao TPB",
        "terapia dialética comportamental para borderline",
        "neurobiologia do transtorno de personalidade borderline"
    ]
    
    for query in related_queries:
        print(f"  - {query}")

def interactive_search():
    """Modo interativo para realizar múltiplas buscas."""
    index_path = input("Caminho para o arquivo de índice [data/index/tpb_index.pkl]: ") or "data/index/tpb_index.pkl"
    
    if not os.path.exists(index_path):
        print(f"Índice não encontrado: {index_path}")
        print("Execute primeiro o script main.py para criar o índice.")
        return
    
    print("\n" + "="*80)
    print("MODO DE BUSCA INTERATIVA - Digite 'sair' para encerrar")
    print("="*80)
    
    while True:
        query = input("\nDigite sua consulta: ")
        if query.lower() in ['sair', 'exit', 'quit']:
            break
        
        top_k = input("Número de resultados [5]: ") or "5"
        try:
            top_k = int(top_k)
        except ValueError:
            print("Valor inválido. Usando 5 como padrão.")
            top_k = 5
        
        # Realizar a busca
        results = search_index(
            query=query,
            index_path=index_path,
            top_k=top_k
        )
        
        # Exibir resultados
        print("\n" + "="*80)
        print(f"RESULTADOS PARA: '{query}'")
        print("="*80)
        
        if not results:
            print("\nNenhum resultado encontrado.")
        else:
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n[{i}] Score: {score:.4f} - {doc.id}")
                
                if doc.metadata:
                    print("\nMetadados:")
                    for key, value in doc.metadata.items():
                        print(f"  - {key}: {value}")
                
                preview = doc.text[:300] + "..." if len(doc.text) > 300 else doc.text
                print(f"\nTexto: {preview}")
                print("\n" + "-"*80)

if __name__ == "__main__":
    # Verificar se há argumentos de linha de comando
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Se não houver argumentos, iniciar modo interativo
        interactive_search()