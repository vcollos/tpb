#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo principal do projeto vector-tpb.
Este arquivo coordena as operações de scraping, processamento e indexação
para o banco de dados vetorial sobre Transtorno de Personalidade Borderline.
"""

import os
import logging
import argparse
from scraping import scielo, pubmed, pepsic, medrxiv
from processing import text_cleaner, pdf_processor
from index import vector_store

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def scrape_all_sources(raw_dir, query_pt="transtorno de personalidade borderline", 
                      query_en="borderline personality disorder", max_articles=10):
    """
    Executa a raspagem de todas as fontes configuradas.
    
    Args:
        raw_dir: Diretório para salvar os dados brutos
        query_pt: Consulta em português
        query_en: Consulta em inglês
        max_articles: Número máximo de artigos por fonte
        
    Returns:
        Lista de caminhos para os arquivos extraídos
    """
    all_files = []
    
    # 1. SciELO (artigos em português e espanhol)
    logger.info("Iniciando scraping de artigos do SciELO")
    scielo_files = scielo.scrape_articles(
        query=query_pt, 
        max_articles=max_articles,
        output_dir=raw_dir
    )
    all_files.extend(scielo_files)
    
    # 2. PubMed Central (artigos em inglês)
    logger.info("Iniciando scraping de artigos do PubMed Central")
    pubmed_files = pubmed.scrape_articles(
        query=query_en,
        max_articles=max_articles,
        output_dir=raw_dir
    )
    all_files.extend(pubmed_files)
    
    # 3. PePSIC (artigos em português sobre psicologia)
    logger.info("Iniciando scraping de artigos do PePSIC")
    pepsic_files = pepsic.scrape_articles(
        query=query_pt,
        max_articles=max_articles,
        output_dir=raw_dir
    )
    all_files.extend(pepsic_files)
    
    # 4. medRxiv (preprints em inglês)
    logger.info("Iniciando scraping de preprints do medRxiv")
    medrxiv_files = medrxiv.scrape_articles(
        query=query_en,
        max_articles=max_articles,
        output_dir=raw_dir
    )
    all_files.extend(medrxiv_files)
    
    logger.info(f"Scraping concluído. Total de {len(all_files)} arquivos extraídos.")
    return all_files

def process_all_files(raw_dir, processed_dir):
    """
    Processa todos os arquivos extraídos.
    
    Args:
        raw_dir: Diretório com os dados brutos
        processed_dir: Diretório para salvar os dados processados
        
    Returns:
        Lista de caminhos para os arquivos processados
    """
    all_processed_files = []
    
    # 1. Processar arquivos de texto
    logger.info("Processando arquivos de texto")
    text_files = text_cleaner.process_documents(
        input_dir=raw_dir, 
        output_dir=processed_dir,
        file_pattern="*.txt"
    )
    all_processed_files.extend(text_files)
    
    # 2. Processar arquivos PDF
    logger.info("Processando arquivos PDF")
    pdf_files = pdf_processor.process_pdfs(
        input_dir=raw_dir,
        output_dir=processed_dir,
        file_pattern="*.pdf"
    )
    all_processed_files.extend(pdf_files)
    
    logger.info(f"Processamento concluído. Total de {len(all_processed_files)} arquivos processados.")
    return all_processed_files

def main():
    """Função principal que coordena o fluxo de trabalho."""
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Extração e indexação de artigos sobre TPB')
    parser.add_argument('--skip-scraping', action='store_true', help='Pular etapa de scraping')
    parser.add_argument('--skip-processing', action='store_true', help='Pular etapa de processamento')
    parser.add_argument('--skip-indexing', action='store_true', help='Pular etapa de indexação')
    parser.add_argument('--query-pt', default='transtorno de personalidade borderline', help='Consulta em português')
    parser.add_argument('--query-en', default='borderline personality disorder', help='Consulta em inglês')
    parser.add_argument('--max-articles', type=int, default=10, help='Número máximo de artigos por fonte')
    args = parser.parse_args()
    
    logger.info("Iniciando processo de extração e indexação")
    
    # Diretórios de dados
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    index_dir = os.path.join("data", "index")
    
    # Garantir que os diretórios existam
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    # Etapa 1: Scraping
    if not args.skip_scraping:
        logger.info("Iniciando scraping de artigos")
        scrape_all_sources(
            raw_dir=raw_dir,
            query_pt=args.query_pt,
            query_en=args.query_en,
            max_articles=args.max_articles
        )
    else:
        logger.info("Etapa de scraping ignorada")
    
    # Etapa 2: Processamento de texto
    if not args.skip_processing:
        logger.info("Processando textos extraídos")
        process_all_files(raw_dir=raw_dir, processed_dir=processed_dir)
    else:
        logger.info("Etapa de processamento ignorada")
    
    # Etapa 3: Indexação vetorial
    if not args.skip_indexing:
        logger.info("Criando índice vetorial")
        index_path = os.path.join(index_dir, "tpb_index.pkl")
        vector_store.create_index(
            input_dir=processed_dir,
            output_path=index_path
        )
    else:
        logger.info("Etapa de indexação ignorada")
    
    logger.info("Processo concluído com sucesso")

if __name__ == "__main__":
    main()