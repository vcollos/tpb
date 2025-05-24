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
                      query_en="borderline personality disorder", max_articles=10, pubmed_api_key=None):
    """
    Executa a raspagem de todas as fontes configuradas.
    
    Args:
        raw_dir: Diretório para salvar os dados brutos
        query_pt: Consulta em português
        query_en: Consulta em inglês
        max_articles: Número máximo de artigos por fonte
        
    Returns:
        Lista de dicionários, cada um detalhando um arquivo extraído: {'path': file_path, 'lang': lang_code, 'type': 'txt'/'pdf'}
    """
    all_files_details = [] 
    
    # 1. SciELO (artigos em português e espanhol)
    logger.info(f"Iniciando scraping do SciELO com query: '{query_pt}', max_articles: {max_articles}")
    try:
        scielo_files = scielo.scrape_articles(
            query=query_pt, 
            max_articles=max_articles,
            output_dir=raw_dir
        )
        all_files_details.extend(scielo_files)
        logger.info(f"SciELO scraping concluído. {len(scielo_files)} itens extraídos.")
    except Exception as e:
        logger.exception(f"Erro durante o scraping do SciELO com query '{query_pt}'.")

    # 2. PubMed Central (artigos em inglês)
    logger.info(f"Iniciando scraping do PubMed Central com query: '{query_en}', max_articles: {max_articles}, API Key: {'Sim' if pubmed_api_key else 'Não'}")
    try:
        pubmed_files = pubmed.scrape_articles(
            query=query_en,
            max_articles=max_articles,
            output_dir=raw_dir,
            api_key=pubmed_api_key # Pass the key here
        )
        all_files_details.extend(pubmed_files)
        logger.info(f"PubMed Central scraping concluído. {len(pubmed_files)} itens extraídos.")
    except Exception as e:
        logger.exception(f"Erro durante o scraping do PubMed Central com query '{query_en}'.")
    
    # 3. PePSIC (artigos em português sobre psicologia)
    logger.info(f"Iniciando scraping do PePSIC com query: '{query_pt}', max_articles: {max_articles}")
    try:
        pepsic_files = pepsic.scrape_articles(
            query=query_pt,
            max_articles=max_articles,
            output_dir=raw_dir
        )
        all_files_details.extend(pepsic_files)
        logger.info(f"PePSIC scraping concluído. {len(pepsic_files)} itens extraídos.")
    except Exception as e:
        logger.exception(f"Erro durante o scraping do PePSIC com query '{query_pt}'.")
    
    # 4. medRxiv (preprints em inglês)
    logger.info(f"Iniciando scraping do medRxiv com query: '{query_en}', max_articles: {max_articles}")
    try:
        medrxiv_files = medrxiv.scrape_articles(
            query=query_en,
            max_articles=max_articles,
            output_dir=raw_dir
        )
        all_files_details.extend(medrxiv_files)
        logger.info(f"medRxiv scraping concluído. {len(medrxiv_files)} itens extraídos.")
    except Exception as e:
        logger.exception(f"Erro durante o scraping do medRxiv com query '{query_en}'.")
    
    logger.info(f"Scraping de todas as fontes concluído. Total de {len(all_files_details)} itens (arquivos de texto/PDF) extraídos.")
    return all_files_details

def process_all_files(raw_files_details: list, processed_dir: str) -> list:
    """
    Processa arquivos PDF e texto extraídos, limpando os textos.
    
    Args:
        raw_files_details: Lista de dicionários {'path': file_path, 'lang': lang_code, 'type': 'txt'/'pdf'}
        processed_dir: Diretório para salvar os arquivos de texto processados e metadados JSON.
        
    Returns:
        Lista de caminhos para os arquivos de texto finais e limpos no processed_dir.
    """
    all_processed_text_files_details = [] # Stores {'path': path_to_txt_in_processed_or_raw, 'lang': lang_code}

    # Certificar que o diretório de processados existe
    os.makedirs(processed_dir, exist_ok=True)

    pdf_files_to_process = [item for item in raw_files_details if item['type'] == 'pdf']
    raw_text_files_to_process = [item for item in raw_files_details if item['type'] == 'txt']

    logger.info(f"Identificados {len(pdf_files_to_process)} PDFs e {len(raw_text_files_to_process)} arquivos de texto brutos para processamento.")

    # 1. Processar arquivos PDF
    if pdf_files_to_process:
        logger.info(f"Iniciando processamento de {len(pdf_files_to_process)} arquivos PDF...")
        pdf_processor_instance = pdf_processor.PDFProcessor()
        for item in pdf_files_to_process:
            logger.debug(f"Processando PDF: {item['path']} (idioma: {item['lang']})")
            try:
                processed_data = pdf_processor_instance.process_pdf(
                    pdf_path=item['path'], 
                    output_dir=processed_dir
                )
                # The process_pdf method now handles logging if text extraction fails
                if processed_data and processed_data.get('text'): 
                    base_name = os.path.splitext(os.path.basename(item['path']))[0]
                    output_text_file = os.path.join(processed_dir, f"{base_name}.txt") 
                    all_processed_text_files_details.append({'path': output_text_file, 'lang': item['lang']})
                    logger.info(f"PDF {item['path']} processado. Texto salvo em: {output_text_file}")
                # No 'else' here as process_pdf logs the error of non-extraction.
                # We just don't add it to the list for further cleaning if text is missing.
            except Exception as e:
                logger.exception(f"Erro inesperado ao processar PDF: {item['path']}")
    else:
        logger.info("Nenhum arquivo PDF para processar.")

    # 2. Adicionar arquivos de texto brutos para limpeza
    if raw_text_files_to_process:
        logger.info(f"Adicionando {len(raw_text_files_to_process)} arquivos de texto brutos à fila de limpeza...")
        for item in raw_text_files_to_process:
            all_processed_text_files_details.append({'path': item['path'], 'lang': item['lang']})
            logger.debug(f"Arquivo de texto bruto {item['path']} adicionado para limpeza.")
    else:
        logger.info("Nenhum arquivo de texto bruto para adicionar à limpeza.")
    
    if not all_processed_text_files_details:
        logger.warning("Nenhum arquivo de texto (de PDFs ou bruto) disponível para a etapa de limpeza.")
        return []

    # 3. Limpeza consolidada de todos os arquivos de texto (de PDFs e brutos)
    logger.info(f"Iniciando limpeza consolidada de {len(all_processed_text_files_details)} arquivos de texto...")
    final_cleaned_text_paths = []
    for text_file_detail in all_processed_text_files_details:
        input_path = text_file_detail['path']
        lang = text_file_detail['lang']
        
        # Determinar o diretório de saída para text_cleaner.process_single_file
        # Se o input_path já está no processed_dir (veio de um PDF), ele será processado no lugar.
        # Se o input_path está no raw_dir, o output será no processed_dir.
        # A função process_single_file já lida com a criação do nome do arquivo de saída no output_dir.
        
        logger.info(f"Limpando arquivo de texto: {input_path} com idioma {lang}")
        cleaned_path = text_cleaner.process_single_file(
            input_file_path=input_path,
            output_dir=processed_dir, # process_single_file colocará o resultado aqui
            language=lang
        )
        if cleaned_path:
            final_cleaned_text_paths.append(cleaned_path)
            logger.info(f"Arquivo {input_path} limpo e salvo como {cleaned_path}")
        else:
            logger.warning(f"Falha ao limpar o arquivo de texto: {input_path}")
            
    logger.info(f"Processamento e limpeza concluídos. Total de {len(final_cleaned_text_paths)} arquivos de texto finais.")
    return final_cleaned_text_paths

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
    parser.add_argument(
        '--pubmed-api-key',
        type=str,
        default=os.getenv('PUBMED_API_KEY'), # Get from env var by default
        help='API Key for NCBI E-utils (PubMed). Can also be set via PUBMED_API_KEY environment variable.'
    )
    args = parser.parse_args()
    
    logger.info(f"Argumentos CLI: query_pt='{args.query_pt}', query_en='{args.query_en}', max_articles={args.max_articles}, pubmed_api_key_present={'Sim' if args.pubmed_api_key else 'Não'}, skip_scraping={args.skip_scraping}, skip_processing={args.skip_processing}, skip_indexing={args.skip_indexing}")
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    index_dir = os.path.join("data", "index")
    
    # Garantir que os diretórios existam
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    
    logger.info(f"Argumentos CLI: query_pt='{args.query_pt}', query_en='{args.query_en}', max_articles={args.max_articles}, skip_scraping={args.skip_scraping}, skip_processing={args.skip_processing}, skip_indexing={args.skip_indexing}")
    
    # Diretórios de dados
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    index_dir = os.path.join("data", "index")
    
    # Garantir que os diretórios existam
    logger.info(f"Garantindo que os diretórios de dados existam: {raw_dir}, {processed_dir}, {index_dir}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    # Etapa 1: Scraping
    scraped_files_details = []
    if not args.skip_scraping:
        logger.info("--- Iniciando Etapa 1: Scraping ---")
        pubmed_api_key_to_pass = args.pubmed_api_key # Explicitly get from args
        scraped_files_details = scrape_all_sources(
            raw_dir=raw_dir,
            query_pt=args.query_pt,
            query_en=args.query_en,
            max_articles=args.max_articles,
            pubmed_api_key=pubmed_api_key_to_pass
        )
        logger.info("--- Etapa 1: Scraping Concluída ---")
    else:
        logger.info("--- Etapa 1: Scraping Ignorada (conforme argumento --skip-scraping) ---")
        # Nota: Se o scraping for pulado, `scraped_files_details` estará vazio.
        # A etapa de processamento pode precisar de lógica adicional para encontrar arquivos existentes
        # se essa for uma funcionalidade desejada ao pular o scraping. (Fora do escopo atual)

    # Etapa 2: Processamento de texto
    processed_text_files = []
    if not args.skip_processing:
        logger.info("--- Iniciando Etapa 2: Processamento de Texto ---")
        if not scraped_files_details and not args.skip_scraping : # Adicionado para verificar se o scraping não produziu nada
             logger.warning("Nenhum arquivo foi retornado pela etapa de scraping. A etapa de processamento pode não ter dados para processar.")
        elif not scraped_files_details and args.skip_scraping:
             logger.info("Scraping foi pulado. A etapa de processamento tentará processar arquivos de execuções anteriores, se houver, ou pode não ter dados se `raw_files_details` não for preenchido de outra forma.")
        
        processed_text_files = process_all_files(
            raw_files_details=scraped_files_details, # Pode estar vazio se o scraping foi pulado ou falhou
            processed_dir=processed_dir
        )
        logger.info("--- Etapa 2: Processamento de Texto Concluída ---")
    else:
        logger.info("--- Etapa 2: Processamento de Texto Ignorada (conforme argumento --skip-processing) ---")
    
    # Etapa 3: Indexação vetorial
    if not args.skip_indexing:
        logger.info("--- Iniciando Etapa 3: Indexação Vetorial ---")
        if not processed_text_files and not args.skip_processing: # Adicionado para verificar se o processamento não produziu nada
            logger.warning("Nenhum arquivo de texto processado disponível. A etapa de indexação não terá dados para indexar.")
        elif not processed_text_files and args.skip_processing:
             logger.info("Processamento foi pulado. A etapa de indexação tentará usar arquivos de execuções anteriores, se houver, ou pode não ter dados se `processed_text_files` não for preenchido de outra forma.")

        index_path = os.path.join(index_dir, "tpb_index.pkl")
        # create_index now expects a list of file paths (processed_text_files)
        vector_store.create_index(
            file_paths=processed_text_files, # Pass the list of cleaned text file paths
            output_path=index_path
            # input_dir parameter might need to be removed or adapted in vector_store.create_index
        )
    else:
        logger.info("Etapa de indexação ignorada")
    
    logger.info("Processo concluído com sucesso")

if __name__ == "__main__":
    main()