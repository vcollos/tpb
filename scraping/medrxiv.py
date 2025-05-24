#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para scraping de preprints do medRxiv.
"""

import os
import logging
import requests
import time
import random
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class MedRxivScraper:
    """Classe para extrair preprints do medRxiv."""
    
    BASE_URL = "https://www.medrxiv.org"
    SEARCH_URL = f"{BASE_URL}/search"
    
    def __init__(self, max_articles=50, delay=1.0):
        """
        Inicializa o scraper.
        
        Args:
            max_articles: Número máximo de artigos a serem extraídos
            delay: Tempo de espera entre requisições (em segundos)
        """
        self.max_articles = max_articles
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        })
    
    def search_articles(self, query):
        """
        Busca artigos com base em uma consulta.
        
        Args:
            query: Termos de busca
            
        Returns:
            Lista de URLs de artigos
        """
        logger.info(f"Buscando preprints no medRxiv com a consulta: '{query}'")
        
        # Parâmetros de busca do medRxiv
        params = {
            "term": query,
            "page": 1
        }
        
        article_urls = []
        try:
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.search-results article')
            
            for result in results[:self.max_articles]:
                link_elem = result.select_one('a.highwire-cite-linked-title')
                if link_elem and 'href' in link_elem.attrs:
                    article_url = urljoin(self.BASE_URL, link_elem['href'])
                    article_urls.append(article_url)
            
            logger.info(f"Encontrados {len(article_urls)} preprints no medRxiv")
            
        except Exception as e:
            logger.error(f"Erro ao buscar preprints no medRxiv: {str(e)}")
        
        return article_urls
    
    def extract_article(self, url):
        """
        Extrai o conteúdo de um preprint.
        
        Args:
            url: URL do preprint
            
        Returns:
            Dicionário com os dados do preprint
        """
        logger.info(f"Extraindo preprint do medRxiv: {url}")
        
        article_data = {
            'url': url,
            'title': '',
            'abstract': '',
            'authors': [],
            'publication_date': '',
            'doi': '',
            'keywords': [],
            'pdf_url': ''
        }
        
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extrair título
            title_elem = soup.select_one('h1.article-title')
            if title_elem:
                article_data['title'] = title_elem.get_text(strip=True)
            
            # Extrair abstract
            abstract_elem = soup.select_one('.abstract')
            if abstract_elem:
                article_data['abstract'] = abstract_elem.get_text(strip=True)
            
            # Extrair autores
            authors_elem = soup.select('.highwire-citation-author')
            article_data['authors'] = [a.get_text(strip=True) for a in authors_elem]
            
            # Extrair data de publicação
            date_elem = soup.select_one('.article-meta-date')
            if date_elem:
                article_data['publication_date'] = date_elem.get_text(strip=True)
            
            # Extrair DOI
            doi_elem = soup.select_one('.article-meta-doi')
            if doi_elem:
                article_data['doi'] = doi_elem.get_text(strip=True).replace('DOI:', '').strip()
            
            # Extrair palavras-chave
            keywords_elem = soup.select('.kwd-text')
            article_data['keywords'] = [k.get_text(strip=True) for k in keywords_elem]
            
            # Extrair URL do PDF
            pdf_link = soup.select_one('a.article-dl-pdf-link')
            if pdf_link and 'href' in pdf_link.attrs:
                article_data['pdf_url'] = urljoin(self.BASE_URL, pdf_link['href'])
            
        except Exception as e:
            logger.error(f"Erro ao extrair preprint do medRxiv {url}: {str(e)}")
        
        return article_data
    
    def download_pdf(self, pdf_url, output_path):
        """
        Baixa o PDF de um preprint.
        
        Args:
            pdf_url: URL do PDF
            output_path: Caminho para salvar o arquivo
            
        Returns:
            Boolean indicando sucesso
        """
        if not pdf_url:
            logger.warning(f"URL do PDF não fornecida para download")
            return False
        
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"PDF baixado com sucesso: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao baixar PDF {pdf_url}: {str(e)}")
            return False

def scrape_articles(query="borderline personality disorder", max_articles=10, output_dir="data/raw"):
    """
    Função principal para extrair preprints do medRxiv.
    
    Args:
        query: Termos de busca
        max_articles: Número máximo de artigos
        output_dir: Diretório para salvar os dados
    
    Returns:
        Lista de caminhos para os arquivos extraídos
    """
    scraper = MedRxivScraper(max_articles=max_articles)
    lang_code = 'en' # Default language for medRxiv
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar artigos
    article_urls = scraper.search_articles(query)
    
    extracted_files = [] # This will now be a list of dictionaries
    for i, url in enumerate(article_urls):
        # Extrair dados do artigo
        article_data = scraper.extract_article(url)
        
        if not article_data['title']:
            continue
        
        # Criar nome de arquivo baseado no título
        safe_title = "".join([c if c.isalnum() else "_" for c in article_data['title']])
        safe_title = safe_title[:50]  # Limitar tamanho
        
        # Salvar metadados e texto
        metadata_path = os.path.join(output_dir, f"medrxiv_{i+1:03d}_{safe_title}.json")
        text_path = os.path.join(output_dir, f"medrxiv_{i+1:03d}_{safe_title}.txt")
        
        # Salvar metadados como JSON
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, ensure_ascii=False, indent=2)
        
        # Salvar texto completo (apenas título e abstract, já que o texto completo está no PDF)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(article_data['title'] + "\n\n")
            f.write("Autores: " + ", ".join(article_data['authors']) + "\n\n")
            f.write("Data: " + article_data['publication_date'] + "\n\n")
            if article_data['doi']:
                f.write("DOI: " + article_data['doi'] + "\n\n")
            f.write(article_data['abstract'])
        
        extracted_files.append({'path': text_path, 'lang': lang_code, 'type': 'txt'})
        
        # Baixar PDF se disponível
        if article_data['pdf_url']:
            pdf_path = os.path.join(output_dir, f"medrxiv_{i+1:03d}_{safe_title}.pdf")
            if scraper.download_pdf(article_data['pdf_url'], pdf_path):
                extracted_files.append({'path': pdf_path, 'lang': lang_code, 'type': 'pdf'})
    
    logger.info(f"Extração concluída. {len(extracted_files)} itens (arquivos de texto/PDF) extraídos para {output_dir}")
    return extracted_files

if __name__ == "__main__":
    # Teste da função de scraping
    logging.basicConfig(level=logging.INFO)
    scrape_articles(query="borderline personality disorder", max_articles=3)