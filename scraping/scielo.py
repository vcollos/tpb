#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para scraping de artigos científicos do SciELO.
"""

import os
import logging
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class SciELOScraper:
    """Classe para extrair artigos científicos do portal SciELO."""
    
    BASE_URL = "https://www.scielo.br"
    SEARCH_URL = f"{BASE_URL}/cgi-bin/wxis.exe/iah/"
    
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
    
    def search_articles(self, query, language="pt"):
        """
        Busca artigos com base em uma consulta.
        
        Args:
            query: Termos de busca
            language: Idioma dos artigos (pt, en, es)
            
        Returns:
            Lista de URLs de artigos
        """
        logger.info(f"Buscando artigos com a consulta: '{query}'")
        
        # Parâmetros de busca do SciELO
        params = {
            "IsisScript": "iah/iah.xis",
            "base": "article",
            "lang": language,
            "format": "standard",
            "count": "50",
            "nextAction": "search",
            "exprSearch": query
        }
        
        article_urls = []
        try:
            # Adicionar delay para evitar sobrecarga do servidor (já existe implicitamente no __init__, mas pode ser bom ter aqui também)
            time.sleep(self.delay * (0.5 + random.random()))
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.record')
            
            for result in results[:self.max_articles]:
                link_elem = result.select_one('a[href*="/scielo.php?"]')
                if link_elem and 'href' in link_elem.attrs:
                    article_urls.append(link_elem['href'])
            
            logger.info(f"Encontrados {len(article_urls)} artigos")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during search_articles: {e}")
        except Exception as e: # Keep a general exception for other unexpected errors like parsing
            logger.error(f"Erro ao buscar artigos: {str(e)}")
        
        return article_urls
    
    def extract_article(self, url):
        """
        Extrai o conteúdo de um artigo.
        
        Args:
            url: URL do artigo
            
        Returns:
            Dicionário com os dados do artigo
        """
        logger.info(f"Extraindo artigo: {url}")
        
        article_data = {
            'url': url,
            'title': '',
            'abstract': '',
            'keywords': [],
            'full_text': '',
            'pdf_url': ''
        }
        
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(url)
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extrair título
            title_elem = soup.select_one('.title')
            if title_elem:
                article_data['title'] = title_elem.get_text(strip=True)
            
            # Extrair resumo
            abstract_elem = soup.select_one('.abstract')
            if abstract_elem:
                article_data['abstract'] = abstract_elem.get_text(strip=True)
            
            # Extrair palavras-chave
            keywords_elem = soup.select('.keyword')
            article_data['keywords'] = [k.get_text(strip=True) for k in keywords_elem]
            
            # Extrair texto completo
            full_text_elems = soup.select('.content')
            article_data['full_text'] = '\n\n'.join([p.get_text(strip=True) for p in full_text_elems])
            
            # Extrair URL do PDF
            pdf_link = soup.select_one('a[href*=".pdf"]')
            if pdf_link and 'href' in pdf_link.attrs:
                pdf_href = pdf_link['href']
                article_data['pdf_url'] = urljoin(self.BASE_URL, pdf_href)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during extract_article for URL {url}: {e}")
        except Exception as e: # General exception for other issues like parsing
            logger.error(f"Erro ao extrair artigo {url}: {str(e)}")
        
        return article_data
    
    def download_pdf(self, pdf_url, output_path):
        """
        Baixa o PDF de um artigo.
        
        Args:
            pdf_url: URL do PDF
            output_path: Caminho para salvar o arquivo
            
        Returns:
            Boolean indicando sucesso
        """
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"PDF baixado com sucesso: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during download_pdf for URL {pdf_url}: {e}")
            return False
        except IOError as e: # Specific error for file writing issues
            logger.error(f"File I/O error while saving PDF from {pdf_url} to {output_path}: {e}")
            return False
        except Exception as e: # General exception for other issues
            logger.error(f"Erro ao baixar PDF {pdf_url}: {str(e)}")
            return False

def scrape_articles(query="transtorno de personalidade borderline", max_articles=10, output_dir="data/raw"):
    """
    Função principal para extrair artigos do SciELO.
    
    Args:
        query: Termos de busca
        max_articles: Número máximo de artigos
        output_dir: Diretório para salvar os dados
    
    Returns:
        Lista de caminhos para os arquivos extraídos
    """
    scraper = SciELOScraper(max_articles=max_articles)
    lang_code = 'pt' # Default language for SciELO
    
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
        metadata_path = os.path.join(output_dir, f"{i+1:03d}_{safe_title}.json")
        text_path = os.path.join(output_dir, f"{i+1:03d}_{safe_title}.txt")
        
        # Salvar metadados como JSON
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, ensure_ascii=False, indent=2)
        
        # Salvar texto completo
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(article_data['title'] + "\n\n")
            f.write(article_data['abstract'] + "\n\n")
            f.write(article_data['full_text'])
        
        extracted_files.append({'path': text_path, 'lang': lang_code, 'type': 'txt'})
        
        # Baixar PDF se disponível
        if article_data['pdf_url']:
            pdf_path = os.path.join(output_dir, f"{i+1:03d}_{safe_title}.pdf")
            if scraper.download_pdf(article_data['pdf_url'], pdf_path):
                extracted_files.append({'path': pdf_path, 'lang': lang_code, 'type': 'pdf'})
    
    logger.info(f"Extração concluída. {len(extracted_files)} itens (arquivos de texto/PDF) extraídos para {output_dir}")
    return extracted_files

if __name__ == "__main__":
    # Teste da função de scraping
    logging.basicConfig(level=logging.INFO)
    scrape_articles(query="transtorno de personalidade borderline", max_articles=3)