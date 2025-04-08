#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para scraping de artigos científicos do PePSIC (Periódicos Eletrônicos em Psicologia).
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

class PePSICScraper:
    """Classe para extrair artigos científicos do PePSIC."""
    
    BASE_URL = "http://pepsic.bvsalud.org"
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
    
    def search_articles(self, query, language="p"):
        """
        Busca artigos com base em uma consulta.
        
        Args:
            query: Termos de busca
            language: Idioma dos artigos (p=português, i=inglês, e=espanhol)
            
        Returns:
            Lista de URLs de artigos
        """
        logger.info(f"Buscando artigos no PePSIC com a consulta: '{query}'")
        
        # Parâmetros de busca do PePSIC
        params = {
            "IsisScript": "iah/iah.xis",
            "base": "article",
            "lang": language,
            "format": "detailed.pft",
            "count": "50",
            "nextAction": "search",
            "exprSearch": query
        }
        
        article_urls = []
        try:
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.record')
            
            for result in results[:self.max_articles]:
                link_elem = result.select_one('a[href*="script=sci_arttext"]')
                if link_elem and 'href' in link_elem.attrs:
                    article_url = urljoin(self.BASE_URL, link_elem['href'])
                    article_urls.append(article_url)
            
            logger.info(f"Encontrados {len(article_urls)} artigos no PePSIC")
            
        except Exception as e:
            logger.error(f"Erro ao buscar artigos no PePSIC: {str(e)}")
        
        return article_urls
    
    def extract_article(self, url):
        """
        Extrai o conteúdo de um artigo.
        
        Args:
            url: URL do artigo
            
        Returns:
            Dicionário com os dados do artigo
        """
        logger.info(f"Extraindo artigo do PePSIC: {url}")
        
        article_data = {
            'url': url,
            'title': '',
            'title_en': '',  # Título em inglês (se disponível)
            'abstract': '',
            'abstract_en': '',  # Resumo em inglês (se disponível)
            'keywords': [],
            'keywords_en': [],  # Palavras-chave em inglês (se disponível)
            'authors': [],
            'journal': '',
            'publication_date': '',
            'full_text': '',
            'pdf_url': ''
        }
        
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extrair título em português
            title_elem = soup.select_one('.title')
            if title_elem:
                article_data['title'] = title_elem.get_text(strip=True)
            
            # Extrair título em inglês
            title_en_elem = soup.select_one('.title[lang="en"]')
            if title_en_elem:
                article_data['title_en'] = title_en_elem.get_text(strip=True)
            
            # Extrair resumo em português
            abstract_elem = soup.select_one('.abstract')
            if abstract_elem:
                article_data['abstract'] = abstract_elem.get_text(strip=True)
            
            # Extrair resumo em inglês
            abstract_en_elem = soup.select_one('.abstract[lang="en"]')
            if abstract_en_elem:
                article_data['abstract_en'] = abstract_en_elem.get_text(strip=True)
            
            # Extrair palavras-chave em português
            keywords_elem = soup.select('.keyword')
            article_data['keywords'] = [k.get_text(strip=True) for k in keywords_elem if not k.has_attr('lang') or k['lang'] != 'en']
            
            # Extrair palavras-chave em inglês
            keywords_en_elem = soup.select('.keyword[lang="en"]')
            article_data['keywords_en'] = [k.get_text(strip=True) for k in keywords_en_elem]
            
            # Extrair autores
            authors_elem = soup.select('.author')
            article_data['authors'] = [a.get_text(strip=True) for a in authors_elem]
            
            # Extrair revista
            journal_elem = soup.select_one('.journalTitle')
            if journal_elem:
                article_data['journal'] = journal_elem.get_text(strip=True)
            
            # Extrair data de publicação
            date_elem = soup.select_one('.publicationDate')
            if date_elem:
                article_data['publication_date'] = date_elem.get_text(strip=True)
            
            # Extrair texto completo
            full_text_elems = soup.select('#article-body p, #article-body h2, #article-body h3')
            article_data['full_text'] = '\n\n'.join([p.get_text(strip=True) for p in full_text_elems])
            
            # Se não encontrou o texto no formato esperado, tentar outra abordagem
            if not article_data['full_text']:
                full_text_elems = soup.select('.content')
                article_data['full_text'] = '\n\n'.join([p.get_text(strip=True) for p in full_text_elems])
            
            # Extrair URL do PDF
            pdf_link = soup.select_one('a[href*=".pdf"]')
            if pdf_link and 'href' in pdf_link.attrs:
                article_data['pdf_url'] = urljoin(self.BASE_URL, pdf_link['href'])
            
        except Exception as e:
            logger.error(f"Erro ao extrair artigo do PePSIC {url}: {str(e)}")
        
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

def scrape_articles(query="transtorno de personalidade borderline", max_articles=10, output_dir="data/raw"):
    """
    Função principal para extrair artigos do PePSIC.
    
    Args:
        query: Termos de busca
        max_articles: Número máximo de artigos
        output_dir: Diretório para salvar os dados
    
    Returns:
        Lista de caminhos para os arquivos extraídos
    """
    scraper = PePSICScraper(max_articles=max_articles)
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar artigos
    article_urls = scraper.search_articles(query)
    
    extracted_files = []
    for i, url in enumerate(article_urls):
        # Extrair dados do artigo
        article_data = scraper.extract_article(url)
        
        if not article_data['title']:
            continue
        
        # Criar nome de arquivo baseado no título
        safe_title = "".join([c if c.isalnum() else "_" for c in article_data['title']])
        safe_title = safe_title[:50]  # Limitar tamanho
        
        # Salvar metadados e texto
        metadata_path = os.path.join(output_dir, f"pepsic_{i+1:03d}_{safe_title}.json")
        text_path = os.path.join(output_dir, f"pepsic_{i+1:03d}_{safe_title}.txt")
        
        # Salvar metadados como JSON
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, ensure_ascii=False, indent=2)
        
        # Salvar texto completo
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(article_data['title'] + "\n\n")
            if article_data['title_en']:
                f.write(article_data['title_en'] + "\n\n")
            f.write(article_data['abstract'] + "\n\n")
            if article_data['abstract_en']:
                f.write(article_data['abstract_en'] + "\n\n")
            f.write(article_data['full_text'])
        
        extracted_files.append(text_path)
        
        # Baixar PDF se disponível
        if article_data['pdf_url']:
            pdf_path = os.path.join(output_dir, f"pepsic_{i+1:03d}_{safe_title}.pdf")
            scraper.download_pdf(article_data['pdf_url'], pdf_path)
            extracted_files.append(pdf_path)
    
    logger.info(f"Extração concluída. {len(extracted_files)} arquivos salvos em {output_dir}")
    return extracted_files

if __name__ == "__main__":
    # Teste da função de scraping
    logging.basicConfig(level=logging.INFO)
    scrape_articles(query="transtorno de personalidade borderline", max_articles=3)