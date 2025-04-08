#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para scraping de artigos científicos do PubMed Central (PMC).
"""

import os
import logging
import requests
import time
import random
import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class PubMedScraper:
    """Classe para extrair artigos científicos do PubMed Central."""
    
    BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc"
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, max_articles=50, delay=1.0, api_key=None):
        """
        Inicializa o scraper.
        
        Args:
            max_articles: Número máximo de artigos a serem extraídos
            delay: Tempo de espera entre requisições (em segundos)
            api_key: Chave de API do NCBI (opcional, aumenta limites de requisição)
        """
        self.max_articles = max_articles
        self.delay = delay
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        })
    
    def search_articles(self, query, retmax=100):
        """
        Busca artigos com base em uma consulta.
        
        Args:
            query: Termos de busca
            retmax: Número máximo de resultados a retornar
            
        Returns:
            Lista de PMCIDs de artigos
        """
        logger.info(f"Buscando artigos no PubMed Central com a consulta: '{query}'")
        
        # Parâmetros de busca
        params = {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": min(retmax, self.max_articles),
            "sort": "relevance"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        pmcids = []
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            pmcids = data.get("esearchresult", {}).get("idlist", [])
            
            logger.info(f"Encontrados {len(pmcids)} artigos no PubMed Central")
            
        except Exception as e:
            logger.error(f"Erro ao buscar artigos no PubMed Central: {str(e)}")
        
        return pmcids[:self.max_articles]
    
    def fetch_article_metadata(self, pmcid):
        """
        Obtém metadados de um artigo pelo PMCID.
        
        Args:
            pmcid: ID do artigo no PubMed Central
            
        Returns:
            Dicionário com metadados do artigo
        """
        logger.info(f"Obtendo metadados do artigo PMCID: {pmcid}")
        
        # Parâmetros para busca de metadados
        params = {
            "db": "pmc",
            "id": pmcid,
            "retmode": "xml"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        metadata = {
            "pmcid": pmcid,
            "title": "",
            "abstract": "",
            "authors": [],
            "journal": "",
            "publication_date": "",
            "keywords": [],
            "doi": "",
            "full_text_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
            "pdf_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
        }
        
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(self.FETCH_URL, params=params)
            response.raise_for_status()
            
            # Parsear XML
            root = ET.fromstring(response.content)
            
            # Extrair título
            title_elem = root.find(".//article-title")
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text
            
            # Extrair abstract
            abstract_elems = root.findall(".//abstract//p")
            if abstract_elems:
                metadata["abstract"] = " ".join([p.text for p in abstract_elems if p.text])
            
            # Extrair autores
            author_elems = root.findall(".//contrib[@contrib-type='author']")
            for author in author_elems:
                surname = author.find(".//surname")
                given_names = author.find(".//given-names")
                if surname is not None and given_names is not None:
                    metadata["authors"].append(f"{surname.text}, {given_names.text}")
            
            # Extrair revista
            journal_elem = root.find(".//journal-title")
            if journal_elem is not None and journal_elem.text:
                metadata["journal"] = journal_elem.text
            
            # Extrair data de publicação
            pub_date = root.find(".//pub-date")
            if pub_date is not None:
                year = pub_date.find("year")
                month = pub_date.find("month")
                day = pub_date.find("day")
                
                date_parts = []
                if year is not None and year.text:
                    date_parts.append(year.text)
                if month is not None and month.text:
                    date_parts.append(month.text)
                if day is not None and day.text:
                    date_parts.append(day.text)
                
                metadata["publication_date"] = "-".join(date_parts)
            
            # Extrair palavras-chave
            kwd_elems = root.findall(".//kwd")
            metadata["keywords"] = [kwd.text for kwd in kwd_elems if kwd.text]
            
            # Extrair DOI
            doi_elem = root.find(".//article-id[@pub-id-type='doi']")
            if doi_elem is not None and doi_elem.text:
                metadata["doi"] = doi_elem.text
            
        except Exception as e:
            logger.error(f"Erro ao obter metadados do artigo PMCID {pmcid}: {str(e)}")
        
        return metadata
    
    def extract_full_text(self, pmcid):
        """
        Extrai o texto completo de um artigo.
        
        Args:
            pmcid: ID do artigo no PubMed Central
            
        Returns:
            Texto completo do artigo
        """
        logger.info(f"Extraindo texto completo do artigo PMCID: {pmcid}")
        
        full_text_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
        full_text = ""
        
        try:
            # Adicionar delay para evitar sobrecarga do servidor
            time.sleep(self.delay * (0.5 + random.random()))
            
            response = self.session.get(full_text_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extrair o conteúdo do artigo
            article_content = soup.select_one("div.jig-ncbiinpagenav")
            if article_content:
                # Extrair parágrafos
                paragraphs = []
                for p in article_content.select("p"):
                    text = p.get_text(strip=True)
                    if text:
                        paragraphs.append(text)
                
                full_text = "\n\n".join(paragraphs)
            
            # Se não conseguiu extrair o conteúdo, tentar outra abordagem
            if not full_text:
                # Tentar extrair o conteúdo da div principal
                main_content = soup.select_one("div.jig-ncbiinpagenav-content")
                if main_content:
                    full_text = main_content.get_text(strip=True)
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto completo do artigo PMCID {pmcid}: {str(e)}")
        
        return full_text
    
    def download_pdf(self, pmcid, output_path):
        """
        Baixa o PDF de um artigo.
        
        Args:
            pmcid: ID do artigo no PubMed Central
            output_path: Caminho para salvar o arquivo
            
        Returns:
            Boolean indicando sucesso
        """
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
        
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
            logger.error(f"Erro ao baixar PDF do artigo PMCID {pmcid}: {str(e)}")
            return False

def scrape_articles(query="borderline personality disorder", max_articles=10, output_dir="data/raw"):
    """
    Função principal para extrair artigos do PubMed Central.
    
    Args:
        query: Termos de busca
        max_articles: Número máximo de artigos
        output_dir: Diretório para salvar os dados
    
    Returns:
        Lista de caminhos para os arquivos extraídos
    """
    scraper = PubMedScraper(max_articles=max_articles)
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar artigos
    pmcids = scraper.search_articles(query)
    
    extracted_files = []
    for i, pmcid in enumerate(pmcids):
        # Obter metadados do artigo
        metadata = scraper.fetch_article_metadata(pmcid)
        
        if not metadata["title"]:
            continue
        
        # Extrair texto completo
        full_text = scraper.extract_full_text(pmcid)
        
        # Criar nome de arquivo baseado no título
        safe_title = "".join([c if c.isalnum() else "_" for c in metadata["title"]])
        safe_title = safe_title[:50]  # Limitar tamanho
        
        # Salvar metadados e texto
        metadata_path = os.path.join(output_dir, f"pmc_{i+1:03d}_{safe_title}.json")
        text_path = os.path.join(output_dir, f"pmc_{i+1:03d}_{safe_title}.txt")
        
        # Adicionar texto completo aos metadados
        metadata["full_text"] = full_text
        
        # Salvar metadados como JSON
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Salvar texto completo
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(metadata["title"] + "\n\n")
            f.write(metadata["abstract"] + "\n\n")
            f.write(full_text)
        
        extracted_files.append(text_path)
        
        # Baixar PDF se disponível
        pdf_path = os.path.join(output_dir, f"pmc_{i+1:03d}_{safe_title}.pdf")
        scraper.download_pdf(pmcid, pdf_path)
        
    logger.info(f"Extração concluída. {len(extracted_files)} arquivos salvos em {output_dir}")
    return extracted_files

if __name__ == "__main__":
    # Teste da função de scraping
    logging.basicConfig(level=logging.INFO)
    scrape_articles(query="borderline personality disorder", max_articles=3)