#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para extração e processamento de texto de arquivos PDF.
"""

import os
import logging
import glob
import json
from typing import List, Dict, Any, Optional

# Configuração de logging
logger = logging.getLogger(__name__)

# Tentar importar bibliotecas para processamento de PDF
try:
    import PyPDF2
    HAVE_PYPDF2 = True
except ImportError:
    logger.warning("PyPDF2 não encontrado. Tentando alternativas...")
    HAVE_PYPDF2 = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAVE_PDFMINER = True
except ImportError:
    logger.warning("pdfminer.six não encontrado. A extração de texto de PDFs pode ser limitada.")
    HAVE_PDFMINER = False

if not (HAVE_PYPDF2 or HAVE_PDFMINER):
    logger.error("Nenhuma biblioteca de processamento de PDF encontrada. Instale PyPDF2 ou pdfminer.six.")

class PDFProcessor:
    """Classe para extração e processamento de texto de PDFs."""
    
    def __init__(self, prefer_pdfminer=True):
        """
        Inicializa o processador de PDF.
        
        Args:
            prefer_pdfminer: Se deve preferir pdfminer.six sobre PyPDF2
        """
        self.prefer_pdfminer = prefer_pdfminer
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extrai texto de um arquivo PDF.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído do PDF
        """
        if not os.path.exists(pdf_path):
            logger.error(f"Arquivo não encontrado: {pdf_path}")
            return ""
        
        text = ""
        
        # Tentar extrair com pdfminer.six primeiro se preferido
        if self.prefer_pdfminer and HAVE_PDFMINER:
            try:
                text = pdfminer_extract_text(pdf_path)
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"Erro ao extrair texto com pdfminer.six: {str(e)}")
        
        # Tentar extrair com PyPDF2 se pdfminer falhou ou não é preferido
        if HAVE_PYPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = '\n'.join([page.extract_text() or '' for page in reader.pages])
                return text
            except Exception as e:
                logger.warning(f"Erro ao extrair texto com PyPDF2: {str(e)}")
        
        # Se pdfminer não é preferido mas está disponível, tentar como fallback
        if not self.prefer_pdfminer and HAVE_PDFMINER:
            try:
                text = pdfminer_extract_text(pdf_path)
                return text
            except Exception as e:
                logger.warning(f"Erro ao extrair texto com pdfminer.six (fallback): {str(e)}")
        
        # Se chegou aqui, nenhum método funcionou
        if not text.strip():
            logger.error(f"Não foi possível extrair texto do PDF: {pdf_path}")
        
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extrai metadados de um arquivo PDF.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Dicionário com metadados do PDF
        """
        metadata = {
            "filename": os.path.basename(pdf_path),
            "path": pdf_path,
            "size": os.path.getsize(pdf_path),
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
            "page_count": 0
        }
        
        if not os.path.exists(pdf_path):
            logger.error(f"Arquivo não encontrado: {pdf_path}")
            return metadata
        
        # Extrair metadados com PyPDF2
        if HAVE_PYPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    info = reader.metadata
                    if info:
                        metadata["page_count"] = len(reader.pages)
                        if info.title:
                            metadata["title"] = info.title
                        if info.author:
                            metadata["author"] = info.author
                        if info.subject:
                            metadata["subject"] = info.subject
                        if info.keywords:
                            metadata["keywords"] = info.keywords
                        if info.creator:
                            metadata["creator"] = info.creator
                        if info.producer:
                            metadata["producer"] = info.producer
                        if info.creation_date:
                            metadata["creation_date"] = str(info.creation_date)
                        if info.modification_date:
                            metadata["modification_date"] = str(info.modification_date)
            except Exception as e:
                logger.warning(f"Erro ao extrair metadados com PyPDF2: {str(e)}")
        
        return metadata
    
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa um arquivo PDF, extraindo texto e metadados.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            output_dir: Diretório para salvar os arquivos processados (opcional)
            
        Returns:
            Dicionário com texto e metadados extraídos
        """
        logger.info(f"Processando PDF: {pdf_path}")
        
        # Extrair texto e metadados
        text = self.extract_text(pdf_path)
        metadata = self.extract_metadata(pdf_path)
        
        result = {
            "text": text,
            "metadata": metadata
        }
        
        # Salvar resultados se output_dir for fornecido
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Criar nome base para os arquivos de saída
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Salvar texto extraído
            text_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Salvar metadados
            metadata_path = os.path.join(output_dir, f"{base_name}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Resultados salvos em: {text_path} e {metadata_path}")
        
        return result

def process_pdfs(input_dir: str, output_dir: str, file_pattern: str = "*.pdf") -> List[str]:
    """
    Processa todos os arquivos PDF em um diretório.
    
    Args:
        input_dir: Diretório com os PDFs de entrada
        output_dir: Diretório para salvar os arquivos processados
        file_pattern: Padrão para selecionar arquivos
        
    Returns:
        Lista de caminhos para os arquivos processados
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar o processador de PDF
    processor = PDFProcessor()
    
    # Encontrar todos os arquivos que correspondem ao padrão
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    logger.info(f"Encontrados {len(input_files)} arquivos PDF para processamento")
    
    processed_files = []
    for input_file in input_files:
        try:
            # Extrair nome do arquivo
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_text_file = os.path.join(output_dir, f"{filename}.txt")
            
            # Processar o PDF
            processor.process_pdf(input_file, output_dir)
            
            processed_files.append(output_text_file)
            logger.info(f"Processado: {input_file} -> {output_text_file}")
            
        except Exception as e:
            logger.error(f"Erro ao processar {input_file}: {str(e)}")
    
    logger.info(f"Processamento concluído. {len(processed_files)} arquivos PDF processados.")
    return processed_files

if __name__ == "__main__":
    # Teste da função de processamento
    logging.basicConfig(level=logging.INFO)
    process_pdfs("data/raw", "data/processed")