#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para limpeza e normalização de textos extraídos.
"""

import os
import re
import logging
import glob
import json
from typing import List, Dict, Any, Optional

# Configuração de logging
logger = logging.getLogger(__name__)

# Tentar importar NLTK para processamento de texto
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    
    # Baixar recursos do NLTK se necessário
    nltk_resources = ['punkt', 'stopwords']
    for resource in nltk_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    
    HAVE_NLTK = True
except ImportError:
    logger.warning("NLTK não encontrado. Algumas funcionalidades de processamento de texto serão limitadas.")
    HAVE_NLTK = False

class TextCleaner:
    """Classe para limpeza e normalização de textos."""
    
    def __init__(self, language='portuguese', remove_stopwords=False):
        """
        Inicializa o limpador de texto.
        
        Args:
            language: Idioma para processamento (português ou inglês)
            remove_stopwords: Se deve remover stopwords
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        
        # Carregar stopwords se NLTK estiver disponível e remoção de stopwords estiver ativada
        self.stopwords = set()
        if HAVE_NLTK and remove_stopwords:
            nltk_lang = 'portuguese' if language == 'portuguese' else 'english'
            self.stopwords = set(stopwords.words(nltk_lang))
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza um texto.
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo e normalizado
        """
        if not text:
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remover emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remover números
        text = re.sub(r'\d+', '', text)
        
        # Remover pontuação
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remover stopwords se necessário
        if self.remove_stopwords and HAVE_NLTK:
            words = word_tokenize(text, language=self.language)
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliza um texto sem remover informações importantes.
        
        Args:
            text: Texto a ser normalizado
            
        Returns:
            Texto normalizado
        """
        if not text:
            return ""
        
        # Remover URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remover emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalizar espaços
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalizar quebras de linha
        text = re.sub(r'\n+', '\n', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extrai sentenças de um texto.
        
        Args:
            text: Texto a ser processado
            
        Returns:
            Lista de sentenças
        """
        if not text:
            return []
        
        if HAVE_NLTK:
            return sent_tokenize(text, language=self.language)
        else:
            # Fallback simples se NLTK não estiver disponível
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def process_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa um documento completo.
        
        Args:
            text: Texto do documento
            metadata: Metadados do documento (opcional)
            
        Returns:
            Dicionário com texto processado e metadados
        """
        if not text:
            return {"text": "", "metadata": metadata or {}}
        
        # Normalizar texto
        normalized_text = self.normalize_text(text)
        
        # Extrair sentenças
        sentences = self.extract_sentences(normalized_text)
        
        # Preparar resultado
        result = {
            "text": normalized_text,
            "sentences": sentences,
            "metadata": metadata or {}
        }
        
        # Adicionar estatísticas básicas
        result["stats"] = {
            "char_count": len(normalized_text),
            "word_count": len(normalized_text.split()),
            "sentence_count": len(sentences)
        }
        
        return result

def process_documents(input_dir: str, output_dir: str, file_pattern: str = "*.txt") -> List[str]:
    """
    Processa todos os documentos de texto em um diretório.
    
    Args:
        input_dir: Diretório com os documentos de entrada
        output_dir: Diretório para salvar os documentos processados
        file_pattern: Padrão para selecionar arquivos
        
    Returns:
        Lista de caminhos para os arquivos processados
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar o limpador de texto
    cleaner = TextCleaner()
    
    # Encontrar todos os arquivos que correspondem ao padrão
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    logger.info(f"Encontrados {len(input_files)} arquivos para processamento")
    
    processed_files = []
    for input_file in input_files:
        try:
            # Extrair nome do arquivo
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename)
            
            # Verificar se existe arquivo de metadados correspondente
            metadata_file = os.path.splitext(input_file)[0] + ".json"
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Ler o texto do arquivo
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Processar o documento
            processed_doc = cleaner.process_document(text, metadata)
            
            # Salvar o documento processado
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(processed_doc["text"])
            
            # Salvar metadados processados
            output_metadata_file = os.path.splitext(output_file)[0] + ".json"
            with open(output_metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": processed_doc["metadata"],
                    "stats": processed_doc["stats"]
                }, f, ensure_ascii=False, indent=2)
            
            processed_files.append(output_file)
            logger.info(f"Processado: {input_file} -> {output_file}")
            
        except Exception as e:
            logger.error(f"Erro ao processar {input_file}: {str(e)}")
    
    logger.info(f"Processamento concluído. {len(processed_files)} arquivos processados.")
    return processed_files

if __name__ == "__main__":
    # Teste da função de processamento
    logging.basicConfig(level=logging.INFO)
    process_documents("data/raw", "data/processed")