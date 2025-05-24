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
    
    def __init__(self, language='en'):
        """
        Inicializa o limpador de texto.
        
        Args:
            language: Idioma para processamento ('en', 'pt', 'es').
        """
        self.language = language
        self.remove_stopwords = True # Defaulting to True as parameter is removed
        
        self.nltk_lang = 'english' # default
        if language == 'pt':
            self.nltk_lang = 'portuguese'
        elif language == 'es':
            self.nltk_lang = 'spanish'
        elif language == 'en':
            self.nltk_lang = 'english'
        else:
            logger.warning(f"Unsupported language code '{language}' for TextCleaner. Defaulting to English for NLTK.")
            # self.nltk_lang remains 'english' as set by default
        
        # Carregar stopwords se NLTK estiver disponível (remove_stopwords is now True by default)
        self.stopwords = set()
        if HAVE_NLTK and self.remove_stopwords: # self.remove_stopwords is True
            try:
                self.stopwords = set(stopwords.words(self.nltk_lang))
            except LookupError:
                logger.error(f"Stopwords for language '{self.nltk_lang}' not found by NLTK. No stopwords will be removed.")
                # Attempt to download if it was a download issue, though initial setup should handle this.
                # nltk.download('stopwords', quiet=True) # Re-download might be too aggressive here.
                # self.stopwords will remain empty if not found
            except Exception as e: # Catch any other potential errors during stopword loading
                logger.error(f"Error loading stopwords for '{self.nltk_lang}': {e}. No stopwords will be removed.")
                self.stopwords = set()

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
        if self.remove_stopwords and HAVE_NLTK and self.stopwords: # Ensure stopwords are loaded
            words = word_tokenize(text, language=self.nltk_lang)
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
            return sent_tokenize(text, language=self.nltk_lang)
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

def process_documents(input_dir: str, output_dir: str, file_pattern: str = "*.txt", language: str = 'en') -> List[str]:
    """
    Processa todos os documentos de texto em um diretório.
    
    Args:
        input_dir: Diretório com os documentos de entrada
        output_dir: Diretório para salvar os documentos processados
        file_pattern: Padrão para selecionar arquivos
        language: Idioma dos documentos ('en', 'pt', 'es')
        
    Returns:
        Lista de caminhos para os arquivos processados
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar o limpador de texto
    cleaner = TextCleaner(language=language)
    
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

def process_single_file(input_file_path: str, output_dir: str, language: str = 'en') -> Optional[str]:
    '''
    Processes a single text document, cleans it, and saves the processed version.

    Args:
        input_file_path: Path to the input text file.
        output_dir: Directory to save the processed text file and its metadata.
        language: Language of the document ('en', 'pt', 'es').

    Returns:
        Path to the processed output text file, or None if processing failed.
    '''
    logger.info(f"Processing single file: {input_file_path} for language: {language}")
    try:
        # Create TextCleaner instance for the given language
        cleaner = TextCleaner(language=language)

        # Determine output file path
        filename = os.path.basename(input_file_path)
        output_file = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True) # Ensure output_dir exists

        # Check for corresponding metadata .json file for the input
        # (e.g., if input_file_path is dir/foo.txt, metadata is dir/foo.json)
        input_metadata_file = os.path.splitext(input_file_path)[0] + ".json"
        metadata = {}
        if os.path.exists(input_metadata_file):
            with open(input_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            # If no specific JSON metadata, create basic metadata from filename
            metadata = {'source_filename': filename}


        # Read text from the input file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Process the document using TextCleaner instance
        # The process_document method in TextCleaner expects text and optional metadata
        processed_doc_data = cleaner.process_document(text=text_content, metadata=metadata)

        # Save the cleaned text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_doc_data["text"])

        # Save the (potentially updated) metadata and stats
        output_metadata_json_path = os.path.splitext(output_file)[0] + ".json"
        with open(output_metadata_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "original_metadata": processed_doc_data.get("metadata", {}), # metadata passed to process_document
                "processing_stats": processed_doc_data.get("stats", {}),
                "language_processed": language
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully processed and saved: {input_file_path} -> {output_file} (metadata: {output_metadata_json_path})")
        return output_file

    except Exception as e:
        logger.error(f"Error processing single file {input_file_path}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Teste da função de processamento
    logging.basicConfig(level=logging.INFO)
    process_documents("data/raw", "data/processed")