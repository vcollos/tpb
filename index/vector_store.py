#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para criação e gerenciamento de índices vetoriais.
"""

import os
import logging
import glob
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional

# Configuração de logging
logger = logging.getLogger(__name__)

# Tentar importar bibliotecas para embeddings e armazenamento vetorial
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    logger.warning("NumPy não encontrado. A indexação vetorial será limitada.")
    HAVE_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers não encontrado. Usando fallback para embeddings.")
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    logger.warning("FAISS não encontrado. Usando fallback para busca vetorial.")
    HAVE_FAISS = False

class Document:
    """Classe para representar um documento no índice vetorial."""
    
    def __init__(self, id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Inicializa um documento.
        
        Args:
            id: Identificador único do documento
            text: Texto do documento
            metadata: Metadados do documento (opcional)
        """
        self.id = id
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Document(id='{self.id}', text='{self.text[:50]}...', metadata={self.metadata})"

class VectorStore:
    """Classe para armazenamento e busca vetorial de documentos."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa o armazenamento vetorial.
        
        Args:
            embedding_model: Nome ou caminho do modelo de embedding
        """
        self.documents = []
        self.embeddings = []
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index = None
        self.dimension = 384  # Dimensão padrão para all-MiniLM-L6-v2
        
        # Inicializar modelo de embedding
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Inicializa o modelo de embedding."""
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Carregando modelo de embedding: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Modelo de embedding '{self.embedding_model_name}' carregado com dimensão: {self.dimension}")
            except Exception as e:
                logger.exception(f"Erro fatal ao carregar modelo de embedding '{self.embedding_model_name}'. As embeddings não funcionarão corretamente.")
                self.embedding_model = None # Ensure it's None if loading failed
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Gera embedding para um texto.
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Vetor de embedding
        """
        if not text.strip():
            # Retornar vetor de zeros para texto vazio
            return [0.0] * self.dimension
        
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode(text).tolist()
            except Exception as e:
                logger.error(f"Erro ao gerar embedding com o modelo '{self.embedding_model_name}': {e}", exc_info=True)
                logger.warning("Recorrendo a embeddings aleatórias devido a erro no modelo principal.")
        else:
            logger.warning(f"Modelo de embedding '{self.embedding_model_name}' não está disponível. Recorrendo a embeddings aleatórias.")
        
        # Fallback: vetor aleatório normalizado
        if HAVE_NUMPY:
            logger.debug("Gerando embedding aleatória normalizada com NumPy.")
            random_vector = np.random.randn(self.dimension)
            normalized = random_vector / np.linalg.norm(random_vector)
            return normalized.tolist()
        else:
            # Fallback sem NumPy
            import random
            import math
            random_vector = [random.gauss(0, 1) for _ in range(self.dimension)]
            magnitude = math.sqrt(sum(x*x for x in random_vector))
            return [x/magnitude for x in random_vector]
    
    def add_document(self, document: Document) -> None:
        """
        Adiciona um documento ao índice.
        
        Args:
            document: Documento a ser adicionado
        """
        embedding = self._get_embedding(document.text)
        self.documents.append(document)
        self.embeddings.append(embedding)
        self.index = None  # Invalidar índice existente
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Adiciona múltiplos documentos ao índice.
        
        Args:
            documents: Lista de documentos a serem adicionados
        """
        for document in documents:
            self.add_document(document)
    
    def _build_index(self) -> None:
        """Constrói o índice FAISS para busca eficiente."""
        if not self.embeddings:
            logger.warning("Nenhum documento para indexar.")
            return
        
        if not HAVE_FAISS or not HAVE_NUMPY:
            logger.warning("FAISS ou NumPy não disponível. Índice não será construído.")
            return
        
        try:
            logger.info("Construindo índice FAISS...")
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings_array)
            logger.info(f"Índice FAISS construído com {self.index.ntotal} vetores.")
        except Exception as e:
            logger.exception("Erro crítico ao construir índice FAISS.")
            self.index = None # Ensure index is None on failure
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Busca documentos semanticamente similares à consulta.
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (documento, score de similaridade)
        """
        if not self.documents:
            logger.warning("Nenhum documento no índice.")
            return []
        
        # Gerar embedding para a consulta
        query_embedding = self._get_embedding(query)
        
        # Usar FAISS se disponível
        if HAVE_FAISS and HAVE_NUMPY:
            if self.index is None:
                self._build_index()
            
            if self.index is not None:
                try:
                    query_array = np.array([query_embedding]).astype('float32')
                    distances, indices = self.index.search(query_array, min(top_k, len(self.documents)))
                    
                    results = []
                    for i, idx in enumerate(indices[0]):
                        if idx < len(self.documents):  # Verificar se o índice é válido
                            # Converter distância L2 para similaridade (1 / (1 + distância))
                            similarity = 1.0 / (1.0 + distances[0][i])
                            results.append((self.documents[idx], similarity))
                    
                    return results
                except Exception as e:
                    logger.exception("Erro durante a busca com FAISS.")
                    logger.warning("Recorrendo à busca por similaridade de cosseno devido a erro no FAISS.")
        
        # Fallback: busca por similaridade de cosseno
        if not (HAVE_FAISS and self.index): # Log fallback only if FAISS was supposed to be used or failed
            if HAVE_FAISS and not self.index:
                 logger.warning("FAISS disponível, mas índice não construído. Usando similaridade de cosseno.")
            elif not HAVE_FAISS:
                 logger.warning("FAISS não disponível. Usando similaridade de cosseno.")


        if HAVE_NUMPY:
            logger.debug("Realizando busca por similaridade de cosseno com NumPy.")
            try:
                query_array = np.array(query_embedding)
                similarities = []
                
                for i, doc_embedding in enumerate(self.embeddings):
                    doc_array = np.array(doc_embedding)
                    # Similaridade de cosseno
                    similarity = np.dot(query_array, doc_array) / (
                        np.linalg.norm(query_array) * np.linalg.norm(doc_array)
                    )
                    similarities.append((i, similarity))
                
                # Ordenar por similaridade (decrescente)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                results = []
                for idx, similarity in similarities[:top_k]:
                    results.append((self.documents[idx], similarity))
                
                return results
            except Exception as e:
                logger.exception("Erro durante a busca por similaridade de cosseno com NumPy.")
        else:
            logger.warning("NumPy não disponível. Recorrendo à busca por similaridade de cosseno com Python puro (pode ser lento).")
        
        # Fallback sem NumPy
        logger.debug("Realizando busca por similaridade de cosseno com Python puro.")
        import math
        
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a*b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a*a for a in vec1))
            magnitude2 = math.sqrt(sum(b*b for b in vec2))
            if magnitude1 * magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Ordenar por similaridade (decrescente)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, similarity in similarities[:top_k]:
            results.append((self.documents[idx], similarity))
        
        return results
    
    def save(self, path: str) -> bool:
        """
        Salva o índice em um arquivo.
        
        Args:
            path: Caminho para salvar o índice
            
        Returns:
            Boolean indicando sucesso
        """
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Preparar dados para serialização
            data = {
                "documents": [
                    {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
                    for doc in self.documents
                ],
                "embeddings": self.embeddings,
                "embedding_model": self.embedding_model_name,
                "dimension": self.dimension
            }
            
            # Salvar com pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Índice com {len(self.documents)} documentos salvo em: {path}")
            return True
            
        except Exception as e:
            logger.exception(f"Erro fatal ao salvar índice em '{path}'.")
            return False
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """
        Carrega um índice de um arquivo.
        
        Args:
            path: Caminho do arquivo de índice
            
        Returns:
            Instância de VectorStore
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Criar instância
            store = cls(embedding_model=data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
            store.dimension = data.get("dimension", 384)
            
            # Carregar documentos
            store.documents = [
                Document(doc["id"], doc["text"], doc.get("metadata", {}))
                for doc in data["documents"]
            ]
            
            # Carregar embeddings
            store.embeddings = data["embeddings"]
            
            logger.info(f"Índice carregado de '{path}' com {len(store.documents)} documentos. Modelo de embedding: {store.embedding_model_name}")
            return store
            
        except Exception as e:
            logger.exception(f"Erro fatal ao carregar índice de '{path}'. Retornando VectorStore vazio.")
            return cls() # Return a new, empty store

def create_index(file_paths: List[str], output_path: str) -> bool:
    """
    Cria um índice vetorial a partir de uma lista de caminhos de arquivos.
    
    Args:
        file_paths: Lista de caminhos para os documentos processados.
        output_path: Caminho para salvar o índice.
        
    Returns:
        Boolean indicando sucesso
    """
    # Inicializar o armazenamento vetorial
    store = VectorStore()
    
    if not file_paths:
        logger.warning("Nenhuma lista de arquivos fornecida para create_index. O índice estará vazio.")
        # Still save an empty index for consistency, or handle as an error?
        # For now, let it save an empty index.
    else:
        logger.info(f"Iniciando criação de índice para {len(file_paths)} arquivos. Saída: {output_path}")

    successful_indexed_count = 0
    failed_files = []

    for input_file in file_paths:
        try:
            logger.debug(f"Processando arquivo para indexação: {input_file}")
            doc_id = os.path.basename(input_file)
            
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"Arquivo '{input_file}' está vazio ou contém apenas espaços em branco. Pulando indexação.")
                failed_files.append(input_file)
                continue

            metadata_file = os.path.splitext(input_file)[0] + ".json"
            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    logger.debug(f"Metadados carregados para {doc_id} de {metadata_file}")
                except json.JSONDecodeError:
                    logger.error(f"Erro ao decodificar JSON de metadados: {metadata_file}. Usando metadados vazios.", exc_info=True)
                except Exception as e:
                    logger.error(f"Erro ao carregar metadados de {metadata_file}: {e}. Usando metadados vazios.", exc_info=True)
            else:
                logger.debug(f"Nenhum arquivo de metadados encontrado para {doc_id} em {metadata_file}. Usando metadados vazios.")
            
            document = Document(doc_id, text, metadata)
            store.add_document(document)
            successful_indexed_count += 1
            logger.info(f"Documento '{doc_id}' adicionado ao índice para construção.")
            
        except Exception as e:
            logger.exception(f"Erro crítico ao processar e indexar o arquivo '{input_file}'.")
            failed_files.append(input_file)
    
    logger.info(f"Total de documentos adicionados para indexação: {successful_indexed_count} de {len(file_paths)}.")
    if failed_files:
        logger.warning(f"{len(failed_files)} arquivos falharam ao serem processados para indexação: {failed_files}")

    # Construir o índice FAISS após adicionar todos os documentos
    if successful_indexed_count > 0:
        store._build_index() # Explicitly build index after all docs are added

    # Salvar o índice
    success = store.save(output_path)
    
    if success:
        logger.info(f"Índice criado e salvo com sucesso em: {output_path} ({successful_indexed_count} documentos indexados).")
    else:
        logger.error(f"Falha ao salvar o índice em: {output_path}")
    
    return success

def search_index(query: str, index_path: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """
    Busca documentos em um índice existente.
    
    Args:
        query: Texto da consulta
        index_path: Caminho para o arquivo de índice
        top_k: Número de resultados a retornar
        
    Returns:
        Lista de tuplas (documento, score de similaridade)
    """
    # Verificar se o índice existe
    if not os.path.exists(index_path):
        logger.error(f"Índice para busca não encontrado em: {index_path}")
        return []
    
    # Carregar o índice
    logger.info(f"Carregando índice de {index_path} para busca.")
    store = VectorStore.load(index_path)
    if not store.documents: # Check if store loading failed or index is empty
        logger.warning(f"Índice em {index_path} está vazio ou não pôde ser carregado corretamente.")
        return []
    
    # Realizar a busca
    logger.info(f"Realizando busca no índice com query: '{query[:100]}...' (top_k={top_k})")
    results = store.search(query, top_k=top_k)
    
    logger.info(f"Busca concluída. {len(results)} resultados encontrados para a query.")
    return results

if __name__ == "__main__":
    # Teste das funções
    logging.basicConfig(level=logging.INFO)
    
    # Criar índice de exemplo
    # Use glob to find sample files for testing
    sample_files_for_testing = glob.glob(os.path.join("data", "processed", "*.txt"))
    if sample_files_for_testing:
        # Optionally, limit the number of files for a quick test, e.g., sample_files_for_testing[:3]
        create_index(sample_files_for_testing, "data/index/tpb_index.pkl")
    else:
        logger.warning("No sample files found in data/processed for testing create_index in __main__.")
    
    # Testar busca
    results = search_index("tratamentos para transtorno borderline", "data/index/tpb_index.pkl")
    
    for doc, score in results:
        print(f"Score: {score:.4f} - {doc.id}")
        print(f"Texto: {doc.text[:200]}...")
        print("-" * 50)