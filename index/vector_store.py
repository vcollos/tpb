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
                logger.info(f"Modelo carregado com dimensão de embedding: {self.dimension}")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo de embedding: {str(e)}")
                self.embedding_model = None
    
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
                logger.error(f"Erro ao gerar embedding: {str(e)}")
        
        # Fallback: vetor aleatório normalizado
        if HAVE_NUMPY:
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
            logger.info(f"Índice construído com {len(self.embeddings)} documentos.")
        except Exception as e:
            logger.error(f"Erro ao construir índice FAISS: {str(e)}")
            self.index = None
    
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
                    logger.error(f"Erro na busca FAISS: {str(e)}")
        
        # Fallback: busca por similaridade de cosseno
        if HAVE_NUMPY:
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
                logger.error(f"Erro na busca por similaridade de cosseno: {str(e)}")
        
        # Fallback sem NumPy
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
            
            logger.info(f"Índice salvo em: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar índice: {str(e)}")
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
            
            logger.info(f"Índice carregado de: {path} com {len(store.documents)} documentos")
            return store
            
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {str(e)}")
            return cls()

def create_index(input_dir: str, output_path: str, file_pattern: str = "*.txt") -> bool:
    """
    Cria um índice vetorial a partir de documentos em um diretório.
    
    Args:
        input_dir: Diretório com os documentos processados
        output_path: Caminho para salvar o índice
        file_pattern: Padrão para selecionar arquivos
        
    Returns:
        Boolean indicando sucesso
    """
    # Inicializar o armazenamento vetorial
    store = VectorStore()
    
    # Encontrar todos os arquivos que correspondem ao padrão
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    logger.info(f"Encontrados {len(input_files)} arquivos para indexação")
    
    # Processar cada arquivo
    for input_file in input_files:
        try:
            # Extrair ID do documento (nome do arquivo)
            doc_id = os.path.basename(input_file)
            
            # Ler o texto do arquivo
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Verificar se existe arquivo de metadados correspondente
            metadata_file = os.path.splitext(input_file)[0] + ".json"
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Criar documento e adicionar ao índice
            document = Document(doc_id, text, metadata)
            store.add_document(document)
            
            logger.info(f"Documento indexado: {doc_id}")
            
        except Exception as e:
            logger.error(f"Erro ao indexar {input_file}: {str(e)}")
    
    # Salvar o índice
    success = store.save(output_path)
    
    if success:
        logger.info(f"Índice criado com sucesso: {output_path}")
    else:
        logger.error(f"Falha ao criar índice: {output_path}")
    
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
        logger.error(f"Índice não encontrado: {index_path}")
        return []
    
    # Carregar o índice
    store = VectorStore.load(index_path)
    
    # Realizar a busca
    results = store.search(query, top_k=top_k)
    
    logger.info(f"Busca concluída. {len(results)} resultados encontrados.")
    return results

if __name__ == "__main__":
    # Teste das funções
    logging.basicConfig(level=logging.INFO)
    
    # Criar índice de exemplo
    create_index("data/processed", "data/index/tpb_index.pkl")
    
    # Testar busca
    results = search_index("tratamentos para transtorno borderline", "data/index/tpb_index.pkl")
    
    for doc, score in results:
        print(f"Score: {score:.4f} - {doc.id}")
        print(f"Texto: {doc.text[:200]}...")
        print("-" * 50)