#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para demonstrar a integração do banco de dados vetorial com LangChain
para responder perguntas sobre Transtorno de Personalidade Borderline.
"""

import os
import logging
import argparse
import pickle
from typing import List, Dict, Any

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verificar se as bibliotecas necessárias estão disponíveis
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    HAVE_LANGCHAIN = True
except ImportError:
    logger.warning("LangChain não encontrado. Instale com 'pip install langchain langchain-community'")
    HAVE_LANGCHAIN = False

try:
    from dotenv import load_dotenv
    load_dotenv()  # Carregar variáveis de ambiente do arquivo .env
    HAVE_DOTENV = True
except ImportError:
    logger.warning("python-dotenv não encontrado. Variáveis de ambiente devem ser definidas manualmente.")
    HAVE_DOTENV = False

class LangChainQA:
    """Classe para integração do banco de dados vetorial com LangChain."""
    
    def __init__(self, index_path="data/index/tpb_index.pkl", use_openai=True):
        """
        Inicializa o sistema de QA.
        
        Args:
            index_path: Caminho para o arquivo de índice
            use_openai: Se deve usar a API OpenAI (requer chave API)
        """
        self.index_path = index_path
        self.use_openai = use_openai
        self.qa_chain = None
        
        # Verificar se o índice existe
        if not os.path.exists(index_path):
            logger.error(f"Arquivo de índice não encontrado em: {index_path}", exc_info=True) # Added exc_info
            return
        
        # Verificar se as bibliotecas necessárias estão disponíveis
        if not HAVE_LANGCHAIN:
            logger.error("Biblioteca LangChain não encontrada. Esta classe não funcionará.")
            return # Cannot proceed without LangChain
        
        # Inicializar o sistema
        logger.info("Iniciando inicialização do sistema LangChainQA...")
        self._initialize()
    
    def _initialize(self):
        """Inicializa o sistema de QA com LangChain."""
        try:
            # Carregar o índice
            logger.info(f"Carregando dados do índice de: {self.index_path}")
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            logger.info(f"Dados do índice carregados. {len(index_data.get('documents', []))} documentos encontrados no pickle.")
            
            # Extrair documentos e embeddings
            documents = []
            # Assuming index_data["embeddings"] are not directly used here if FAISS is built from documents + new embeddings
            if not index_data.get("documents"):
                logger.error(f"Nenhum documento encontrado no arquivo de índice: {self.index_path}")
                return

            for doc_dict in index_data["documents"]: # Embeddings from file are not used for LangChain FAISS
                langchain_doc = Document(
                    page_content=doc_dict["text"],
                    metadata=doc_dict.get("metadata", {})
                )
                documents.append(langchain_doc)
            
            # Inicializar embeddings
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Inicializando embeddings com HuggingFaceEmbeddings, modelo: {embedding_model_name}")
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            
            # Criar vetor store
            logger.info(f"Criando FAISS vector store a partir de {len(documents)} documentos.")
            self.vector_store = FAISS.from_documents(documents, embeddings)
            logger.info("FAISS vector store criado com sucesso.")
            
            # Configurar o modelo de linguagem
            llm_choice = "OpenAI" if self.use_openai else "Local LLM (HuggingFace)"
            logger.info(f"Configurando modelo de linguagem: {llm_choice}")

            if self.use_openai:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning("Chave API OpenAI (OPENAI_API_KEY) não encontrada no ambiente.")
                    logger.info("Recorrendo a modelo local devido à ausência da chave OpenAI.")
                    self.use_openai = False # Force fallback
                else:
                    logger.info("Usando OpenAI LLM.")
                    llm = OpenAI(temperature=0) # Add model_name if specific needed, e.g., "text-davinci-003"
            
            if not self.use_openai: # This 'if' will also catch the fallback from the block above
                logger.info("Tentando carregar modelo LLM local via HuggingFacePipeline.")
                try:
                    from langchain.llms import HuggingFacePipeline
                    # import torch # Not explicitly used here, but good for transformers
                    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                    
                    model_id = "google/flan-t5-small" 
                    logger.info(f"Carregando modelo local: {model_id}")
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id) # Or AutoModelForSeq2SeqLM for T5
                    
                    pipe = pipeline(
                        "text2text-generation" if "t5" in model_id else "text-generation", # Adjust task for T5 models
                        model=model,
                        tokenizer=tokenizer,
                        max_length=512 # Adjust as needed
                    )
                    llm = HuggingFacePipeline(pipeline=pipe)
                    logger.info(f"Modelo local {model_id} carregado com sucesso.")
                    
                except Exception as e:
                    logger.exception(f"Erro crítico ao carregar modelo LLM local '{model_id}'. O sistema de QA não funcionará.")
                    return # Cannot proceed without an LLM
            
            # Template de prompt
            logger.info("Configurando template de prompt para RetrievalQA.")
            template = """
            Você é um assistente especializado em Transtorno de Personalidade Borderline (TPB).
            Use as informações a seguir para responder à pergunta.
            Se a informação não estiver nas fontes, diga que não sabe a resposta.
            
            Fontes:
            {context}
            
            Pergunta: {question}
            
            Resposta:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Criar chain de QA
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff", # Default, good for smaller contexts
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 docs
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True # Optionally return source documents for inspection
            )
            
            logger.info("Sistema de QA (RetrievalQA chain) inicializado com sucesso.")
            
        except Exception as e:
            logger.exception("Erro fatal durante a inicialização do sistema de QA.")
    
    def answer_question(self, question: str) -> str:
        """
        Responde a uma pergunta usando o banco de dados vetorial.
        
        Args:
            question: Pergunta sobre TPB
            
        Returns:
            Resposta baseada nas fontes do banco de dados
        """
        if not self.qa_chain:
            logger.error("Tentativa de responder pergunta, mas o sistema de QA não está inicializado.")
            return "Erro: Sistema de QA não inicializado corretamente. Verifique os logs."
        
        logger.info(f"Recebida pergunta para QA: '{question[:100]}...'")
        try:
            # Executar a chain de QA
            result = self.qa_chain({"query": question})
            
            answer = result.get("result", "Nenhuma resposta encontrada.")
            source_docs = result.get("source_documents")
            if source_docs:
                logger.info(f"Resposta gerada a partir de {len(source_docs)} documentos fonte.")
                for i, doc in enumerate(source_docs):
                    logger.debug(f"Fonte {i+1}: ID='{doc.metadata.get('id', 'N/A')}', Trecho='{doc.page_content[:100]}...'")
            else:
                logger.info("Nenhum documento fonte retornado com a resposta.")

            return answer
            
        except Exception as e:
            logger.exception(f"Erro ao processar pergunta no QA chain: '{question[:100]}...'")
            return f"Erro ao processar a sua pergunta. Consulte os logs para mais detalhes."

def main():
    """Função principal para demonstrar o sistema de QA."""
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Sistema de QA sobre TPB com LangChain')
    parser.add_argument('--question', help='Pergunta sobre TPB')
    parser.add_argument('--index-path', default='data/index/tpb_index.pkl', help='Caminho para o arquivo de índice')
    parser.add_argument('--no-openai', action='store_true', help='Não usar API OpenAI (usar modelo local)')
    args = parser.parse_args()
    
    # Verificar se as bibliotecas necessárias estão disponíveis
    if not HAVE_LANGCHAIN:
        print("Este script requer LangChain. Instale com 'pip install langchain langchain-community'")
        return
    
    # Inicializar o sistema de QA
    qa_system = LangChainQA(
        index_path=args.index_path,
        use_openai=not args.no_openai
    )
    
    if args.question:
        # Modo de linha de comando
        answer = qa_system.answer_question(args.question)
        
        print("\n" + "="*80)
        print(f"PERGUNTA: {args.question}")
        print("="*80)
        print(f"\nRESPOSTA:\n{answer}")
        print("\n" + "="*80)
    else:
        # Modo interativo
        print("\n" + "="*80)
        print("SISTEMA DE QA SOBRE TRANSTORNO DE PERSONALIDADE BORDERLINE")
        print("Digite 'sair' para encerrar")
        print("="*80)
        
        while True:
            question = input("\nSua pergunta: ")
            if question.lower() in ['sair', 'exit', 'quit']:
                break
            
            answer = qa_system.answer_question(question)
            
            print("\n" + "-"*80)
            print(f"RESPOSTA:\n{answer}")
            print("-"*80)

if __name__ == "__main__":
    main()