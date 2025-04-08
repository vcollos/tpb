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
            logger.error(f"Índice não encontrado: {index_path}")
            return
        
        # Verificar se as bibliotecas necessárias estão disponíveis
        if not HAVE_LANGCHAIN:
            logger.error("LangChain é necessário para este script.")
            return
        
        # Inicializar o sistema
        self._initialize()
    
    def _initialize(self):
        """Inicializa o sistema de QA com LangChain."""
        try:
            # Carregar o índice
            logger.info(f"Carregando índice de {self.index_path}")
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Extrair documentos e embeddings
            documents = []
            for doc_dict, embedding in zip(index_data["documents"], index_data["embeddings"]):
                langchain_doc = Document(
                    page_content=doc_dict["text"],
                    metadata=doc_dict.get("metadata", {})
                )
                documents.append(langchain_doc)
            
            # Inicializar embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Criar vetor store
            self.vector_store = FAISS.from_documents(documents, embeddings)
            
            # Configurar o modelo de linguagem
            if self.use_openai:
                # Verificar se a chave API está disponível
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning("Chave API OpenAI não encontrada. Defina a variável de ambiente OPENAI_API_KEY.")
                    logger.info("Usando fallback para modelo local.")
                    self.use_openai = False
            
            if self.use_openai:
                # Usar OpenAI
                llm = OpenAI(temperature=0)
            else:
                # Usar modelo local via LangChain
                try:
                    from langchain.llms import HuggingFacePipeline
                    import torch
                    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                    
                    # Carregar modelo local (exemplo com um modelo pequeno)
                    model_id = "google/flan-t5-small"  # Pode ser substituído por outro modelo
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_length=512
                    )
                    
                    llm = HuggingFacePipeline(pipeline=pipe)
                    
                except Exception as e:
                    logger.error(f"Erro ao carregar modelo local: {str(e)}")
                    logger.error("Não foi possível inicializar o sistema de QA.")
                    return
            
            # Template de prompt
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
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info("Sistema de QA inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar o sistema de QA: {str(e)}")
    
    def answer_question(self, question):
        """
        Responde a uma pergunta usando o banco de dados vetorial.
        
        Args:
            question: Pergunta sobre TPB
            
        Returns:
            Resposta baseada nas fontes do banco de dados
        """
        if not self.qa_chain:
            return "Sistema de QA não inicializado corretamente."
        
        try:
            # Executar a chain de QA
            result = self.qa_chain({"query": question})
            
            # Extrair e retornar a resposta
            return result["result"]
            
        except Exception as e:
            logger.error(f"Erro ao responder pergunta: {str(e)}")
            return f"Erro ao processar a pergunta: {str(e)}"

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