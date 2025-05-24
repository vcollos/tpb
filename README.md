# Vector-TPB: Banco de Dados Vetorial sobre Transtorno de Personalidade Borderline

Este projeto implementa um sistema para coleta, processamento e indexação vetorial de artigos científicos e publicações acadêmicas sobre Transtorno de Personalidade Borderline (TPB) a partir de diversas fontes abertas.

## Visão Geral

O Vector-TPB extrai automaticamente artigos científicos, preprints e outras publicações acadêmicas sobre TPB de múltiplas fontes, processa os textos e cria um índice vetorial que permite buscas semânticas e análises avançadas.

### Fontes Suportadas

- **SciELO**: Artigos científicos em português, espanhol e inglês da América Latina
- **PubMed Central**: Artigos biomédicos de acesso aberto em inglês
- **PePSIC**: Periódicos de psicologia em português e espanhol
- **medRxiv**: Preprints na área de saúde e medicina

## Estrutura do Projeto

```
vector-tpb/
├── data/                  # Diretório para armazenar dados
│   ├── raw/               # Dados brutos extraídos das fontes
│   ├── processed/         # Dados processados e limpos
│   └── index/             # Índices vetoriais gerados
├── scraping/              # Módulos para extração de dados
│   ├── scielo.py          # Extração de artigos do SciELO
│   ├── pubmed.py          # Extração de artigos do PubMed Central
│   ├── pepsic.py          # Extração de artigos do PePSIC
│   └── medrxiv.py         # Extração de preprints do medRxiv
├── processing/            # Módulos para processamento de texto
│   ├── text_cleaner.py    # Limpeza e normalização de texto
│   └── pdf_processor.py   # Extração de texto de PDFs
├── index/                 # Módulos para indexação vetorial
│   └── vector_store.py    # Criação e gerenciamento de índices
├── main.py                # Script principal de execução
└── requirements.txt       # Dependências do projeto
```

## Requisitos

- Python 3.8 ou superior
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/vcollos/tpb.git
   cd tpb
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Configuração

O projeto pode ser configurado usando variáveis de ambiente, que podem ser definidas em um arquivo `.env`.
Para começar, copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Depois, edite o arquivo `.env` para preencher os valores desejados.

### Chave de API PubMed (NCBI E-utils)

-   **Variável**: `PUBMED_API_KEY`
-   **Argumento CLI**: `--pubmed-api-key`

Utilizar uma chave de API do NCBI E-utils para o PubMed pode fornecer limites de requisição mais altos, resultando em uma extração de dados mais rápida desta fonte.

**Como definir:**

1.  **Arquivo `.env` (recomendado para desenvolvimento):**
    Adicione a seguinte linha ao seu arquivo `.env`:
    ```
    PUBMED_API_KEY="sua_chave_api_aqui"
    ```
2.  **Variável de ambiente direta:**
    Você pode definir a variável de ambiente diretamente no seu terminal:
    ```bash
    export PUBMED_API_KEY="sua_chave_api_aqui" # Linux/Mac
    set PUBMED_API_KEY="sua_chave_api_aqui"    # Windows
    ```
3.  **Argumento de Linha de Comando:**
    Você pode fornecer a chave diretamente ao executar `main.py`:
    ```bash
    python main.py --pubmed-api-key "sua_chave_api_aqui"
    ```

**Precedência:** O argumento de linha de comando (`--pubmed-api-key`) terá precedência sobre a variável de ambiente (`PUBMED_API_KEY` no arquivo `.env` ou exportada) se ambos forem fornecidos.

## Uso

### Execução Completa

Para executar o pipeline completo (extração, processamento e indexação):

```
python main.py
```

### Opções de Linha de Comando

O script principal aceita vários argumentos para personalizar a execução:

```
python main.py --query-pt "transtorno borderline" --query-en "borderline personality disorder" --max-articles 20
```

Argumentos disponíveis:
- `--skip-scraping`: Pula a etapa de extração de dados
- `--skip-processing`: Pula a etapa de processamento de texto
- `--skip-indexing`: Pula a etapa de indexação vetorial
- `--query-pt`: Define a consulta em português (padrão: "transtorno de personalidade borderline")
- `--query-en`: Define a consulta em inglês (padrão: "borderline personality disorder")
- `--max-articles`: Define o número máximo de artigos por fonte (padrão: 10)
- `--pubmed-api-key TEXT`: Chave de API para NCBI E-utils (PubMed). Também configurável via variável de ambiente `PUBMED_API_KEY`.

### Exemplos de Uso

1. Apenas extrair dados (sem processamento ou indexação):
   ```
   python main.py --skip-processing --skip-indexing
   ```

2. Processar e indexar dados já extraídos:
   ```
   python main.py --skip-scraping
   ```

3. Extrair mais artigos com consultas específicas:
   ```
   python main.py --query-pt "personalidade limítrofe tratamento" --query-en "borderline personality treatment" --max-articles 30
   ```

## Buscando no Índice Vetorial

Após a criação do índice, você pode realizar buscas semânticas usando o módulo `vector_store`:

```python
from index.vector_store import search_index

# Buscar artigos semanticamente similares à consulta
results = search_index(
    query="tratamentos eficazes para transtorno borderline",
    index_path="data/index/tpb_index.pkl",
    top_k=5
)

# Exibir resultados
for doc, score in results:
    print(f"Score: {score:.4f} - {doc.id}")
    print(f"Texto: {doc.text[:200]}...")
    print("-" * 50)
```

## Extensão

O projeto foi projetado para ser facilmente extensível:

1. Para adicionar uma nova fonte de dados, crie um novo módulo em `scraping/` seguindo o padrão dos existentes.
2. Para adicionar novos métodos de processamento, estenda os módulos em `processing/`.
3. Para implementar diferentes técnicas de indexação vetorial, modifique o módulo `vector_store.py`.

## Limitações

- A extração de dados depende da estrutura atual dos sites, que pode mudar ao longo do tempo.
- O processamento de PDFs pode não ser perfeito para todos os formatos e layouts.
- A qualidade da indexação vetorial depende dos modelos de embedding utilizados.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.