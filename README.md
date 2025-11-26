# StandardRagDemo

A Retrieval-Augmented Generation (RAG) project using Python, LangChain, FAISS, and Google Gemini LLM.

This project demonstrates building a **RAG pipeline** that:

- Loads various document types (`PDF`, `TXT`, `CSV`, etc.)
- Splits documents into chunks for embedding
- Generates embeddings using `SentenceTransformers`
- Stores embeddings in a **FAISS vector store**/**Chroma DB**
- Retrieves relevant chunks based on a query
- Summarizes retrieved content using an LLM

---
### **Setup Instructions**

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/StandardRagDemo.git
cd StandardRagDemo
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```


3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
Create a .env file with your API keys:
GEMINI_API_KEY=your_api_key_here
```

5. **Run the app**
```bash
python -m app
```

## Key Notebook

The project includes a **comprehensive notebook** demonstrating the full **RAG pipeline**:

### 1️⃣ Document Data Structures
- Use **LangChain `Document`** objects to store documents.
- Metadata includes:
  - `source` – source file name
  - `author` – author of the document
  - `pages` – number of pages
  - `date_created` – creation date

### 2️⃣ Text and PDF Ingestion
- Create sample text files programmatically.
- Load PDFs using:
  - `PyMuPDFLoader`  
  - `TextLoader` for plain text

### 3️⃣ Chunking
- Split documents into smaller chunks for embedding accuracy.
- Use **`RecursiveCharacterTextSplitter`** with configurable:
  - `chunk_size`  
  - `chunk_overlap`

### 4️⃣ Embeddings & Vector Store
- Generate embeddings using **`Embedder`** class with **SentenceTransformer** models.
- Store embeddings in:
  - **ChromaDB** (`VectorStore`)  
  - **FAISS** index for fast similarity search
- Metadata is saved for retrieval and source tracking.

### 5️⃣ RAG Retriever
- Retrieve top-K chunks based on **cosine similarity** using **`RAGRetriever`**.
- Filters results using configurable similarity thresholds.

### 6️⃣ RAG Pipelines
- **Simple RAG**
  - Retrieve context chunks.
  - Generate response using **ChatGoogleGenerativeAI** or **ChatGroq**.

- **Advanced RAG**
  - Includes **confidence scoring**, **source tracking**, **context preview**.
  - Adjustable retrieval thresholds for filtering.
  - Returns a structured output with:
    - `answer` – LLM-generated answer
    - `sources` – source files and page numbers
    - `confidence` – highest similarity score
    - `context` – full retrieved context (optional)

---


