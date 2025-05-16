# PDF RAG

An advanced document processing and question-answering application built with Streamlit. This application enables you to upload PDF documents, extract their content, and ask questions about the information they contain.

## Features

- **PDF Processing**: Extract text from PDF documents with intelligent chunking
- **Semantic Search**: Use state-of-the-art embeddings to find relevant document sections
- **LLM Integration**: Connect with Groq API for AI-powered document question answering
- **Customizable Settings**: Adjust chunk size, model parameters, and more

## How It Works

1. **Text Extraction**: When you upload a document, the app extracts all text content
2. **Chunking**: The text is divided into smaller, manageable chunks
3. **Embedding**: Each chunk is converted to a vector embedding using all-MiniLM-L6-v2
4. **Semantic Search**: When you ask a question, it's compared against all chunk embeddings
5. **Result Retrieval**: The most relevant document sections are retrieved
6. **LLM Integration (Optional)**: If using the Groq API, the relevant chunks are passed to an LLM with your question to generate a comprehensive answer

## Installation

```bash
# Clone the repository
git clone https://github.com/arsh21-turing/pdf_rag.git
cd pdf_rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
streamlit run app.py
```

### Basic Usage

1. **Upload Document**: Navigate to the Upload tab and select a PDF document
2. **Set Parameters**: Configure the chunk size and overlap as needed
3. **Process Document**: Click the "Process Document" button to extract text and create embeddings
4. **Ask Questions**: Go to the Q&A tab and enter your question
5. **View Results**: Use the "Search Document" button for basic retrieval or "Get LLM Answer" (requires API key) for AI-generated responses

### API Integration

To use the LLM integration:

1. Select "Use Groq LLM API" in the sidebar
2. Enter your Groq API key
3. When asking questions, click "Get LLM Answer" for AI-generated responses

## Supported Models

### Embedding Model

- **all-MiniLM-L6-v2**: A lightweight, efficient embedding model that converts text to vector representations

### LLM Models (via Groq)

- **Llama 3 (8B)**: Fast, efficient language model for general-purpose text generation
- **Llama 3 (70B)**: More powerful language model for complex reasoning
- **Gemma (7B)**: Google's lightweight model for general-purpose text tasks
- **Mixtral 8x7B**: Powerful mixture-of-experts model with broader knowledge

## Technical Details

The application is built with:

- **Streamlit**: For the web interface
- **PDF Processing**: Custom extraction pipeline for reliable text extraction
- **Vector Embeddings**: Sentence transformers for semantic search
- **CrewAI**: For LLM integration and response generation

## Performance Considerations

- Larger documents may take longer to process
- Smaller chunk sizes increase precision but may lose context
- Larger chunk sizes preserve more context but may reduce search precision

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.