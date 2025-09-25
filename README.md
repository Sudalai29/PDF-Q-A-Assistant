# PDF Q&A Assistant 📘

A powerful PDF question-answering system that combines document retrieval, AI-powered responses, and automatic summarization using LangChain, LangGraph, and Gradio.

## Features

- **PDF Processing**: Automatically loads and chunks PDF documents for efficient retrieval
- **Semantic Search**: Uses FAISS vector store with HuggingFace embeddings for relevant document retrieval
- **AI-Powered Q&A**: Leverages OpenRouter API with Mistral models for intelligent responses
- **Auto-Summarization**: Provides concise summaries of detailed answers
- **Interactive UI**: Clean Gradio web interface for easy interaction
- **Multi-Agent Workflow**: Uses LangGraph for orchestrated document processing pipeline

## Prerequisites

- Python 3.8+
- OpenRouter API key
- PDF document to analyze

## Installation

1. **Clone or download the script**

2. **Install required dependencies**:
```bash
pip install langchain langchain-community langchain-core langgraph gradio
pip install pypdf faiss-cpu sentence-transformers openai certifi
```

3. **Set up your OpenRouter API key**:
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Replace `"sk-or-v1-"` in the script with your actual API key

## Configuration

### PDF File Path
Update the PDF path in the script:
```python
loader = PyPDFLoader(r"Path to your file")
```
Replace with your PDF file path.

### Model Configuration
The script uses Mistral models via OpenRouter:
- **Q&A Model**: `mistralai/mistral-7b-instruct:free`    ##change the model as per your needs
- **Summary Model**: `mistralai/mistral-7b-instruct:free`

You can modify these in the `ChatOpenAI` configurations.

### Parameters You Can Adjust
- `chunk_size=1000`: Size of text chunks for processing
- `chunk_overlap=100`: Overlap between chunks
- `k=4`: Number of relevant documents to retrieve
- `temperature=0`: Creativity level (0 = deterministic, 1 = creative)

## Usage

### Running the Application

```bash
python your_script_name.py
```

This will:
1. Load and process your PDF
2. Create embeddings and vector store
3. Launch a Gradio web interface
4. Open your browser automatically (usually at `http://127.0.0.1:7860`)

### Using the Interface

1. **Enter your question** in the text box
2. **Click Submit** or press Enter
3. **View results**:
   - **Detailed Answer**: Comprehensive response based on PDF content
   - **Summary**: Concise 50-word summary of the answer

### Example Questions

- "What are the key AWS services covered?"
- "Explain the difference between EC2 and Lambda"
- "What are the security best practices mentioned?"
- "Summarize the networking concepts"

## Architecture

### Workflow Pipeline
```
User Question → Document Retrieval → Answer Generation → Summarization → Display Results
```

### Components

1. **Document Loader**: PyPDFLoader processes PDF files
2. **Text Splitter**: RecursiveCharacterTextSplitter creates manageable chunks
3. **Embeddings**: HuggingFace sentence-transformers for semantic understanding
4. **Vector Store**: FAISS for efficient similarity search
5. **LLM**: Mistral models via OpenRouter for generation
6. **Orchestration**: LangGraph manages the multi-step workflow
7. **UI**: Gradio provides the web interface

### LangGraph State Flow
```python
GraphState: question → docs → answer → summary
```

## Customization

### Adding New Models
```python
llm = ChatOpenAI(
    model="your-preferred-model",
    temperature=0,
    openai_api_key="your-key",
    openai_api_base="https://openrouter.ai/api/v1"
)
```

### Modifying Prompts
```python
qa_prompt = PromptTemplate.from_template(
    """Your custom prompt here.
    Context: {context}
    Question: {question}
    Answer:"""
)
```

### Changing Chunk Parameters
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Larger chunks
    chunk_overlap=200  # More overlap
)
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**:
   - The script includes SSL configuration with certifi
   - Ensure `certifi` is installed: `pip install certifi`

2. **API Key Issues**:
   - Verify your OpenRouter API key is correct
   - Check your account has sufficient credits

3. **PDF Loading Errors**:
   - Ensure the PDF path is correct and accessible
   - Try using forward slashes or raw strings for paths

4. **Memory Issues**:
   - Reduce `chunk_size` for large PDFs
   - Decrease the number of retrieved documents (`k` parameter)

### Performance Tips

- **Large PDFs**: Consider increasing `chunk_size` to 1500-2000
- **Better Accuracy**: Increase `k` to retrieve more relevant documents
- **Faster Processing**: Use smaller embedding models or reduce chunk overlap

## Dependencies

```
langchain
langchain-community  
langchain-core
langgraph
gradio
pypdf
faiss-cpu
sentence-transformers
openai
certifi
```

## License

This project is open source. Please ensure you comply with the terms of service for:
- OpenRouter API
- HuggingFace models
- All other dependencies

## Contributing

Feel free to submit issues and enhancement requests!

## Future Enhancements

- [ ] Support for multiple PDF files
- [ ] Chat history functionality
- [ ] Advanced filtering options
- [ ] Export functionality for Q&A pairs
- [ ] Integration with local LLMs
- [ ] Support for other document formats

---

**Note**: Remember to keep your API keys secure and never commit them to version control!
