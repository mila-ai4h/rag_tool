# Project Notes

## Future Improvements

### Testing
- [ ] Add unit tests
- [ ] Add integration tests
  - Create collection
  - Add documents to collection
  - Query collection
  - Delete collection

### Infrastructure & Dependencies
- [ ] Use a proper vector store. Candidates are:
  - Pinecone
  - Qdrant
  - Chroma
  - Weaviate
- [ ] Remove dependencies only used in scripts/test_langgraph.py:
  - langgraph
  - grandalf
  
### Features & Functionality
- [ ] Support using multimodal models for document content extraction
  - Explore open source solutions (docling, markitdown)
- [ ] Support additional LLM and embedding models:
  - tinyllm integration
  - VertexAI LLMs
  - VertexAI embedding models
- [ ] Document management improvements:
  - Support removing document (and its embeddings) from a vector store. Note that this is not possible with the current llamaindex vectorstore
  - Remove existing embeddings before re-indexing the same document to prevent duplicates
- [ ] Configuration management:
  - Do we want to expose additional parameters to end users:
    - Embedding model
    - Chunk size
    - Chunk overlap
    - LLM selection
  - If we do, we would have to persist a collection's configuration for consistent usage during indexing and query
- [ ] Review llamaindex usage and necessity

### Documentation
- [ ] Enhance FastAPI documentation (http://localhost:8000/docs)

## Current Limitations


2. **Configuration**
   - Parameter changes (.env) require document re-indexing
   - No support for dynamic parameter updates

3. **Document Management**
   - Re-indexing the same document twice creates duplicate embeddings
   - No built-in deduplication

4. **API Limitations**
   - Collection creation and PDF indexing endpoint requires local server access

## Implementation Notes

### Langgraph Example
The current implementation:
1. Queries collection and retrieves LLM answer ("rag" step)
2. Passes matching chunks and answer to another LLM to get an answer the the same question ("synth" step)

**Potential Improvements:**
- Use LLM only once, or
- Use second LLM for different processing of the content
