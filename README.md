# Bella Italia — Full RAG Chatbot

A complete production ready RAG chatbot built with FastAPI, LangChain
and Pinecone. Customers can have full multi-turn conversations about
the restaurant — menu questions, FAQ questions and follow up questions
— all answered accurately from real documents with conversation memory.

## Features

- Full conversation memory — multi-turn chat with session history
- Query processing — follow up questions resolved using chat context
- RAG answers — AI answers only from real document content
- Cross document search — searches menu PDF and FAQ simultaneously
- Source tracking — every answer shows which document it came from
- Score threshold filtering — only relevant results returned to AI
- History endpoint — retrieve full conversation history
- Clear endpoint — reset conversation for a session
- Live update endpoint — refresh documents without restarting server
- Input validation — empty messages and session IDs rejected

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend web framework |
| LangChain | RAG and retrieval framework |
| Pinecone | Cloud vector database |
| HuggingFace | Free local embedding model |
| Groq API | AI language model |
| LLaMA 3.3 70B | AI model |
| PyPDF | PDF document reading |
| Pydantic | Data validation |
| python-dotenv | Environment variable management |

## Project Structure
full-rag-chatbot/
│
├── env/
├── main.py
├── menu.pdf
├── faq.txt
├── .env
└── requirements.txt

## Setup

1. Clone the repository
git clone https://github.com/yourusername/bella-italia-rag-chatbot

2. Create and activate virtual environment
python -m venv env
env\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Create `.env` file and add your API keys
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

5. Add your documents to project folder
menu.pdf  →  restaurant menu PDF
faq.txt   →  frequently asked questions

6. Run the server
uvicorn main:app --reload

## API Endpoints

### POST /chat
Main chatbot endpoint with conversation memory.

**Request:**
```json
{
    "session_id": "user_1",
    "message": "Do you have vegan food?"
}
```

**Response:**
```json
{
    "answer": "Yes we have Vegan Arrabbiata — spicy tomato pasta with no animal products priced at $12.",
    "sources": ["menu.pdf", "faq.txt"],
    "session_id": "user_1",
    "history_length": 2
}
```

### POST /history
Returns full conversation history for a session.

**Request:**
```json
{
    "session_id": "user_1"
}
```

**Response:**
```json
{
    "session_id": "user_1",
    "history": [
        {"role": "user", "content": "Do you have vegan food?"},
        {"role": "assistant", "content": "Yes we have Vegan Arrabbiata..."}
    ],
    "history_length": 2
}
```

### POST /clear
Clears conversation history for a session.

**Request:**
```json
{
    "session_id": "user_1"
}
```

**Response:**
```json
{
    "message": "History cleared for session user_1",
    "session_id": "user_1"
}
```

### POST /update
Refreshes AI knowledge when documents are updated.
No request body needed.

**Response:**
```json
{
    "message": "Documents updated successfully!"
}
```

## How It Works
menu.pdf + faq.txt
↓
PyPDFLoader and TextLoader load documents
↓
RecursiveCharacterTextSplitter splits into chunks
chunk_size=500, chunk_overlap=50
↓
HuggingFace converts chunks to 384 dimension embeddings
↓
Stored permanently in Pinecone cloud index
↓
User sends message
↓
Query processed with chat history context
follow up questions resolved automatically
↓
Retriever finds relevant chunks
score_threshold=0.5 filters irrelevant results
↓
Context and history injected into structured prompt
↓
AI generates accurate answer from real documents
↓
Answer saved to session history
↓
Response returned with sources and session info

## Query Processing

Follow up questions are automatically resolved:
User: "Do you have vegan food?"
AI:   "Yes we have Vegan Arrabbiata"
User: "How much does it cost?"
↓
Query processing resolves "it" to "Vegan Arrabbiata"
Reformulated: "How much does Vegan Arrabbiata cost?"
↓
AI: "It costs $12"  ← correct answer ✅

## Conversation History Management
Sessions stored in memory per user
Last 6 messages used for context injection
Last 4 messages used for query processing
History cleared when /clear endpoint called
History resets when server restarts

## Document Pipeline
One time setup:
Load → Split → Embed → Store in Pinecone
Subsequent restarts:
Skip embedding — data already in Pinecone
Update documents:
Call /update → delete old index → rebuild pipeline

## Retriever Configuration

```python
search_type = "similarity_score_threshold"
score_threshold = 0.5   # filters results below 50% similarity
k = 4                   # returns top 4 results
```

## Environment Variables
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

## Notes

- Never commit your .env file to GitHub
- Documents embedded once — skipped on subsequent restarts
- Call /update after changing any document
- Session history resets when server restarts
- HuggingFace model downloads automatically on first run
- Pinecone free tier — 1 index, 2GB storage, no credit card