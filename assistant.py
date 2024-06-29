from typing import Optional
import json
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.tools.yfinance import YFinanceTools

from phi.llm.openai import OpenAIChat
from phi.assistant.duckdb import DuckDbAssistant

db_url = "postgresql+psycopg://ai:ai@localhost:5532/finanzas"


def get_auto_rag_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Auto RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # Define the embeddings table based on the embeddings model
    embeddings_table = (
        "auto_rag_documents_groq_ollama" if embeddings_model == "nomic-embed-text" else "auto_rag_documents_groq_openai"
    )



    return DuckDbAssistant(
        name="auto_rag_assistant_groq",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="auto_rag_assistant_groq", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 3 references are added to the prompt
            num_documents=3,
        ),
        description="Eres un experto en analisis análisis financiero para microcreditos. Con conocimientos en  Evaluación de Riesgo Crediticio, Análisis de Capacidades de Pago, Conocimiento del Entorno Socioeconómico, Gestión de Cartera de Microcréditos.",
        instructions=[
            "Primero obtén información adicional sobre la pregunta del usuario.",
            "Puedes utilizar la herramienta search_knowledge_base para buscar en tu base de conocimientos o la herramienta duckduckgo_search para buscar en Internet.",
            "Si el usuario pregunta sobre finanzas , utiliza la herramienta YFinanceTools para responder cualquier consulta con finanzas.",
            "Si el usuario pregunta sobre eventos actuales, utiliza la herramienta duckduckgo_search para buscar en Internet.",
            "Si el usuario pide un resumen de la conversación, utiliza la herramienta get_chat_history para obtener el historial de chat con el usuario.",
            "Procesa cuidadosamente la información que has reunido y proporciona una respuesta clara y concisa al usuario.",
            "responde siempre en español"
            "Responde directamente al usuario con tu respuesta, no digas 'aquí está la respuesta' o 'esta es la respuesta' o 'según la información proporcionada'",
            "NUNCA menciones tu base de conocimientos ni digas 'según la herramienta search_knowledge_base' o 'según la herramienta {some_tool}'."
        ],

        semantic_model=json.dumps(
            {
                "tables": [
                    {
                        "name": "historial transacciones",
                        "description": "Contiene el historila de transaciones entre 2022 y 2023.",
                        "path": "transactions_2022_2023.csv",
                    }
                ]
            }
        ),
        # Show tool calls in the chat
        show_tool_calls=False,
        # This setting gives the LLM a tool to search for information
        search_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        tools=[DuckDuckGo(),YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            company_info=True,
        )],
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # Adds chat history to messages
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )