from root_config import *
from utils.init import *
# ================================================== #
from utils.db_config import DB
from RAG.web_weaviate import WebWeaviate
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from RAG.web_researcher import WebResearchRetriever
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from RAG.web_researcher import WebResearchRetriever
from preprocessing.embeddings_class import AnglEEmbedding
from RAG.weaviate_class import Weaviate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    
    def __init__(self, embedding_model:Embeddings) -> None:
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, streaming=True, openai_api_key=OPEN_AI_KEY)
        self.client = DB().connect()
        self.embedding_model = embedding_model
        self.db = Weaviate(self.client, embedder=self.embedding_model)
        self.web_client = weaviate.connect_to_local()
        self.web_db = WebWeaviate(self.web_client, embedder=self.embedding_model)
        self.search =  DuckDuckGoSearchAPIWrapper()
        self.web_retriever = WebResearchRetriever.from_llm( vectorstore=self.web_db , llm=self.llm,  search=self.search)

    def get_page_text(self, doc_id: str, page_no: int):
        col = self.client.collections.get("bookipedia")
        filters = wvc.query.Filter.by_property("source_id").equal(doc_id) & wvc.query.Filter.by_property("page_no").equal(page_no)
        res = col.query.fetch_objects(filters= filters, limit = FETCHING_LIMIT, sort = SORT)
        text = ''
        for o in res.objects:
            text += o.properties["text"] + '\n'
        return text


    def generate_retrieval_query(self, user_prompt:str, chat_summary:str):

        # Define the retrieval query template
        RETRIEVING_QUERY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing user prompt for better chuncks retrieval from web and vectorstore.
            Given a user prompt and a summary of the chat, your task is to generate a retrieval query that is both concise and effective.
            The query should be designed to retrieve the most relevant information efficiently.
            Inputs: User Prompt: '''{user_prompt}'''
            Chat Summary: '''{chat_summary}'''""",
        )

        # Create the retrieval query chain
        retrieving_query_chain = RETRIEVING_QUERY_PROMPT | self.llm

        # Invoke the retrieval query chain with the user prompt and chat summary
        retrieving_query = retrieving_query_chain.invoke({"user_prompt": user_prompt, "chat_summary": chat_summary})

        # Return the content of the retrieval query
        return retrieving_query.content
    

    
    
    def generate_context(self, user_prompt: str, chat_summary: str, doc_ids: list[str] = None, enable_web_retrieval=True) :
        
        def generate_vecdb_context( retrieval_query: str, doc_ids: list[str]):
        
            docs = self.db.similarity_search(query=retrieval_query, source_ids=doc_ids, auto_merge = False)
            

            content = [doc.page_content for doc in docs]
            metadata = [{"doc_id": doc.metadata['source_id'], "page_no": doc.metadata['page_no'], "text": doc.page_content} for doc in docs]
            
            return content, metadata
        
        def generate_web_context( retrieval_query: str):
            docs = self.web_retriever.invoke(retrieval_query)

            content = [doc.page_content for doc in docs]
            metadata = [doc.metadata['source'] for doc in docs]
            return content, list(set(metadata))

        
        #  Generate the retrieval query
        retrieval_query = self.generate_retrieval_query(user_prompt, chat_summary)
        
        #  Generate context from the Weaviate Vector Database
        vecdb_context, vecdb_metadata = [] , []
        if doc_ids:
            vecdb_context, vecdb_metadata = generate_vecdb_context(retrieval_query, doc_ids)
        
        # Optionally generate context from the web (if doc_ids are provided)
        web_context , web_metadata = [] , []
        
        if enable_web_retrieval:
            web_context, web_metadata = generate_web_context(retrieval_query)
        
        # Combine the contexts
        context = vecdb_context + web_context
        metadata = vecdb_metadata + web_metadata
                
        return context, metadata
    

    def generate_chat_summary(self, response: str, user_question: str, summary: str):
        # Define the chat summary prompt template
        CHAT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing the process of summarizing user chat .
            Given the previous chat history, user prompt, and LLM response, summarize the chat effectively.
            Focus on new information and important highlights.
            Previous Chat Summary: '''{chat}'''
            User Question: '''{user_question}'''
            LLM Response: '''{response}'''""",
        )

        # Create the chat summary chain
        chat_summary_chain = CHAT_SUMMARY_PROMPT | self.llm

        # Invoke the chat summary chain with the chat, retrieving_query, and response
        chat_summary = chat_summary_chain.invoke({"chat": summary, "user_question": user_question, "response": response})

        # Return the content of the chat summary
        return chat_summary.content

    def generate_answer(self, user_prompt: str, chat_summary: str, chat: str,doc_ids: list[str] = None, enable_web_retrieval=True):
        
        self.context, self.metadata = self.generate_context(user_prompt=user_prompt, chat_summary= chat_summary, doc_ids= doc_ids, enable_web_retrieval= enable_web_retrieval)
        
        
        QA_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with answering user question.
            Given the previous chat history, user question, and relevant context, provide a clear, concise, and informative answer.
            Your response should be structured in a way that is easy to understand and directly addresses the user's question.
            Make sure to highlight the most relevant information from the context and link any sources or references appropriately.
            Previous Chat History: '''{chat}'''
            User Question: '''{user_prompt}'''
            Relevant Context: '''{context}''' """
        )


        qa_chain = QA_PROMPT | self.llm | StrOutputParser()

        return qa_chain.astream({"chat": chat, "user_prompt": user_prompt, "context": ('\n'.join(self.context))})