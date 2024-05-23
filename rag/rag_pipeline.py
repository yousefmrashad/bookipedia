# Utils
from root_config import *
from utils.init import *

from rag.web_weaviate import WebWeaviate
from rag.web_researcher import WebResearchRetriever
from rag.weaviate_retriever import Weaviate
# ===================================================================== #

class RAGPipeline:
    def __init__(self, embedding_model: Embeddings, client: WeaviateClient) -> None:
        self.llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0, streaming=True, openai_api_key=OPEN_AI_KEY)
        self.embedding_model = embedding_model
        self.client = client
        self.db = Weaviate(self.client, self.embedding_model)
        # self.web_db = WebWeaviate(self.client, embedder=self.embedding_model)
        # self.search =  DuckDuckGoSearchAPIWrapper(backend="html")
        # self.web_retriever = WebResearchRetriever.from_llm(vectorstore=self.web_db , llm=self.llm,  search=self.search)
    # --------------------------------------------------------------------- #

    def get_page_text(self, doc_id: str, page_no: int) -> str:
        col = self.client.collections.get(DB_NAME)
        filters = id_filter(doc_id) & page_filter(page_no)
        res = col.query.fetch_objects(filters=filters, limit=FETCHING_LIMIT, sort=SORT)
        texts = [o.properties["text"] for o in res.objects]
        text = merge_chunks_v2(texts)
        
        return text
    # --------------------------------------------------------------------- #
    
    # -- Summary Methods -- #

    def summary_splitter(self, text: str) -> list[str]:
        token_count = count_tokens(text)
        no_splits = math.ceil(token_count / SUMMARY_TOKEN_LIMIT)
        chunk_size = token_count // no_splits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP,
                                                        length_function=count_tokens,
                                                        separators=SEPARATORS,
                                                        is_separator_regex=True)
        chunks = text_splitter.split_text(text)
        return chunks
    # --------------------------------------------------------------------- #
    
    async def summarize_text(self, text: str) -> str:
        SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
            You are an assistant tasked with summarizing the provided text.
            Your goal is to create a concise summary that captures the essence of the original content,
            focusing on the most important points and key information.
            Please ensure that the summary is clear, informative, and easy to understand.
            THE INPUT TEXT: ```{text}```  """)
        
        self.summary_chain = SUMMARY_PROMPT | self.llm | StrOutputParser()
        
        chunks = self.summary_splitter(text)
        chunks_summaries = await asyncio.gather(*(self.summary_chain.ainvoke({"text": t}) for t in chunks))
        summary = '\n\n'.join(chunks_summaries)
        
        return summary
    # --------------------------------------------------------------------- #
    
    async def summarize_pages(self, doc_id: str, page_nos: list[int]) -> AsyncIterator[str]:
        pages_text = '\n\n'.join([self.get_page_text(doc_id , page_no) for page_no in page_nos])

        summary = await self.summarize_text(text=pages_text)

        # Re-summerize the summary if it's too fucking long [idk if that's right]
        if (count_tokens(summary) > SUMMARY_TOKEN_LIMIT):
            summary = self.summarize_text(text=summary)

        return self.summary_chain.astream({"text": summary})
    # --------------------------------------------------------------------- #

    # Chat Summary #
    def generate_chat_summary(self, user_prompt: str, llm_response: str, prev_chat_summary: str):
        
        # Define the chat summary prompt template
        CHAT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing the process of summarizing user chat .
            Given the previous chat history, user prompt, and LLM response, summarize the chat effectively.
            Focus on new information and important highlights.
            Previous Chat Summary: ```{chat_summary}```
            User Question: ```{user_prompt}```
            LLM Response: ```{llm_response}```""",
        )

        # Create the chat summary chain
        chat_summary_chain = CHAT_SUMMARY_PROMPT | self.llm

        # Invoke the chat summary chain with the chat, retrieving_query, and response
        llm_input = {
            "chat_summary": prev_chat_summary,
            "user_prompt": user_prompt,
            "llm_response": llm_response
        }
        chat_summary = chat_summary_chain.invoke(llm_input)

        # Return the content of the chat summary
        return chat_summary.content
    # --------------------------------------------------------------------- #

    # -- Retrieval Methods -- #
    def generate_retrieval_query(self, user_prompt: str, chat_summary: str) -> tuple[str, str]:
        retrieval_method_schema = ResponseSchema(name="retrieval_method",
                                                 description="This is a string can only be one of the following words: (vectorstore, web, vectorstore+web, none)")
        retrieval_query_schema = ResponseSchema(name="retrieval_query",
                                                description="This is the retrieval query")
        
        response_schemas = [retrieval_method_schema, retrieval_query_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # Define the retrieval query template
        RETRIEVAL_QUERY_PROMPT = ChatPromptTemplate.from_template(
            """You are an intelligent assistant responsible for determining the appropriate method to retrieve information
            based on a given user prompt and a chat summary.
            Here are the steps to follow:

            (1) Extract the Retrieval Method:
            Analyze the user prompt only to decide if there is any specific request about how to retrieve information.
            
            Choose from the following methods:

            Vectorstore: Use this method if the user didn't point to any retrieving method in the prompt
            or if the information is likely in the document that the user is currently reading.

            Web: Use this method if the information is likely to be found on the internet
            or requires the latest, up-to-date information.

            Hybrid: Use this method if the information could benefit from both local dataset retrieval
            and the latest information from the internet, ensuring comprehensive coverage.

            None: Use this method if the user is asking about something mentioned in the chat history
            or if the information is unlikely to be found in the document.

            (2) Generate the Retrieval Query: Formulate a precise and effective retrieval query based on the user's prompt and the chat summary.

            User Prompt: ```{user_prompt}```
            Chat Summary: ```{chat_summary}```

            {format_instructions}
            """
        )

        # Create the retrieval query chain
        retrieval_query_chain = RETRIEVAL_QUERY_PROMPT | self.llm

        # Invoke the retrieval query chain with the user prompt and chat summary
        response = retrieval_query_chain.invoke({"user_prompt": user_prompt, "chat_summary": chat_summary, "format_instructions": format_instructions})
        response_dict = output_parser.parse(response.content)

        # Return the content of the retrieval query
        # return response_dict["retrieval_method"], response_dict["retrieval_query"]
        return "Vectorstore", response_dict["retrieval_query"]
    # --------------------------------------------------------------------- #

    # Context generation methods
    def generate_vecdb_context(self, retrieval_query: str, doc_ids: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        docs = self.db.similarity_search(query=retrieval_query, source_ids=doc_ids, auto_merge=True)

        content = [doc.page_content for doc in docs]
        metadata = [{"doc_id": doc.metadata['source_id'], "page_no": doc.metadata['page_no'], "text": doc.page_content} for doc in docs]
        
        return content, metadata
    
    def generate_web_context(self, retrieval_query: str, rerank=True) -> tuple[list[str], list[str]]:
        return [], []
        
        # if (rerank):
        #     docs = self.web_retriever.invoke(retrieval_query, k=3)
        #     docs = self.db.rerank_docs(retrieval_query, docs, top_k=5)
        # else:    
        #     docs = self.web_retriever.invoke(retrieval_query, k=2)

        # content = [doc.page_content for doc in docs]
        # metadata = list(set([doc.metadata['source'] for doc in docs]))

        # return content, metadata
    # --------------------------------------------------------------------- #

    def generate_context(self, retrieval_method: str, retrieval_query: str,
                        doc_ids: list[str] = None,
                        enable_web_retrieval=False) -> tuple[str, dict[str, str]]:
        
        # Generate vectorestore context
        vecdb_context, vecdb_metadata = [], []
        if (retrieval_method in ("Vectorstore", "Hybrid")):
            vecdb_context, vecdb_metadata = self.generate_vecdb_context(retrieval_query, doc_ids)
        
        # Generate web context
        web_context , web_metadata = [], []
        if (retrieval_method in ("Web", "Hybrid")) or (enable_web_retrieval):
            web_context, web_metadata = self.generate_web_context(retrieval_query)

        # Combine the contexts
        context = vecdb_context + web_context
        context = '\n'.join(context)
        metadata = {"doc_sources": vecdb_metadata, "web_sources": web_metadata}

        return context, metadata
    # --------------------------------------------------------------------- #
    
    def generate_answer(self, user_prompt: str, chat: str, context: str) -> AsyncIterator[str]:
        QA_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with answering user question.
            Given the previous chat history, user question, and relevant context, provide a clear, concise, and informative answer.
            Your response should be structured in a way that is easy to understand and directly addresses the user's question.
            Previous Chat History: ```{chat}```
            User Question: ```{user_prompt}```
            Relevant Context: ```{context}``` """
        )
        qa_chain = QA_PROMPT | self.llm | StrOutputParser()
        
        llm_input = {
            "user_prompt": user_prompt,
            "context": context,
            "chat": chat
        }
        return qa_chain.astream(llm_input)
    # --------------------------------------------------------------------- #

