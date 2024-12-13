# pip install llama-index-retrievers-bm25
# pip install chromadb
# pip install torch transformers python-pptx Pillow
# pip install llama-index-llms-openai

# import nltk
# print(nltk.__version__)
# nltk.download('wordnet')

# pip install llama-index-embeddings-openai
# pip install llama-index-readers-file     
import openai
from dotenv import load_dotenv
import os
from llama_index.llms.openai import OpenAI

# os.environ['OPENAI_API_KEY'] = 

# response = OpenAI().complete('영훈초등학교는 어떤 학교인가요?')
# print(response)


# 삼성전자 사업보고서를 pdf형태로 다운받고 해당 문서 기반 답변하기

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# # 비교적 일관된 출력을 내는 0.3
# Settings.llm = OpenAI(temperature=0.3, model="gpt-4o-mini")

# # 저장된 폴더에서 사업보고서를 가져온다.
# folder_path = './data/samsumg/사업보고서'
# documents = SimpleDirectoryReader(folder_path).load_data()

# # 로드한 문서로부터 벡터인덱스를 생성한다. 대규모 데이터셋에서도 성능을 좋게 보임.벡터간 유사도를 생성하여 대규모 데이터셋에서 높은 성능을 보인다.
# index = VectorStoreIndex.from_documents(
#     documents,
# )

# # 생성된 문서에서 상위 10개의 유사한 문서를 검색한다.
# # 여러 문서를 검색해서 더 넓은 컨택스트를 찾는다.
# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=10,
# )

# # 검색된 문서로부터 응답을 생성하는 객체를 만든다.
# response_synthesizer = get_response_synthesizer()

# # 검색기, 응답기, 유사도 검색기를 조합하여 쿼리 엔진을 구성한다.
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
# )

# response = query_engine.query('삼성전자의 사업보고서를 요약해줘')
# print(response)



# query_list = ["현재 삼성전자의 최대 주주는 누구인가요?", "삼성전자의 DX부문 매출액은 얼마인가요?"]

# for q in query_list:
#   print(f"Q: {q}")
#   print(f"A: {query_engine.query(q)}")
#   print("\n=========================\n")




# class QueryEngineHandler:
#     def __init__(self):
#         # Load environment variables
#         load_dotenv()

#         # Set OpenAI API key
#         os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#         if not os.getenv("OPENAI_API_KEY"):
#             raise ValueError("OpenAI API Key is not set. Check your .env file.")
        


#         # Initialize LLM settings
#         Settings.llm = OpenAI(temperature=0.3, model="gpt-4o-mini")

#         self.PDF_FOLDER_PATH = os.path.abspath("./uploaded_files")
#         os.makedirs(self.PDF_FOLDER_PATH, exist_ok=True)

#         if not os.listdir(self.PDF_FOLDER_PATH):
#             raise ValueError("No PDF files found in the uploaded_files directory. Please upload files first.")

#         # Load documents from the folder
#         documents = SimpleDirectoryReader(self.PDF_FOLDER_PATH).load_data()

#         # Create a vector index from the documents
#         self.index = VectorStoreIndex.from_documents(documents)

#         # Set up the retriever
#         self.retriever = VectorIndexRetriever(
#             index=self.index,
#             similarity_top_k=10,  # Retrieve top 10 similar documents
#         )

#         # Set up the response synthesizer
#         self.response_synthesizer = get_response_synthesizer()

#         # Create the query engine
#         self.query_engine = RetrieverQueryEngine(
#             retriever=self.retriever,
#             response_synthesizer=self.response_synthesizer,
#             node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
#         )

#     def process_query(self, question: str) -> str:
#         response = self.query_engine.query(question)
#         if not response or not response.response:
#             return "No relevant information found in the provided documents."
#         return response.response



class QueryEngineHandler:
    def __init__(self, pdf_folder_path: str):
        load_dotenv()

        # Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Initialize LLM settings
        Settings.llm = OpenAI(temperature=0.3, model="gpt-4")

        # Define folder path for PDF documents
        self.PDF_FOLDER_PATH = os.path.abspath(pdf_folder_path)

        # Ensure directory exists
        os.makedirs(self.PDF_FOLDER_PATH, exist_ok=True)

        # Check if folder contains files
        if not os.listdir(self.PDF_FOLDER_PATH):
            raise ValueError(f"No PDF files found in {self.PDF_FOLDER_PATH}. Please upload files.")

        # Load documents and create vector index
        documents = SimpleDirectoryReader(self.PDF_FOLDER_PATH).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

        # Set up retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10
        )

        # Set up response synthesizer
        self.response_synthesizer = Settings.llm.get_response_synthesizer()

        # Create the query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )

    def process_query(self, question: str) -> str:
        # Retrieve information based on the user's question
        response = self.query_engine.query(question)
        if not response or not response.response:
            return "No relevant information found in the provided documents."
        return response.response

    def generate_gpt_response(self, question: str, extracted_info: str) -> str:
        # Combine the extracted information and user's question
        gpt_prompt = (
            "Based on the following extracted information, answer the user's question:\n\n"
            f"Extracted Information:\n{extracted_info}\n\n"
            f"User's Question:\n{question}\n\n"
            "Answer:"
        )

        # Call OpenAI GPT API
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": gpt_prompt}
                ]
            )
            return gpt_response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating GPT response: {str(e)}"