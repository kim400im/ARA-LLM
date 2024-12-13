from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import JSONResponse
# from query_engine import QueryEngineHandler


from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()
app = FastAPI()
# 환경 변수에서 API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=openai_api_key)
# client = OpenAI()

# OpenAI API 키 설정

# 입력 모델 정의
class InputData(BaseModel):
    prompt: str

class Prompt(BaseModel):
    prompt: str

class Query(BaseModel):
    question: str

# 업로드 파일 저장 경로
UPLOAD_DIRECTORY = "./uploaded_files"



# 디렉토리 생성
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.get("/")
async def read_root():
    return {"message": "Hello Python"}

@app.post("/process_prompt")
async def process_prompt(prompt: Prompt):
    # 여기서는 간단히 입력받은 prompt를 그대로 반환합니다.
    # 실제 LLM 처리는 이 부분에 구현하면 됩니다.
    return {"response": f"Processed: {prompt.prompt}"}


# 비교적 일관된 출력을 내는 0.3
Settings.llm = OpenAI(temperature=0.3, model="gpt-4o-mini")

# 저장된 폴더에서 사업보고서를 가져온다.
folder_path = './uploaded_files'


@app.post("/process")
async def process_input(query: Query):
    try:
        documents = SimpleDirectoryReader(folder_path).load_data()

        index = VectorStoreIndex.from_documents(
            documents,
        )

        # 생성된 문서에서 상위 10개의 유사한 문서를 검색한다.
        # 여러 문서를 검색해서 더 넓은 컨택스트를 찾는다.
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )

        # 검색된 문서로부터 응답을 생성하는 객체를 만든다.
        response_synthesizer = get_response_synthesizer()

        # 검색기, 응답기, 유사도 검색기를 조합하여 쿼리 엔진을 구성한다.
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )

        response = query_engine.query(query.question)
        print(response)

        if isinstance(response, dict):  # 딕셔너리라면
            return {"response": response.get("response", "No valid response found")}
        elif hasattr(response, "response"):  # 객체에 response 속성이 있다면
            return {"response": response.response}
        else:
            return {"response": str(response)}  # 문자열로 변환하여 반환
        #return {"response": response}



        # query_list = ["현재 삼성전자의 최대 주주는 누구인가요?", "삼성전자의 DX부문 매출액은 얼마인가요?"]
    #     print("hello")
    #     # Check if there are any PDF files
    #     if not os.listdir(UPLOAD_DIRECTORY):
    #         return {"error": "No PDF files found. Please upload files first."}
        
    #     print("hi")
    #     query_engine_handler = QueryEngineHandler(UPLOAD_DIRECTORY)
    #     print("gg")
    #     extracted_info = query_engine_handler.process_query(query.question)

    #     print("gg")

    #     # GPT 요청 생성
    #     gpt_prompt = f"Extracted Info:\n{extracted_info}\nQuestion:\n{query.question}\nAnswer:"
    #     print(f"GPT Prompt: {gpt_prompt}")  # 디버깅: GPT 프롬프트 확인

        
    #     # Process the user's question
    #     # extracted_info = query_engine_handler.process_query(query.question)

    #     # response = query_engine_handler.process_query(query.question)

    #     # Return the generated response
    #     # return {"response": response}
        
    #     # If no relevant info was found, return a friendly message
    #     if not extracted_info or extracted_info == "No relevant information found in the provided documents.":
    #         return {"response": "I'm sorry, I couldn't find relevant information in the uploaded documents."}

    #     # Return the GPT-4 response based on the extracted info
    #     # gpt_response = query_engine_handler.generate_gpt_response(query.question, extracted_info)
        
    #     gpt_response = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": gpt_prompt}
    #         ]
    #     )
    #     print(f"GPT Response: {gpt_response}")  # 디버깅: GPT 응답 확인
    #     # return {"response": gpt_response}
    #     return {"response": gpt_response['choices'][0]['message']['content'].strip()}

    except Exception as e:
        return {"error": f"Error processing the question: {str(e)}"}
    

# @app.post("/process")
# async def process_input(data: InputData):
#     try:
#         # ChatGPT API 호출
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # 또는 사용하고자 하는 모델
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": data.prompt}
#             ]
#         )
#         # 응답 반환
#         return {"response": response.choices[0].message.content.strip()}
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {"error": "An error occurred while processing your request."}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 파일 저장 경로 지정
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        
        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        return JSONResponse({"message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)