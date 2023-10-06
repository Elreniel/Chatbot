import os
import torch
import transformers
import streamlit as st
import time
import logging
logging.basicConfig(format='%(process)d - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList, MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from typing import List

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_list = ("tiiuae/falcon-40b-instruct", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf")
model_eng2tr_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
model_tr2eng_name = "Helsinki-NLP/opus-mt-tc-big-tr-en"
embedding_name = "sentence-transformers/all-mpnet-base-v2"
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

st.set_page_config(layout="wide",
                   page_title="Akbank Asistan",
                   # page_icon="./images/akbank_logo.png"
                   )

hide_streamlit_style =  """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache_resource
def get_llama2_chain(model_id):

    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type='nf4',
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_compute_dtype=bfloat16
                                                 )

    model_config = transformers.AutoConfig.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                trust_remote_code=True,
                                                config=model_config,
                                                # quantization_config=bnb_config,
                                                device_map='auto'
                                                )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    stop_list = ['Human:',
                 'Human: ',
                 ' Human:',
                 ' Human: ',
                 '\nHuman: ',
                 '\nHuman:'
                 ]

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    generate_text = pipeline(model=model,
                            tokenizer=tokenizer,
                            return_full_text=True,
                            task='text-generation',
                            stopping_criteria=stopping_criteria,
                            temperature=0.1,
                            max_new_tokens=512,
                            repetition_penalty=1.1
                            )


    llm = HuggingFacePipeline(pipeline=generate_text)

    vectorstore = get_embeddings(embedding_name)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant. Answer the question based on the context below. The context is:\n{context}")

    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "{question}"
    )

    chain = ConversationalRetrievalChain.from_llm(llm,
                                                  vectorstore.as_retriever(),
                                                  return_source_documents=True,
                                                  combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_messages(
                                                      [system_message_prompt, human_message_prompt])})

    return chain


@st.cache_resource
def get_llm_chain(model_id):

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True,
                                                 load_in_8bit=True,
                                                 device_map="auto",
                                                 token=HUGGINGFACE_TOKEN)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    generation_config = model.generation_config
    generation_config.temperature = 0.1
    generation_config.max_new_tokens = 512
    generation_config.repetition_penalty = 1.1
    generation_config.num_return_sequences = 1
    generation_config.use_cache = False


    if model_id == "tiiuae/falcon-40b-instruct":
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

    class StopGenerationCriteria(StoppingCriteria):
        def __init__(
                self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
        ):
            stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
            self.stop_token_ids = [
                torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
            ]

        def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_ids in self.stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False


    stop_tokens = [["Human", ":"], ["AI", ":"]]
    stopping_criteria = StoppingCriteriaList([StopGenerationCriteria(stop_tokens, tokenizer, model.device)])

    generation_pipeline = pipeline(model=model,
                                   tokenizer=tokenizer,
                                   return_full_text=True,
                                   task="text-generation",
                                   stopping_criteria=stopping_criteria,
                                   generation_config=generation_config,
                                   )

    llm = HuggingFacePipeline(pipeline=generation_pipeline)

    vectorstore = get_embeddings(embedding_name)

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever()
                                        )

    return chain


@st.cache_resource
def get_translation_tokenizer_model(model_id):
    tokenizer = MarianTokenizer.from_pretrained(model_id,
                                                max_length=4096)
    model = MarianMTModel.from_pretrained(model_id)

    return tokenizer, model


def get_embeddings(embedding_name):
    documents = []
    for file in os.listdir("documents"):
        if file.endswith(".pdf"):
            pdf_path = "./documents/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./documents/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./documents/" + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_name, model_kwargs={"device": device})

    vectorstore = FAISS.from_documents(all_splits, embeddings)

    return vectorstore



def create_session_list(listName):
    if listName not in st.session_state:
        st.session_state[listName] = []


def create_session_variable(variableName):
    if variableName not in st.session_state:
        st.session_state[variableName] = ""


sessionList = ["messages", "chat_history"]
sessionVariables = []

for tempVariable in sessionList:
    create_session_list(tempVariable)

for tempVariable in sessionVariables:
    create_session_variable(tempVariable)

with st.columns([0.3, 0.8])[1]:
    st.title("Welcome to my Virtual AI Asistant!")

model_id = st.selectbox('Please choose an LLM Model', model_list)
chain = get_llm_chain(model_id)

tokenizer_eng2tr, model_eng2tr = get_translation_tokenizer_model(model_eng2tr_name)
tokenizer_tr2eng, model_tr2eng = get_translation_tokenizer_model(model_tr2eng_name)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Sana nasıl yardımcı olabilirim?"):
    logging.info(f"Query: {query}")
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    query_eng = tokenizer_tr2eng.decode(model_tr2eng.generate(**tokenizer_tr2eng(query,
                                                                                 return_tensors="pt",
                                                                                 padding=True
                                                                                 )
                                                              )[0],
                                        skip_special_tokens=True
                                        )
    logging.info(f"Query Eng: {query_eng}")

    with st.chat_message("assistant"):
        with st.spinner("Lütfen bekleyiniz..."):
            message_placeholder = st.empty()
            full_response = ""

            startTime = time.time()
            assistant_response_eng = chain.run(query_eng)
            endTime = time.time()
            logging.info(f"Response Eng: {assistant_response_eng}")

            result = assistant_response_eng.split("Human:")[0]

            assistant_response = tokenizer_eng2tr.decode(model_eng2tr.generate(**tokenizer_eng2tr(result,
                                                                                                  return_tensors="pt",
                                                                                                  padding=True
                                                                                                  )
                                                                               )[0],
                                                         skip_special_tokens=True
                                                         )
            logging.info(f"Response: {assistant_response}")

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(f"{full_response} Cevap süresi: {str(int(endTime - startTime))} saniye")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
