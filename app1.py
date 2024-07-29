import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import os
load_dotenv()


# Load the GROQ and Google API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Kidney Care AI")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template("""
Based on the provided patient data, generate a summary and recommendations for managing Chronic Kidney Disease (CKD).
<context>
{context}
<context>
Patient Data: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./ckd")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

if st.button("Prepare Document Embeddings"):
    vector_embedding()
    st.write("Document Embeddings Prepared")

# Input fields for patient data
age = st.number_input("Age", min_value=0)
weight = st.number_input("Weight (kg)", min_value=0.0)
#medical_conditions = st.text_area("Medical Conditions")
heart_rate = st.number_input("Heart Rate", min_value=0)
spo2 = st.number_input("SpO2 Level (%)", min_value=0.0, max_value=100.0, step=0.1)
blood_pressure_systolic = st.number_input("Blood Pressure (Systolic)", min_value=0)
ankle_swelling = st.selectbox("Ankle Swelling", ["mild", "moderate","severe"])
breathlessness = st.selectbox("Breathlessness", ["mild", "moderate","severe"])
#current_medications = st.text_area("Current Medications")
#average_diff = st.number_input("Average Difference in Key Metrics", min_value=-100.0, max_value=100.0, step=0.1)
#progress = st.selectbox("Progress", ["0", "1", "2", "3"])
#progress_category = st.selectbox("Progress Category", ["Improving", "Worsening", "Serious"])
predicted_progress = st.selectbox("Predicted Progress", ["Improving", "Worsening", "Serious"])

# Create a dictionary with the patient data
patient_data = {
    "Age": age,
    "Weight": weight,
    #"Medical Conditions": medical_conditions,
    "Heart Rate": heart_rate,
    "SpO2 Level": spo2,
    "Blood Pressure (Systolic)": blood_pressure_systolic,
    "Ankle Swelling": ankle_swelling,
    "Breathlessness": breathlessness,
    #"Current Medications": current_medications,
    #"Average Difference in Key Metrics": average_diff,
    #"Progress": progress,
    #"Progress Category": progress_category,
    "Predicted Progress": predicted_progress
}

def generate_recommendations(predicted_progress):
    if predicted_progress == "Serious":
        return "Your condition requires immediate medical attention. Please consult your doctor promptly."
    elif predicted_progress == "Worsening":
        return ("If your condition is worsening, consider increasing your frusemide dose. "
                "Monitor your condition more frequently and watch out for any new symptoms. "
                "Additionally, reduce your protein intake and fluid intake by 300 ml to avoid volume overload.")
    elif predicted_progress == "Improving":
        return "If your condition is improving, continue taking your medicines on time and monitor your condition regularly."
    else:
        return "Invalid progress prediction"

if st.button("Generate Recommendations") and "vectors" in st.session_state:
    # Prepare the input for the LLM
    patient_data_input = "\n".join([f"{key}: {value}" for key, value in patient_data.items()])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': patient_data_input})
    st.write(f"Response time: {time.process_time() - start} seconds")
    st.write(response['answer'])

    # Display the recommendations based on predicted progress
    recommendations = generate_recommendations(predicted_progress)
    st.write(f"Recommendations: {recommendations}")

    # With a Streamlit expander
    with st.expander("🧬🧬🧬🧬🧬🧬🧬🧬"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("💊💊💊💊💊💊💊💊💊💊💊💊💊💊")
            
