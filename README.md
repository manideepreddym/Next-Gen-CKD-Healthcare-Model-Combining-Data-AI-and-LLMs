# Kidney Care AI
### '''Next-Gen-CKD-Healthcare-Model-Combining-Data-AI-and-LLMs'''
## Introduction

Chronic Kidney Disease (CKD) is a progressive condition characterized by the gradual loss of kidney function over time. Effective management of CKD requires continuous monitoring and personalized treatment recommendations. Our goal with the Kidney Care AI tool is to streamline CKD management by providing tailored recommendations and actionable insights based on patient-specific data, thereby improving quality of life and health outcomes.

## What It Does

The Kidney Care AI system:
- Collects patient data and clinical notes.
- Processes and analyzes this information using machine learning models.
- Provides personalized recommendations for managing CKD.
- Integrates the RAG (Red, Amber, Green) model to categorize patient progress and suggest appropriate actions based on disease severity.

## How We Built It

1. **Data Collection and Ingestion:**
   - Load patient data and clinical notes from PDF documents using `PyPDFDirectoryLoader`.

2. **Data Preprocessing:**
   - Split documents into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Embedding Generation and Storage:**
   - Convert text data into vectors for similarity search using `GoogleGenerativeAIEmbeddings`.
   - Store embeddings for efficient retrieval using `FAISS`.

4. **Model Training and Predictions:**
   - Generate patient summaries and recommendations using `ChatGroq` and `ChatPromptTemplate`.

5. **Recommendation Logic:**
   - Implement logic to generate recommendations based on predicted patient progress.

6. **RAG Model Integration:**
   - Combine retriever and generator for relevant document context and accurate recommendations using `create_retrieval_chain`.

7. **Deployment and User Interaction:**
   - Create an interactive web application using `Streamlit`.

## Challenges We Ran Into

- Ensuring the accuracy of recommendations based on limited patient data.
- Integrating various components (data ingestion, preprocessing, embedding generation, and model training) seamlessly.
- Handling and processing large volumes of text data efficiently.

## Accomplishments We're Proud Of

- Successfully integrating multiple AI technologies to create a comprehensive system for CKD management.
- Developing a user-friendly interface for patients and healthcare providers to interact with the system.
- Achieving meaningful and actionable recommendations that can positively impact patient care.

## What We Learned

- The importance of high-quality data for training machine learning models.
- Techniques for efficiently processing and embedding large text datasets.
- Best practices for integrating AI models into interactive applications.

## Key Points About CKD

### Risk Factors
- **Diabetes:** High blood sugar levels can damage the blood vessels in the kidneys.
- **High Blood Pressure:** Can damage the small blood vessels in the kidneys.
- **Other Risk Factors:** Heart disease, obesity, family history of kidney failure, and age over 60.

### Symptoms
- Swelling in the ankles and feet
- Fatigue
- Difficulty concentrating
- Decreased appetite
- Changes in urine output

### Diagnosis
- **Blood Tests:** Measure GFR to assess kidney function.
- **Urine Tests:** Look for protein or blood in the urine.
- **Imaging Tests:** Assess kidney structure.

### Treatment
- **Managing Underlying Conditions:** Such as diabetes and hypertension.
- **Lifestyle Changes:** Diet, exercise, and avoiding nephrotoxic medications.

### Recommendations for Managing CKD
- **Serious Condition (End-Stage Renal Disease - ESRD):** Immediate actions and lifestyle modifications.
- **Worsening Condition (Stage 3-4 CKD):** Medical management, lifestyle modifications, and monitoring.
- **Stable Condition (Stage 1-2 CKD):** Medical management, lifestyle modifications, and monitoring.

## Patient Data Analysis and Recommendations

### Patient Data Required
- Age
- Weight
- Medical conditions
- Heart rate
- SpO2 levels
- Blood pressure (systolic)
- Ankle swelling
- Breathlessness
- Current medications
- Average difference in key metrics
- Progress
- Progress category
- Predicted progress

## Built With
- **Gemini**
- **Grok**
- **LangChain**
- **Python**
- **RAG**

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

We welcome contributions from the community. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For any questions or feedback, please contact us at [manideepreddy966@gmail.com](mailto:manideepreddy966@gmail.com).


## Running the Application

To run the `app.py` file and start the Streamlit application, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/kidney-care-ai.git
   cd kidney-care-ai
## Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

## Install Required Dependencies:

pip install -r requirements.txt

## Run the Application:

streamlit run app.py


