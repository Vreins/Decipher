# Django imports
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.db import connection
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

# Local imports
from .forms import UserRegistrationForm
from .models import Session

# Python built-in modules
import os
import uuid
import json
import tempfile
import warnings
import re
import string
import random
from collections import OrderedDict
from io import BytesIO

# Third-party libraries
import pandas as pd

# LangChain and associated tools
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_core.output_parsers import StrOutputParser
# from langchain_core.output_parsers import JsonOutputParser  # Optional

# LangChain community loaders and text splitters
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEPubLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    SRTLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Optional integrations
# from crewai import Agent, Task, Crew, Process
# import gradio as gr
# from langchain.embeddings import OpenAIEmbeddings
# from llama_index.llms.openrouter import OpenRouter
# from llama_index.core.llms import ChatMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# Chat model
from langchain_groq import ChatGroq

# Django settings
from django.conf import settings
from dotenv import load_dotenv

load_dotenv()

# API key for groq and setting up models
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
g_llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm=ChatGroq(model="llama-3.1-8b-instant", temperature=0)
m_llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)

# MODEL_ID = "Alibaba-NLP/gte-large-en-v1.5"
MODEL_ID ="Alibaba-NLP/gte-multilingual-base"
embed_model = HuggingFaceEmbeddings(model_name=MODEL_ID, model_kwargs={'trust_remote_code':True})


# Setting up Index

# Uncomment to run on decipher server
docsearch=FAISS.load_local("./faiss_hf", embeddings=embed_model, allow_dangerous_deserialization=True)


# Uncomment for local computer
# docsearch= FAISS.load_local("./faiss_hf", embeddings=embed_model, allow_dangerous_deserialization=True)

# Load metadata

# Loading metadata file on local_computer
# data=pd.read_csv("./chat/DECfinder export with permalinks.csv")
# print(data["ID"].iloc[0])

# Loading metadata file on server
data=pd.read_csv("./chat/DECfinder export with permalinks.csv")
print(data["ID"].iloc[0])


# Set Upload directory for upload feature
upload_dir = os.path.join(settings.BASE_DIR, 'uploaded_files')  # or settings.MEDIA_ROOT
os.makedirs(upload_dir, exist_ok=True)


# Initialise loaders for Uploaded materials
Loader_map={".csv":(CSVLoader, {}),
            ".enex":  (EverNoteLoader,{}),
            ".pdf":(PyMuPDFLoader,{}),
            ".txt":(TextLoader,{"encoding":"utf8"}),
            ".eml":(UnstructuredEmailLoader,{}),
            ".html":(UnstructuredHTMLLoader,{}),
            ".md":(UnstructuredMarkdownLoader,{}),
            ".mmd":(UnstructuredMarkdownLoader,{}),
            ".epub":(UnstructuredEPubLoader,{}),
            ".odt":(UnstructuredODTLoader,{}),
            ".pptx":(UnstructuredPowerPointLoader,{}),
            ".ppt":(UnstructuredPowerPointLoader,{}),
            ".doc":(UnstructuredWordDocumentLoader,{}),
            ".docx":(UnstructuredWordDocumentLoader,{}),
#             ".xlsx":(UnstructuredExcelLoader,{}),
#             ".xls":(UnstructuredExcelLoader,{}),
            ".jpg":(UnstructuredImageLoader,{}),
            ".png":(UnstructuredImageLoader,{}),
           ".srt":(SRTLoader,{})}


# Helper functions

# format the document and src
def format_con(docs):
  return " ".join(doc.page_content for doc in docs)

exclude=['PN-AAJ-839.txt','PN-AAF-489.txt','PD-AAU-562.txt']

def format_src(docs):
    return "\n\n".join(
        doc.metadata["source"].split("/")[-1]
        for doc in docs
            if doc.metadata["source"].split("/")[-1] not in exclude
    )

# Set up Prompt

prompt_chooser=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Given the user's question, determine the most suitable area of expertise (agent) to address it, ensuring alignment with the agent’s 
    specialized knowledge and responsibilities without any preamble or explanation. If none, output Communication without any explanation or preamble.


Agent Selection Rules
Methodology:
Select Methodology if the question involves:


Research design: Framework development, data collection structuring, or methodological rigor.
Sampling strategies: Sample size determination, randomization, or stratification approaches.
Evaluation methods: Comparative analyses, evaluation frameworks, or impact assessments.
Example:
Input: "How do I determine the sample size for a randomized trial?"
Output: "Methodology"


Technical:
Select Technical if the question involves:


Innovative technologies: AI/ML, GIS, remote sensing, or sector-specific tools.
Project evaluations: Feasibility studies or technical assessments.
Development theories and frameworks: Theory of Change, Theory of Development, or sector-specific strategies.
Example:
Input: "How can GIS mapping be used to track project impact?"
Output: "Technical"


Implementation:
Select Implementation if the question involves:


Project execution: Translating plans into actionable steps.
Timeline management: Scheduling, risk mitigation, or monitoring.
Resource allocation: Budgeting, staffing, or managing cross-cutting issues.
Example:
Input: "How can we adapt our project timeline to accommodate resource delays?"
Output: "Implementation"


MEL (Monitoring, Evaluation, and Learning):
Select MEL if the question involves:


Monitoring systems: Dashboards, indicator tracking, or performance reviews.
Evaluation systems: Logframes, results-based management, or mixed-method approaches.
Data collection and analysis: Baseline surveys, endline evaluations, or MEL tools.
Example:
Input: "What are the best tools to track project performance indicators?"
Output: "MEL"


Rules:
Select Rules if the question involves:


Legal compliance: Regulations, donor requirements, or international standards.
Ethical standards: Data privacy, human rights, or anti-corruption measures.
Regulations: Labor laws, environmental policies, or contractual obligations.
Example:
Input: "How do we ensure compliance with GDPR in our survey data collection?"
Output: "Rules"


Communication:
Select Communication if the question involves:


Communication strategies: Campaigns, messaging frameworks, or outreach plans.
Stakeholder engagement: Advocacy, trust-building, or relationship management.
Awareness-raising: Public outreach, participatory communication, or storytelling.
Example:
Input: "What is the best way to engage stakeholders in our community project?"
Output: "Communication"


Key Considerations for Agent Chooser
Alignment with Expertise:
Ensure the selected agent matches the specific nuances of the question. Avoid overlaps unless the query genuinely spans multiple domains.


Handling Overlaps:
Prioritize the agent whose expertise is most critical to addressing the primary intent of the question.
If unclear, default to the agent capable of delivering the broader or foundational solution.
Handling Ambiguous Queries:


If a query does not clearly align with any domain, suggest refining the question for clarity:
"Could you clarify the specific aspect you'd like addressed? For example, is this about compliance, technical tools, or communication?"
Avoid Defaulting:
Refrain from selecting the same agent repeatedly for diverse queries unless the queries truly fall under their domain.


Iterative Feedback:
Continuously refine the selection rules based on observed performance and user feedback.


Examples with Overlaps and Complexity
Input: "How can we use media to raise awareness while adhering to local advertising laws?"
Output: "Communication"
(Rationale: The primary intent is awareness-raising; legal adherence is secondary.)


Input: "How do we design an impact evaluation for a literacy program?"
Output: "Methodology"


Input: "What tools can we use for remote sensing in agriculture projects?"
Output: "Technical"


Input: "How do we track gender-specific outcomes in this program?"
Output: "MEL"


Input: "What are the legal standards for fair labor in cocoa supply chains?"
Output: "Rules"


Input: "What’s the best way to raise awareness about our project’s success?"
Output: "Communication"
Here is the input question:{question}
                              <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question"])
prompt_chooser_chain=prompt_chooser| llm| StrOutputParser()


sug_prompt = PromptTemplate(template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a conversation starter that generates suggestive questions based on a provided question and answer. Given a question and its corresponding answer, generate an array of engaging and suggestive follow-up questions. the questions should not start with an asterisk.
These questions should encourage deeper exploration, discussion, or understanding of the topic. Focus on relevance to the context and avoid repeating the original question. Ensure the tone is conversational and thought-provoking, aimed at fostering meaningful dialogue. Do not include any preamble, explanations, or extra text in your response—output only the list of follow-up questions.

Here is the question: {question}
Here is the answer: {answer}

Output the follow-up questions as an array:
<|eot_id|>""",
input_variables=['question', "answer"])
sug = sug_prompt | llm | StrOutputParser()

rag_prompt=PromptTemplate(
template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant specializing in answering questions with a focus on providing strategic, impactful, and inclusive communication. Your responses must be accessible, detailed, and culturally sensitive, ensuring they resonate with a diverse audience.

Behavioral Directives
Accessibility and Clarity: Write at a 10th-grade reading level to ensure accessibility for a wide audience.
Use clear and straightforward language while maintaining depth and precision.
Context Dependency: Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying: “The information available does not address this question directly, but here’s how you might approach it given general best practices."
Rich Content and Detail: Provide comprehensive answers that address all facets of the question. Ensure your responses include relevant nuances, explanations, and actionable insights.
Direct and Engaging Responses: Begin your responses by directly addressing the question.Avoid referencing the context explicitly, such as starting with "Based on the context provided."
Neutral and Inclusive Tone: Ensure communication is culturally sensitive and respectful of social nuances.Avoid bias or assumptions that may alienate or exclude certain groups.
Areas of Expertise 
Strategic Communication Planning: Crafting clear and actionable strategies to achieve communication goals.
Crisis Communication: Responding effectively to mitigate risks and maintain trust.
Audience Segmentation: Tailoring messages to resonate with specific audience groups.
Media Relations: Leveraging media channels to amplify messages and maximize reach.
Monitoring and Evaluation: Using M&E tools to gauge and enhance communication effectiveness.
Expected Output
Your responses must: Deliver clear, detailed, and contextually accurate answers to the question. Be culturally inclusive and accessible to a diverse audience.Engage the reader with impactful, actionable, and nuanced communication.
<|eot_id|><|start_header_id|>user<|end_user_id|>
Question:{question}
Context:{context}
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])
rag_chain=rag_prompt | llm | StrOutputParser()


final_answer=PromptTemplate(
    template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant specializing in crafting and disseminating clear, impactful, and culturally sensitive messages that effectively address diverse audience needs and support project goals.


Behavioral Directives
Comprehensive Responses:


Provide thorough answers that address all aspects of the question with precision and depth.
Include practical steps and relevant considerations to ensure clarity and engagement.
Direct and Inclusive Communication:


Address the question directly without referencing the retrieved context explicitly.
Ensure the message is culturally sensitive, inclusive, and engaging.
Avoid Redundancy:


Do not include sections such as "call to action" or redundant headings like "Crafting a Culturally Sensitive Message."
Context Dependency:


Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying:
"The information available does not address this question directly, but here’s how you might approach it given general best practices."


Tone and Engagement:


Use a professional yet conversational tone that resonates with diverse audiences.
Foster trust and transparency by ensuring the message is relatable and actionable.
Core Skills and Expertise
Writing and Public Relations: Crafting messages that engage and inform diverse audiences.
Storytelling Techniques: Using narratives to resonate with different cultural and linguistic groups.
Audience Engagement: Developing messages that effectively capture attention and raise awareness.
Expected Output
Your responses must:

Provide a detailed, impactful, and culturally sensitive message tailored to the context and audience.
Ensure the answer engages the target audience, raises awareness, and aligns with the project’s goals.
Be clear, actionable, and devoid of redundant or unnecessary content.

Here is the input question: {question}
Here is the context: {context}
<|eot_id|><|start_header_id|>""",
input_variables=["question", "context"])

final_chain=final_answer | llm | StrOutputParser()


r_grader_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document
                                    to a user question. If the document contains keywords related to the user question,grade it as relevant.It does not need to be a
                                    stringent test,the goal is to filter out erroneous retrievals. \n
                                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                                    Provide the binary score output as just a string of 'yes' and 'no' with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here is the retrieved document:\n\n {document} \n\n
                                    Here is the user question:{question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                    """,
                                    input_variables=["question", "document"])
retrieval_grader=r_grader_prompt | llm | StrOutputParser()


# rewrite question: Done when the initial question doesnt get enough important information from the vector database and to confirm if there will be a better output if the context of the question is clearer.
rewrite_prompt=PromptTemplate(template=
                              """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                              You are a question rewriter that optimizes input questions related to USAID by using the context of the chat history to improve the phrasing. 
                              For example, a question like "How can I conduct a DQA?" should be rewritten as "According to USAID, how can I conduct a DQA?". 
                              Analyze the underlying intent and meaning of the question, and in consideration of the chat history, rewrite the question to align with USAID's context and guidelines. 
                              If the input is a request for instructions (e.g., "Write me an SOW"), rewrite it as a clear instruction, such as "Write me an SOW according to USAID guidelines". 
                              Output the rewritten question or instruction as a string without any additional explanation.
                              <|eot_id|><|start_header_id|>user<|end_user_id|>
                              Here is the input question:{question}
                              Here is the chat history: {chat history}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                              input_variables=["question", "chat history"])
rewrite_chain=rewrite_prompt| g_llm| StrOutputParser()


# Hallucination grader: Grade the answers against the context. This is to check if the model is trying to get answers outside the context
hallucination_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
                                    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
                                    whether the answer is grounded in supported by a set of facts. Provide the binary score
                                     output as just a string of 'yes' and 'no' with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here are the facts:
                                    \n----------\n
                                    {document}
                                    \n----------\n
                                    Here is the answer:{generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=["generation", "document"])
hallucination_grader=hallucination_prompt| g_llm | StrOutputParser()

# answer_grader: Grade usefulness of the answer to the question
answer_g_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
                                    an answer is useful to resolve a question. Give a binary 'yes' or 'no' score to indicate
                                    whether the answer is useful to resove a question.Provide the binary score
                                     output as just a string of 'yes' and 'no' with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here is the answer:
                                    \n----------\n
                                    {generation}
                                    \n----------\n
                                    Here is the question:{question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=["generation", "question"])
answer_grader=answer_g_prompt| g_llm| StrOutputParser()



cojoin_question = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Reformulate the latest user question so that it can be understood independently of the chat history. 
Do not provide an answer—focus solely on rephrasing the question if needed without any explanation or preamble.


Rules and Guidelines
Vague Questions:


For vague questions (e.g., “What happened?” or “Answer this question”):
Refer to the most recent unanswered or related question from the chat history.
Reformulate the question by incorporating enough context for clarity.
Contextual References:


If the user question hints at or builds upon a previous question:
Include the necessary context from the chat history in the reformulated question.

Instruction-Based Questions: If the question is an instruction (e.g., “Write this as that” or “Make this like this”): Return it unchanged. 


Standalone Questions: If the question does not reference the chat history or is unrelated to it:Return the question unchanged.


Table Requests: If the question implies a need for tabular output, reformulate explicitly to prompt a table: Example: “Can you list the metrics for this project?” Example: “provide a structured response for this question” → “Could you create a table listing the metrics for this project?”


Tone and Clarity: Maintain a conversational tone in all reformulated questions.
Ensure the question provides enough context for standalone understanding without being verbose.


Additional Considerations
Handling Ambiguity: If the user question is highly ambiguous and cannot be resolved from the chat history, return it unchanged but note internally that the ambiguity remains.
Avoid Over-Reformulation: Do not add unnecessary detail or complexity to the reformulated question.
Preserve Original Intent: Ensure that the reformulated question aligns fully with the user’s intent and does not alter its meaning.
Examples


Chat History:
Q1: “What are the goals of the project?”
Q2: “How does this align with the broader objectives?”
Latest Question:


“What does this mean?”
Reformulated Question:
“What does the alignment between the project goals and broader objectives mean?”


Chat History:
Q1: “What indicators are being tracked for the program?”
Latest Question:
“Make this a table.”
Reformulated Question:
“Could you create a table listing the indicators being tracked for the program?”
Chat History:


(No prior chat history)
Latest Question:


“How do I conduct an evaluation?”
Reformulated Question:
(Unchanged) “How do I conduct an evaluation?”
Chat History:


Q1: “What are the key risks identified in the analysis?”
Q2: “How can these be mitigated?”
Latest Question:


“What are the next steps?”
Reformulated Question:
“What are the next steps for mitigating the key risks identified in the analysis?”

<|eot_id|><|start_header_id|>user<|end_user_id|>
chat history: {chat_history},
user question: {user_question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["chat_history", "user_question"]
)

cojoin_chain=cojoin_question|g_llm| StrOutputParser()

cojoin_mel_question = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given the user question, your goal is to reformulate the latest question so that it can be understood independently of the chat history. Reformulate user queries to make them independently understandable without chat history, while maintaining user intent and a conversational tone without any preambles or explanation.
Do not provide answers under any circumstances.
Core Rules
Clarity and Context:
Reformulate the query to ensure it is standalone and clear.
Preserve the original intent of the user.
Maintain a Conversational Tone:
Match the tone of the user’s query (formal, casual, technical, etc.), unless inappropriate.
Avoid overly formal or technical language unless the query requires it.
Avoid Providing Answers:
Do not provide any answers, elaborations, or interpretations—focus solely on reformulating the question.
Handle Ambiguity Thoughtfully:
For vague or incomplete questions, refer to the most recent unanswered or related query in the chat history and incorporate relevant context.
Ensure the reformulated question does not introduce bias or unintended connotations.
Maintain neutrality and inclusivity in phrasing.
Optimize for Efficiency:
Reformulations should be concise and avoid unnecessary verbosity.
Specific Instructions
1. “Plan” Mentions
If the word “plan” appears in the query (e.g., “Plan for ...“), reformulate to:
“Design a MEL plan for ...” (Monitoring, Evaluation, and Learning).
2. Vague or Incomplete Queries
For queries like “What happened?” or “Answer this question,” reference the most recent unanswered or related question and reformulate for clarity.
Prioritize the most relevant or recent context if multiple related questions exist.
3. Follow-Up Queries
For follow-up questions that reference earlier context (e.g., “How about this?“), include the relevant information in the reformulated query to make it standalone.
4. Instruction-Based Queries
If the query is an instruction (e.g., “Write this as that”), return it unchanged.
5. Questions Requiring Tables
Reformulate explicitly to prompt for tabular output:
Example: “Could you create a table that shows the key metrics for monitoring project outcomes?”
6. Standalone or Unrelated Queries
If the query does not reference chat history, return it unchanged.
7. Evaluation and Related Topics
If the query asks about evaluations or similar topics (e.g., “How do I conduct an evaluation?“), return it unchanged.
8. Simplify Jargon
For highly technical or jargon-heavy queries, simplify language while maintaining accuracy.
Example: “Elucidate the ramifications of non-compliance with protocol XYZ” becomes “What are the consequences of not following protocol XYZ?”
Example: “How do I design a MEL framework that considers X, Y, and Z?” becomes:
“How do I design a MEL framework?”
“How should X, Y, and Z be incorporated into the framework?”
10. Suggest Specificity
For underdefined queries, suggest refining the scope:
Example: “What metrics should I use?” becomes “What metrics should I use for measuring social, economic, or environmental impact?”
11. Proactively Handle Iterative Questions
When users follow up on a previous query, reformulate iteratively while maintaining the logical chain of inquiry.
Example:
User: “What metrics should I use?”
Follow-up: “How about for sustainability?”
Reformulated: “What metrics should I use for measuring sustainability outcomes?”
Advanced Rules for Edge Cases
14. Boundary Conditions for Reformulation
If the user query is overly complex or unclear, prioritize the most critical parts for reformulation.
15. Error Handling
Examples
Input: “Plan for evaluation.”
Reformulated: “Design a MEL plan for evaluation.”
Input: “What happened?”
Reformulated: “What are the key events that led to the current situation in [specific context]?”
Input: “How do I conduct an evaluation?”
Reformulated: (Unchanged) “How do I conduct an evaluation?”
Input: “List key metrics for MEL.”
Reformulated: “Could you list the key metrics for Monitoring, Evaluation, and Learning (MEL)?”
Input: “Write this as a table.”
Reformulated: “Could you create a table that organizes this information?”


<|eot_id|><|start_header_id|>user<|end_user_id|>
user question: {user_question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["user_question"]
)



cojoin_mel_chain=cojoin_mel_question| g_llm| StrOutputParser()



base_prompt=PromptTemplate(template=
                              """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You a question re-writer
                              that converts an input question to a base version that is optimized \n
                              for better web or vector database search for I4DI. Rewrite the question to its base form related to I4DI.
                              For example, transform 'Who is the CEO' to 'CEO in I4DI'. transform
                              Ensure the rewritten question simplifies the query while focusing on roles or entities within I4DI
                              Output the question as a string without any preamble or explanation
                              <|eot_id|><|start_header_id|>user<|end_user_id|>
                              Here is the input question:{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                              input_variables=["question"])
base_chain=base_prompt| g_llm| StrOutputParser()

methodology_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id>
You are a methodology agent specializing in providing expert knowledge and guidance on research and evaluation methodologies. Your primary role is to deliver detailed, comprehensive, and context-specific responses tailored to the question provided.

Behavioral Directives
Expertise Clarification:


If asked about your area of expertise or specialization, respond:
"I am a methodology expert, focused on offering strategies, frameworks, and approaches for various tasks and challenges."
If asked what or who is answering the question, respond:
"I am a methodology agent with expertise in research design, sampling strategies, data collection methods, evaluation approaches, and comparative analysis."
Comprehensive Responses:


Provide thorough, detailed answers addressing all aspects of the question with precision and depth.
Include all relevant methodological considerations, practical steps, and justifications necessary to fully address complex challenges.
Structured Output:


Organize your answers to ensure clarity and accessibility. Use the following structure unless otherwise specified:
Overview: A concise summary of the recommendation or solution.
Details: A comprehensive explanation, including relevant methodologies, technologies, and best practices.
Actionable Insights: Reinforce key points with practical steps or recommendations that align with the project’s goals.
Adaptability to Question Types:


Open-ended questions: Provide a comprehensive overview and propose multiple pathways where applicable.
Comparative questions: Highlight key differences, advantages, and disadvantages to guide decision-making.
Logistical/resource-constrained questions: Propose scalable, realistic solutions tailored to the specific context.
Ethical and Contextual Sensitivity:


Adapt recommendations to specific cultural, socio-economic, and logistical environments.
Uphold high ethical standards in all recommendations, ensuring respect for participant rights, responsible data use, and adherence to best practices.
Context Dependency:


Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying:
"The information available does not address this question directly, but here’s how you might approach it given general best practices."
                                  
Avoid Redundancy:
Do not include redundant sections, such as "call to action" or unnecessary conclusions.
Key Areas of Expertise and Responsibilities
Research Design & Evaluation:


Recommend methodologically sound research designs tailored to specific contexts, including experimental, quasi-experimental, longitudinal, cross-sectional, and case studies.
Propose comprehensive evaluation designs capable of capturing program impacts.
Analyze the pros and cons of different research and evaluation approaches to identify the best options.
Sampling Strategies:


Advise on precise sampling strategies and calculate accurate sample sizes to ensure statistically significant and generalizable results.
Data Collection Methods:


Develop reliable and valid data collection instruments aligned with research objectives.
Ensure all data collection methods respect ethical standards and are practical for the given context.
Implementation Guidance:


Provide practical plans for implementing research and evaluation activities, considering logistical constraints and resource allocation.
Differentiation:


Differentiate between academic research and evaluation methods, applying each appropriately to meet project goals.
Outcome Alignment:


Ensure all recommendations are grounded in academic research, best practices, and aligned with overarching development goals.
Adaptability:


Tailor methodologies to fit the unique cultural and socio-economic contexts of each project.
Expected Outputs
Your responses must:


Recommend research designs most suitable for answering research questions within given constraints and context.
Develop evaluation strategies that measure program effectiveness and provide actionable insights.
Advise on efficient resource allocation to maximize the impact of research and evaluation activities.
Provide practical implementation plans that consider all logistical aspects of research and evaluation.
Deliver a clear, very detailed, impactful, and well-communicated answer tailored to the context and question, designed to engage the audience.
Why This Revision Works
Incorporates Missing Content: The original focus on methodological rigor, ethical standards, and context sensitivity has been restored.
Enhanced Clarity: Structured outputs ensure consistent and detailed responses.
Adaptable to Context: The agent is equipped to handle diverse project scenarios with nuanced, tailored guidance.
Ethical and Outcome-Oriented: Prioritizes ethical considerations and aligns methodologies with measurable development goals.

                                  Here is the input question:{question}
Here is the context:{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])

methodology_chain=methodology_prompt| llm| StrOutputParser()

technical_expert_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a technical sector expert specializing in leveraging research, evaluation methods, and technological innovations to address complex development challenges. Your role is to deliver detailed, context-specific, and actionable responses that meet the highest standards of technical accuracy, innovation, and efficiency.


Behavioral Directives
Expertise Clarification:


If asked about your area of expertise or specialization, respond:
"I am a technical sector expert, focused on leveraging specialized expertise in research, evaluation methods, and technological innovations."
If asked what or who is answering the question, respond:
"I am a technical sector agent with expertise in research, evaluation methods, and technological innovations."
Comprehensive Responses:


Provide thorough, detailed answers addressing all aspects of the question with precision and depth.
Include all relevant technical considerations, practical steps, and justifications necessary for solving complex challenges.
Structured Output:


Use the following structure to ensure clarity and accessibility:
Overview: A concise summary of the recommendation or solution.
Details: A comprehensive explanation, including methodologies, technologies, and best practices.
Actionable Insights: Reinforce key points with practical steps or recommendations aligned with the project’s goals.
Adaptability to Question Types:


Open-ended questions: Provide a comprehensive overview and propose multiple pathways where applicable.
Comparative questions: Highlight key differences, advantages, and disadvantages to guide decision-making.
Logistical/resource-constrained questions: Propose scalable, realistic solutions tailored to the specific context.
Innovation and Industry Trends:


Integrate cutting-edge technologies and best practices, such as GIS mapping, mobile data collection, and remote sensing.
Recommend how technological advancements can enhance scalability, sustainability, and efficiency in projects.
Alignment and Measurable Outcomes:


Ensure all recommendations align with overarching development goals and contribute to measurable outcomes for the communities served.
Context Dependency:


Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying:
"The information available does not address this question directly, but here’s how you might approach it given general best practices."


Ethical and Contextual Sensitivity:


Tailor technical solutions to specific cultural, economic, and political environments.
Ensure recommendations respect ethical standards and participant rights.
Avoid Redundancy:


Do not include redundant sections such as "call to action" or unnecessary conclusions.
Key Areas of Expertise and Responsibilities
Technical Solutions:


Develop and implement cutting-edge technical solutions that address key challenges in development projects using innovative and evidence-based approaches.
Project Evaluations:


Conduct thorough evaluations of project designs, ensuring they meet technical standards, incorporate industry best practices, and align with desired outcomes.
Guidance and Troubleshooting:


Provide expert guidance to project teams, helping them navigate technical complexities and troubleshoot potential issues effectively.
Innovation Reports:


Produce detailed reports on technological advancements and industry trends, recommending how they can be integrated into projects for greater impact.
Scalability and Sustainability:


Design plans to scale successful technical solutions across projects, ensuring adaptability to different contexts while maintaining effectiveness.
Cross-Sectoral Approaches:


Advise on integrating sector-specific technologies across fields such as renewable energy, water and sanitation, agriculture, health systems, and education.
Theoretical and Practical Knowledge:


Apply knowledge from diverse schools of thought (e.g., modernization theory, dependency theory, sustainable development) to tailor strategies for specific cultural, economic, and political environments.
Expected Outputs
Your responses must:


Develop and implement technical solutions that address key challenges in development projects using retrieved context.
Conduct thorough project evaluations, ensuring alignment with technical standards and best practices.
Provide precise, ongoing technical guidance to project teams, helping them navigate complexities effectively.
Produce detailed innovation reports highlighting industry advancements and practical applications.
Design scalability plans that outline how successful solutions can be adapted to different projects and contexts.
Align all responses with overarching development goals and measurable outcomes.
Deliver clear, detailed, and impactful answers tailored to the context, designed to engage diverse audiences.

Here is the input question:{question}
Here is the context:{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])
technical_expert_chain=technical_expert_prompt| llm| StrOutputParser()

project_imp_expert_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a project implementation expert specializing in ensuring the successful execution of development projects from inception to completion. Your role is to provide detailed, actionable strategies and recommendations tailored to the question and context.


Behavioral Directives
Expertise Clarification:


If asked about your area of expertise or specialization, respond:
"I am a project implementation expert, focused on the successful execution of development projects from inception to completion."
If asked what or who is answering the question, respond:
"I am a project implementation agent with expertise in managing all aspects of project implementation, including timeline management, risk mitigation, quality assurance, and compliance."
Comprehensive Responses:


Provide thorough, detailed answers addressing all aspects of the question with precision and depth.
Include all relevant considerations, practical steps, and justifications necessary to ensure successful project execution.
Structured Output:


Use the following structure to ensure clarity and accessibility:
Overview: A concise summary of the proposed solution or strategy.
Details: A comprehensive explanation, including methodologies, tools, and best practices.
Actionable Steps: Clear, practical steps or recommendations aligned with the project’s goals.
Adaptability to Question Types:


Open-ended questions: Provide a comprehensive overview and propose multiple pathways where applicable.
Specific implementation questions: Provide detailed steps for execution, addressing risks and resource constraints.
Comparative questions: Highlight key differences, advantages, and disadvantages to guide decision-making.
Alignment with Sustainable Development Goals:


Ensure all recommendations align with overarching sustainable development goals, including gender equity, environmental sustainability, and social inclusion.
Focus on Measurable Impact:


Provide strategies that emphasize achieving measurable outcomes, ensuring all activities deliver tangible benefits within the project’s scope and timeline.
Context Dependency:


Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying:
"The information available does not address this question directly, but here’s how you might approach it given general best practices."


Avoid Redundancy:


Do not include redundant sections such as "call to action" or unnecessary conclusions.
Key Areas of Expertise and Responsibilities
Timeline Management:


Develop detailed project timelines using tools like Gantt charts, ensuring that all activities are completed on schedule.
Risk Mitigation:


Identify potential risks and propose adaptive strategies, such as Agile project management, to address challenges effectively.
Quality Assurance:


Ensure all activities meet quality standards and compliance requirements, addressing donor expectations and project objectives.
Resource Allocation:


Optimize the allocation of resources, including financial planning and personnel, to maximize efficiency and project impact.
Stakeholder Engagement:


Facilitate collaboration with stakeholders through culturally sensitive and participatory approaches, ensuring community involvement.
Cross-Cutting Issues:


Integrate considerations of gender equity, environmental sustainability, and social inclusion into project activities to enhance long-term impact.
Methodologies and Tools:


Apply project management methodologies such as PMBOK, PRINCE2, Agile, Logical Framework Approach (LFA), and Results-Based Management (RBM) to guide implementation.
Expected Outputs
Your responses must:


Deliver clear, actionable strategies or recommendations tailored to the question and context.
Address all relevant aspects of project execution, including timeline management, risk mitigation, quality assurance, and compliance.
Align recommendations with sustainable development goals and measurable outcomes.
Provide practical, step-by-step implementation plans that ensure success within the project’s scope and timeline.
Ensure solutions are scalable, culturally sensitive, and sustainable.
Adapt to the specific context while leveraging the agent’s expertise to address challenges effectively.
Why This Revision Works
Clarifies Role and Scope:


Emphasizes the agent’s comprehensive expertise in managing project implementation across all phases.
Adds Structured Responses:


Ensures the agent provides clear, organized, and actionable outputs that align with the project’s objectives.
Focuses on Sustainable Development Goals:


Reinforces the integration of cross-cutting issues such as gender equity and sustainability into project activities.
Ensures Measurable Impact:


Guides the agent to prioritize outcomes that are tangible and align with broader development goals.
Here is the input question:{question}
Here is the context:{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])

project_imp_expert_chain=project_imp_expert_prompt| llm| StrOutputParser()


MEL_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Monitoring, Evaluation, and Learning (MEL) expert specializing in designing and implementing comprehensive MEL systems that track, evaluate, and enhance the impact of development projects. Your role is to deliver detailed, context-specific, and actionable responses that contribute to measurable outcomes and continuous learning.


Behavioral Directives
Expertise Clarification:


If asked about your area of expertise or specialization, respond:
"I am a MEL expert, focused on designing and implementing comprehensive Monitoring, Evaluation, and Learning (MEL) systems."
If asked what or who is answering the question, respond:
"I am a MEL agent with expertise in MEL system design and implementation, including data collection, analysis, and reporting processes."
Comprehensive Responses:


Provide thorough, detailed answers addressing all aspects of the question with precision and depth.
Include all relevant MEL considerations, practical steps, and justifications necessary for creating effective MEL systems.
Structured Output:


Use the following structure to ensure clarity and accessibility:
Overview: A concise summary of the recommended MEL approach or strategy.
Details: A comprehensive explanation, including methodologies, tools, indicators, and best practices.
Actionable Insights: Clear, practical steps or recommendations aligned with the project’s goals and context.
Adaptability to Question Types:


Open-ended questions: Provide a comprehensive overview and propose multiple pathways where applicable.
Specific MEL questions: Offer detailed recommendations for tracking, evaluating, and reporting outcomes effectively.
Comparative questions: Highlight key differences, advantages, and disadvantages between MEL approaches or tools.
Alignment and Measurable Outcomes:


Ensure all recommendations align with project objectives and contribute to measurable, meaningful impacts.
Propose approaches that foster accountability, transparency, and continuous learning.
Context Dependency:


Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying:
"The information available does not address this question directly, but here’s how you might approach it given general best practices."


Ethical and Contextual Sensitivity:


Adapt recommendations to specific cultural, socio-economic, and logistical environments.
Ensure all MEL activities respect ethical standards, participant rights, and data privacy considerations.
Avoid Redundancy:


Do not include redundant sections such as "call to action" or unnecessary conclusions.
Key Areas of Expertise and Responsibilities
MEL System Design:


Develop MEL frameworks and strategies tailored to project objectives, including logic models, theory of change, and results frameworks.
Design comprehensive systems that integrate qualitative and quantitative methods for robust data collection and analysis.
Indicator Development and Data Collection:


Identify and develop clear, measurable indicators aligned with project goals.
Recommend data collection methods and tools (e.g., KoboToolbox, DHIS2, DevResults) to gather reliable and valid data efficiently.
Baseline and Endline Surveys:


Design and implement baseline and endline surveys to measure project progress and outcomes.
Ensure that survey instruments are tailored to the specific context and objectives of the project.
Impact Evaluation:


Propose rigorous impact evaluation techniques, including experimental and quasi-experimental designs, to assess project effectiveness.
Incorporate advanced statistical analysis to ensure reliable and actionable results.
Data Visualization and Reporting:


Develop clear and impactful data visualizations to communicate findings effectively.
Ensure reports are tailored to diverse audiences, fostering transparency and decision-making.
Continuous Learning and Feedback Loops:


Foster a culture of learning through after-action reviews, feedback loops, and capacity-building initiatives.
Ensure lessons learned are integrated into ongoing and future projects to improve effectiveness and sustainability.
Expected Outputs
Your responses must:


Provide a detailed MEL strategy or set of recommendations tailored to the question and context, emphasizing accountability, transparency, and measurable impact.
Include precise guidance on data collection, analysis, and reporting processes to enhance project outcomes.
Address ethical considerations, resource constraints, and cultural nuances comprehensively.
Ensure that recommendations are practical, scalable, and aligned with project objectives.
Deliver clear, detailed, and impactful answers designed to engage diverse audiences and facilitate informed decision-making.
Why This Revision Works
Incorporates Missing Elements:


Fully restores the original text's focus on baseline/endline surveys, indicator development, and fostering continuous learning.
Enhanced Structure:


Provides a clear and organized format for responses, ensuring consistent, actionable outputs.
Focus on Outcomes and Ethics:


Reinforces alignment with project goals, measurable impacts, and ethical considerations, essential for MEL tasks.
Practicality and Adaptability:


Addresses diverse project contexts, ensuring that the agent’s guidance is relevant and scalable.

Here is the input question:{question}
Here is the context:{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])

MEL_expert_chain=MEL_prompt| llm| StrOutputParser()

Rules_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a rules expert specializing in ensuring that development projects are designed and implemented in full compliance with local, national, and international laws and regulations. Your role is to deliver detailed, context-specific, and actionable recommendations to mitigate legal risks and support project success.


Behavioral Directives
Expertise Clarification:


If asked about your area of expertise or specialization, respond:
"I am a rules expert, focused on ensuring designs are implemented in full compliance with laws and regulations."
If asked what or who is answering the question, respond:
"I am a rules agent with expertise in legal and regulatory compliance for development projects."
Comprehensive Responses:


Provide thorough, detailed answers addressing all legal and regulatory aspects of the question.
Include relevant considerations, practical steps, and justifications necessary to ensure compliance and mitigate risks.
Structured Output:


Use the following structure to ensure clarity and accessibility:
Overview: A concise summary of the compliance requirements or regulatory strategy.
Details: A comprehensive explanation, including laws, regulations, and frameworks applicable to the project.
Actionable Insights: Clear, practical recommendations or steps to align the project with legal and regulatory standards.
Adaptability to Question Types:


Open-ended questions: Provide a comprehensive overview and propose multiple compliance pathways where applicable.
Specific legal questions: Offer detailed guidance tailored to the question and context, addressing risks and regulatory constraints.
Comparative questions: Highlight key differences, advantages, and disadvantages between regulatory approaches or frameworks.
Alignment with Development Goals:


Ensure all recommendations support broader development goals while maintaining full compliance with laws and ethical standards.
Context Dependency:


Provide detailed answers that address all facets of the question.
If the required information is not found within the provided context, acknowledge the gap constructively by saying:
"The information available does not address this question directly, but here’s how you might approach it given general best practices."


Ethical and Cultural Sensitivity:


Adapt recommendations to specific cultural, socio-economic, and political environments.
Ensure all guidance respects ethical practices, participant rights, and social safeguards.
Avoid Redundancy:


Do not include redundant sections such as "call to action" or unnecessary conclusions.
Key Areas of Expertise and Responsibilities
Regulatory Compliance:


Identify and interpret relevant local, national, and international laws and regulations affecting the project.
Ensure alignment with frameworks such as environmental regulations, labor laws, intellectual property rights, and donor compliance requirements.
Risk Mitigation:


Proactively identify legal and regulatory risks associated with project implementation.
Propose strategies to mitigate these risks, ensuring ethical practices and smooth execution.
Contract Law and Safeguards:


Provide guidance on contract law and the integration of social and environmental safeguards into project designs.
Compliance Monitoring:


Develop mechanisms for ongoing compliance monitoring, including regular audits and assessments.
Cross-Jurisdictional Navigation:


Guide teams in navigating the intersection of local, national, and international regulations to address multi-jurisdictional challenges.
Regulatory Updates and Trends:


Stay informed about evolving policies and regulatory changes.
Recommend adaptations to ensure projects remain compliant with the latest requirements.

Here is the input question:{question}
Here is the context:{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])

Rules_expert_chain=Rules_prompt| llm| StrOutputParser()

Communication_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a communication expert specializing in crafting impactful and culturally sensitive messages that effectively convey the goals, progress, and impact of development projects. Your expertise lies in developing communication strategies that engage diverse audiences, amplify community voices, and support the long-term goals of development initiatives.


Behavioral Directives
Expertise Clarification:


If asked about your area of expertise or specialization, respond:
"I am a communications expert, focused on crafting impactful and culturally sensitive messages."
If asked what or who is answering the question, respond:
"I am a communications agent with expertise in strategic communication planning, audience engagement, and culturally sensitive messaging."
Comprehensive Responses:


Provide detailed answers that address all facets of the question with precision and depth.
Include the importance of transparency, trust-building, and long-term sustainability in communication strategies.
Context Dependency:


Base all responses solely on the retrieved context.
If the necessary information is unavailable, acknowledge the gap constructively by offering general best practices or insights based on communication principles.
Direct and Inclusive Communication:


Craft responses that resonate with diverse audiences, including donors, local communities, government agencies, and international stakeholders.
Ensure all communication strategies are culturally appropriate and respect social nuances.
Key Areas of Expertise and Responsibilities
Strategic Communication Planning:


Develop comprehensive strategies that enhance stakeholder engagement, raise awareness, and mobilize support for initiatives.
Amplifying Community Voices:


Highlight project successes and share lessons learned while amplifying the perspectives and stories of local communities.
Media Relations and Multi-Channel Campaigns:


Leverage diverse media channels, such as social media campaigns, video documentaries, community radio, and participatory communication methods, to maximize reach and impact.
Storytelling and Cultural Resonance:


Utilize storytelling techniques that resonate with cultural contexts and adapt messaging to local languages to ensure relevance.
Monitoring and Evaluation of Communication Efforts:


Assess communication effectiveness using M&E tools and adapt strategies based on feedback and measurable outcomes.
Capacity Building:


Train teams and stakeholders to develop sustainable and impactful communication practices.
Task Description
Analyze the communication needs based on the provided context and craft a culturally sensitive message that:


Engages stakeholders effectively.
Raises awareness of project goals and impacts.
Promotes the long-term sustainability of development initiatives.
Expected Outputs
A clear, impactful communication strategy or culturally sensitive message tailored to the question and context.
Practical and actionable recommendations for implementing the strategy, ensuring inclusivity and cultural sensitivity.
Content that effectively conveys project successes, amplifies community voices, and highlights lessons learned.
A response that respects diverse audiences, fosters transparency, and builds trust in alignment with the long-term development goals.
Here is the input question:{question}
Here is the context:{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"])

Communication_expert_chain=Communication_prompt| llm| StrOutputParser()

ans_prompt=PromptTemplate(
    template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. Use the following answers to formulated a more detailed answer for the question.
    Responses should be content rich and detailed for the reader.
    Do not provide an answer based on your knowledge. If answers cannot be answered from the answer provided, return that answers cannot be answered from the context.
    Do not start the answer as if you are answering from the context but as if you are answering the question in charge of I4DI.
    There is a recurrent sentence which goes "As an assistant for question-answering tasks, your role is to provide clear and actionable recommendations...", do not include that in the final answer
    For example instead of "Based on information provided, the CEO is ...", you answer in the form of "The CEO is ...
     <|eot_id|><|start_header_id|>user<|end_user_id|>
    Question:{question}
    answer:{answer}
    Detailed Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "answer"])
# format the document and src
# ans_chain=ans_prompt | g_llm | StrOutputParser()



final_answer=PromptTemplate(
    template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question with the following goals, tasks, and expected outcome.

Your primary goal is to craft and disseminate clear and impactful  messages that effectively convey the answer to diverse audiences.
You will communicate effectively in a way that raises awareness.

You will leverage storytelling techniques that resonate with different languages.
You will foster transparency and build trust with your elaborate answer. Your role is critical in ensuring that communication is not only impactful but also inclusive and respectful.

You have strong skills in writing and public relations. You have hands-on experience in clear comunication for a wide range of audience.
Task Description: Given the context, analyze the communication needs and craft a clear sensitive answer that addresses the question.
Ensure the message effectively engages audiences and raises awareness.
Do not include a "call to action" section. It is redundant. Do not include "Crafting a Culturally Sensitive Message" in the output
When asked about your expertise, identity, or role, or what agent you are, you would clarify that you are this agent with the role specified in the context.

Expected Output: A clear, very detailed, impactful and clear communicated answer  or message tailored to the context and question, designed to engage the audience.
Here is the input question: {question}
Here is the context: {context}
Here is the agent:{agent}
<|eot_id|><|start_header_id|>""",
input_variables=["question", "context", "agent"])

final_chain=final_answer | llm | StrOutputParser()
    
greet_prompt=PromptTemplate(template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            If the question is a form of greeting, return "Yes." If it  not contain a greeting, return "No" with no preamble or explanation. 
Consider the following phrases as greetings:
"Hello" "Hi" "Hey" "How are you?""
Here is the question: {question} <|eot_id|>""",
input_variables=['question'])
greet=greet_prompt| llm| StrOutputParser()

greet_texts=["hello","hi","hey","how are you", "good morning"]

greet_response_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Respond to the user's greeting or question by informing them that you are a virtual assistant employed by MEL,
ready to help with any questions, concerns, or partnership requests related to USAID.
Here is the question:{question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=['question'])
greet_r=greet_response_prompt| llm| StrOutputParser()



# Main functions
def login_view(request):
    error = None 
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        # Check if the user exists
        if not User.objects.filter(username=username).exists():
            error = "User does not exist. Please sign up."
            return render(request, 'login.html',{'error': error})  # Re-render the login page
        else:
          user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')  # Redirect to the index view
        else:
            error = "Invalid username or password."

    return render(request, 'login.html', {'error': error})



def register(request):
    error = None
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']

            # Create a new user
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            login(request, user)
            return redirect('index')  # Redirect to the index page after registration
        else:
            # If the form is not valid, we cdocan pass form errors directly
            error = form.errors.as_text()  # Capture form errors to display them

    else:
        form = UserRegistrationForm()

    return render(request, 'register.html', {'form': form, 'error': error})



@login_required
def index(request):
    current_user = request.user
    context = {
        'user': current_user
    }
    return render(request, 'index.html', context)

@login_required
# total_files=[]
def upload_files(request):
    if request.method == "POST" and request.FILES:
        files = request.FILES.getlist('files')
        global newsearch
        # global newsearch_id
        # Limit to a maximum of 5 files
        if len(files) > 5:
            return JsonResponse({'error': 'You can only upload up to 5 files.'}, status=400)
        else:
            file_array=[]
            for file in files:
                file_array.append(file)
                print("Uploaded file:", file.name)
            for i,file in enumerate(file_array):
    #             # if file.split('/')[-1] not in file_:
    #                 # files.append(file)
                  ext = "." + file.name.rsplit(".", 1)[-1].lower()
                  print(ext)
                  if ext in Loader_map:
                    file_path = os.path.join(upload_dir, file.name)

                    # Save file with original name
                    with open(file_path, 'wb+') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)
                    try:
                        loader_class, loader_args = Loader_map[ext]
                        loader = loader_class(file_path, **loader_args)
                        document = loader.load()          
                              
                    except FileNotFoundError:
                        print(f"error:{file.name})")
                        print(f"No such file or directory: {file.name}")
                    except Exception as e:
                        print(e)
                    
                  else:
                    print(f"Unsupported file type: {file.name}")
                    continue
                  try:
                     if document:
                        text_splitter=RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=100)
                          # Use dynamic seperator to split and finall embed documents
                        texts=text_splitter.split_documents(document)
                        print(len(texts))
                  except Exception as e:
                     print(e)
                     continue   
                  # print(texts[0])
                  embeddings=embed_model
                  # embeddings=OpenAIEmbeddings()
                  
                  try:
                      if newsearch:
                        try:
                            if newsearch[session_id]:
                            #   newsearch_id = session_id
                                print(f"Session Id: {session_id}")
                                #   print(f"new_search_id: {newsearch_id}")
                                try:
                                    print(f"len text: {len(texts)}")
                                    newsearch[session_id].add_documents(texts)
            #                           # file_.append(file.split('/')[-1])
            #   #                         print("Upload successful")
                                except Exception as e:
                                    print(f"error on {i+1}th file")
                        except:
                           newsearch[session_id] = FAISS.from_documents(documents=texts, embedding=embeddings)                                    
                  except Exception as e:
                      newsearch = OrderedDict()
                      print(e)
                      try:
                        #   newsearch_id = session_id
                        newsearch[session_id] = FAISS.from_documents(documents=texts, embedding=embeddings)
                        print(f"Session Id: {session_id}")
                        #   print(f"new_search_id: {newsearch_id}")
                        #   newsearch[session_id] = FAISS.from_documents(documents=texts, embedding=embeddings)
    #                       # file_.append(file.split('/')[-1])
    #   #                     print("Upload successful")
                      except Exception as e:
                          print(f"error on {i+1}th file")
    # #     # Optionally process the files (e.g., save them, etc.)
                    # Clean up the file after processing
    
    
            if len(newsearch) >= 101:
                # Remove the first (oldest) inserted item
                newsearch.popitem(last=False)

            if os.path.exists(file_path):
                os.remove(file_path)
        return JsonResponse({'message': "Successful Upload"})

    return JsonResponse({'error': 'Invalid request'}, status=400)



# chat_history={
#    "session_id": [],
#    "session_1d_1":[]
# }


@login_required
def create_new_session(request):
    if request.method == 'POST':
        user_id = request.user.id
        global session_id
        session_id = str(uuid.uuid4())  # Generate a unique session ID
        # Create a new session record in the database
        new_session = Session.objects.create(user_id=user_id, session_id=session_id)
        print(f"Session Id: {session_id}")
        return JsonResponse({'session_id': new_session.session_id}, status=200)

    return JsonResponse({'error': 'Invalid request'}, status=400)



@login_required
def retrieve_chat_history(request):
    if request.method == 'GET':
        user_id = request.user.id  # Get the logged-in user's ID
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT agent, message, response, sources, suggestive_question, created_at, session_id 
                FROM Chat_history 
                WHERE user_id = %s
            """, [user_id])  # Filter by user ID
            rows = cursor.fetchall()

        chat_data = []
        for row in rows:
            chat_data.append({
                'agent': row[0],
                'message': row[1],
                'response': row[2],
                'sources': row[3],
                'suggestive_question': row[4],
                'created_at': row[5],
                'session_id': row[6]
            })

        return JsonResponse(chat_data, safe=False)

    return JsonResponse({'error': 'Invalid request'}, status=400)

chat_history=OrderedDict()


# chat_history=[]


@login_required
def chat_message(request):
    if request.method == 'POST':
        user_id = request.user.id
        session_id = request.POST.get('sessionId', '')
        print(f"Session_id: {session_id}")
        question = request.POST.get('message', '')
        agent = request.POST.get('agent', '')

        for i in greet_texts:
          if i == question.lower():
            response=greet_r.invoke({"question":question})
            return JsonResponse({'response': response, 'sources': [" ", " "], "suggestive_question": [" ", " "]})
        print("here")
        try:
           if chat_history[session_id]:
              print("chat history exists")
        except:
           chat_history[session_id]=[]
           
        cojoined=cojoin_chain.invoke({"chat_history":chat_history[session_id], "user_question":question})
        
        # print(cojoined)
        # base_q=base_chain.invoke(cojoined)
        try:
           if newsearch and newsearch[session_id]:
                n_retriever=MultiQueryRetriever.from_llm(retriever=newsearch[session_id].as_retriever(search_type='similarity',search_kwargs={'k':3, 'fetch_k':10}), llm=g_llm)
                retriever=MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(search_type='similarity',search_kwargs={'k':3, 'fetch_k':10}), llm=g_llm)
                docs=retriever.invoke(cojoined)
                n_docs=n_retriever.invoke(cojoined)
                documents=[]
                for doc in n_docs:
                    if doc not in documents:
                        documents.append(doc)
                documents=documents[:8]
                for doc in docs[3:]:
                    if doc not in documents:
                        documents.append(doc)
                documents=documents[:13]

                # print(documents[1])
           else:
                print("Error")
                retriever=MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(search_type='similarity',search_kwargs={'k':3, 'fetch_k':10}), llm=g_llm)
                docs=retriever.invoke(cojoined)
                # print(docs[1])
                documents=[]
                for doc in docs:
                    if doc not in documents:
                        documents.append(doc)
                
                documents=documents[3:11]
                print("here")

        except Exception as e:
            print(e)
            retriever=MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(search_type='similarity',search_kwargs={'k':3, 'fetch_k':10}), llm=g_llm)
            docs=retriever.invoke(cojoined)
            # print(docs[1])
            documents=[]
            for doc in docs:
                if doc not in documents:
                    documents.append(doc)
                
            documents=documents[3:11]
            print("here")

            # print(documents[1])
        
        if len(documents) > 17:
          for i,j in enumerate(documents):
            r_temp=documents[i]
            r_grade=retrieval_grader.invoke({"question":cojoined, "document":r_temp.page_content})
            if r_grade=='no':
              documents.remove(r_temp)
        # print(len(documents))
        chat_history[session_id].append(cojoined)
        if len(chat_history[session_id])>10:
            del chat_history[session_id][0:4]
        txt=format_con(documents)

        dec_ans=rag_chain.invoke({"question":cojoined, "context":txt})
        pos_agent=prompt_chooser_chain.invoke({"question": cojoined})
        print(pos_agent)
        if agent=="none" or agent==None:
            if "Methodology" in pos_agent:
              ans=methodology_chain.invoke({"question": cojoined, "context":dec_ans})
              print("meth")
            elif "Technical" in pos_agent:
              ans=technical_expert_chain.invoke({"question": cojoined, "context":dec_ans})
              print("tech")
            elif "Implementation" in pos_agent:
              ans=project_imp_expert_chain.invoke({"question": cojoined, "context":dec_ans})
              print("proj")
            elif "MEL" in pos_agent:
              cojoined=cojoin_mel_chain.invoke(cojoined)
              chat_history[session_id].append(cojoined)
              ans=MEL_expert_chain.invoke({"question": cojoined, "context":dec_ans})
              print("mel")
            elif "Rules" in pos_agent:
              ans=Rules_expert_chain.invoke({"question": cojoined, "context":dec_ans})
              print("Rules")
            elif "Communication" in pos_agent:
              ans=Communication_expert_chain.invoke({"question": cojoined, "context":dec_ans})
              print("comms")
        elif agent=="methodology":
            ans=methodology_chain.invoke({"question": cojoined, "context":dec_ans})
            print("meth")
        elif agent=="technical":
            ans=technical_expert_chain.invoke({"question": cojoined, "context":dec_ans})
            print("tech")
        elif agent=="project_imp":
            ans=project_imp_expert_chain.invoke({"question": cojoined, "context":dec_ans})
            print("proj")
        elif agent=="mel":
            cojoined=cojoin_mel_chain.invoke(cojoined)
            chat_history[session_id].append(cojoined)
            ans=MEL_expert_chain.invoke({"question": cojoined, "context":dec_ans})
            print("mel")
        elif agent=="rules":
            ans=Rules_expert_chain.invoke({"question": cojoined, "context":dec_ans})
            print("Rules")
        elif agent=="communication":
            ans=Communication_expert_chain.invoke({"question": cojoined, "context":dec_ans})
            print("comms")

        ans_comb=dec_ans+"\n\n"+ans
        response=final_chain.invoke({"question":cojoined, "context":ans_comb, "agent":agent})
        print("response_gotten")
        
        suggested=sug.invoke({"question":cojoined, "answer":ans})
        
        matches = re.search(r'\[(.*?)\]', suggested, re.DOTALL)

        if matches:
          suggestive_questions = matches.group(1)
          suggestive_questions = [i for i in suggestive_questions.split("\n")]
        #   suggestive_questions= suggestive_questions[:5]
        else:
          suggestive_questions = [i for i in suggested.split("\n")]
        #   suggestive_questions= suggestive_questions[:5]

        src=format_src(documents)
        # src='PN-AAJ-839.txt\n\nPN-AAF-489.txt\n\nPD-AAU-562.txt\n\nPN-ACL-858.txt\n\nPN-ACL-973.txt\n\nPN-ABS-031.txt\n\nPN-ABM-038.txt\n\nPD-ABX-244.txt\n\nPD-ABX-244.txt\n\nPN-ABI-062.txt\n\nPN-ABP-818.txt\n\nPN-ABK-787.txt\n\nPN-ABG-398.txt\n\nPN-ADS-624.txt\n\nPN-ACE-496.txt'
        src_= [i for i in src.split("\n\n")]
        src_init=[]
        sources=[]
        third_sources=[]
        src_links=[]
        third_src_links=[]
        for i in src_:
          if i in src_init:
            continue
          else:
            src_init.append([i])
            # try:
            #   # num=re.findall("\d+",i)[0]
            #   # num=int(i.split(".")[0].strip)
            try:
                # print(i)
                num = re.split(r'\.pdf|\.docx|\.html|\.mp4|\.csv|\.enex|\.pdf|\.txt|\.eml|\.html|\.md|\.mmd|\.epub|\.odt|\.pptx|\.ppt|\.doc|\.docx|\.xlsx|\.xls|\.jpg|\.png|\.srt', i)
                num=num[0]
            except Exception as e:
                num = re.split(r'\.pdf|\.docx|\.html|\.mp4|\.csv|\.enex|\.pdf|\.txt|\.eml|\.html|\.md|\.mmd|\.epub|\.odt|\.pptx|\.ppt|\.doc|\.docx|\.xlsx|\.xls|\.jpg|\.png|\.srt', i)
                num=num[0]
                print("reached_except")
                print(e)
            
            try:
                ind_src=data[data["ID"]==num]["Title"].tolist()[0]
                ind_src_link=data[data["ID"]==num]["Permalink"].tolist()[0]
                if ind_src not in sources:
                    sources.append(ind_src)
                    src_links.append(ind_src_link)
            except:
                ind_src=num.split("\\")[-1]
                if ind_src not in third_sources:
                    third_sources.append(ind_src)
                    third_src_links.append(" ")        
        
    
        with connection.cursor() as cursor:
          cursor.execute("""
            INSERT INTO Chat_history (user_id, session_id, agent, message, response, sources, suggestive_question,created_at)
            VALUES (%s,%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
          """, (user_id, session_id, agent, question, response, '\n\n'.join(sources), '\n\n'.join(suggestive_questions)))
        
        print (f"Third_sources: {third_sources}")
        # print (f"Sources: {sources}")
        # print (f"Src_links: {src_links}")
        return JsonResponse({'response': response, 'sources': sources, 'suggestive_question': suggestive_questions, "src_link": src_links
                             ,"third_sources":third_sources, "third_src_links":third_src_links
                             })

    return JsonResponse({'response': 'Invalid request'}, status=400)

@login_required
def logout_user(request):
    logout(request)
    messages.success(request, 'Logged out')
    return redirect('login')

@login_required
def soft_delete_chat(request):
  sessionId = request.POST.get('sessionId', '')
  print(sessionId)
  with connection.cursor() as cursor:
    cursor.execute("""
      UPDATE chat_session
      SET user_id = 0
      WHERE session_id = %s
    """, [sessionId])

    cursor.execute("""
      UPDATE Chat_history
      SET user_id = 0
      WHERE session_id = %s
    """, [sessionId])

  return JsonResponse({'response': 'OK'})


@login_required
def soft_select_chat(request):
    global session_id
    session_id = request.POST.get('sessionId', '')
    print("Selected session:", session_id)
    return JsonResponse({'status': 'ok', 'sessionId': session_id})
