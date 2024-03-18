import os
import re
import datetime
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.agents import initialize_agent, AgentType,Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.utilities import SearxSearchWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import flask
from flask import Flask, render_template, request, redirect, url_for, session,Response
from deep_translator import GoogleTranslator


os.environ["GOOGLE_API_KEY"] = str("AIzaSyAusJRVBJdvcB7HRhXkVwQYi4HAV-P7f7M")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


#creating dict of model for tracking
dict_language={"English":'en-US',"हिंदी":'hi-IN',"ಕನ್ನಡ":'kn-IN',"తెలుగు":'te-IN',"தமிழ்":'ta-IN',"मराठी":'mr-IN'}
dict_translator={"English":'en',"हिंदी":'hi',"ಕನ್ನಡ":'kn',"తెలుగు":'te',"தமிழ்":'ta',"मराठी":'mr'}


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8,max_tokens=150,request_timeout=5,callbacks=[FinalStreamingStdOutCallbackHandler()],streaming=True, convert_system_message_to_human=True)
searx_search=SearxSearchWrapper(searx_host="http://metasearch.outgrowdigital.com",k=3) # k is for max number of items

### initializing gemeini chat Api################
def geminigpt(input_query):
    #input = "what is agriculture and how it helps people?" 
    query = '"{}"'.format(input_query)
    out = llm([ HumanMessage(content=query)]).content.lower()
    print(out)
    return out

### Outgrow knowledge RAG################################
prompt_template = """
You are Agricultural Assistant is designed to be able to assist with a wide range of tasks. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant can use his knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
#model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8, max_tokens=200,request_timeout=5, convert_system_message_to_human=True)

prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def outgrow_knowledge(input_query):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")    
        new_db = FAISS.load_local("faiss_index_outgrow", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(input_query)
        response = chain(
                {"input_documents":docs, "question": input_query}
                , return_only_outputs=False)

        print(response["output_text"])
        return response["output_text"]

#### Initializing tools and Agent################
tools = [ 
    Tool(
    name='searx-search',
    func= searx_search.run,
    description="Useful for when you need to do a search on the internet to find information of curent event. give result basis on indian country."
),
    Tool(
    name='outgrow_knowledge',
    func= outgrow_knowledge,
    description="Use this tool only if need to find information of pest or disease contol measures or symptoms."
),

    Tool(
    name='geminigpt',
    func= geminigpt,    
    description="Use this tool if outgrow_knowledge not given any answer"
)

]

agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION ,max_iterations=3,verbose=True)

fixed_prompt = '''
You are Agricultural Assistant is designed to be able to assist with a wide range of tasks. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant need to use searx-search tool if require internet to answer?. If question related to pest and disesae control measures or symptoms use outgrow_knowledge tool. if you not get information from outgrow_knowledge tool  use chat_gpt tool get answer.

Assistant can use his knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.
'''

agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

def chatbot(input):    
    reply=agent.run(input)
    return reply

###Input classification########################
Sysmessage='''you are a assistant - classify inputed sentence if it belongs to mentioned domain,
* agricultural domain.
* horticultural technique question.
* pest and disease. 
* Any information of fruits ,vegetables, flowers or crop related . 
* vegetable or crop market price or commodity prices.
* soil related.
* weather or forecast related.
* greatings or formal introduction sentence.
respond only with  yes or no'''
def input_classifying(text):    
    #text = "what are the facors affecting weather?"
    query = '"{}"'.format(text)
    reply = llm([SystemMessage(content=Sysmessage), HumanMessage(content=query)]).content.lower()
    print(reply,'relation to agri')
    return reply

### Tools classification #################################
directorSys = '''you are a assistant - classify inputed sentence if it require internet to answer or Belongs to weather related domain or Belongs to pest and disease control measures domain?.
                Respond only with yes or no.'''
def tool_classify(text):
    #text = "what are the facors affecting cricket?"
    query = '"{}"'.format(text)
    yn = llm([SystemMessage(content=directorSys), HumanMessage(content=query)]).content.lower()
    print(yn,' requirement of tools')
    return yn



app = Flask(__name__,template_folder="template",static_folder="static")
app.secret_key = '123456'  # Replace with a strong secret key
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

@app.route('/')
def dashboard():           
    return render_template('index.html', username='username123',farmernumber='farmer_number',location='location',internnumber='intern_number')


@app.route('/text')
def chatbot_response():
        text = request.args.get('msg')
        language_input = request.args.get('lang')
        username = 'session'
        print(username, text, language_input)
        input=text
        tool='no_tool' 
        warn_output = ""
        breaking_point=''
        try:
            if (len(text)==0):
                if language_input is None:
                    warn_output= "Please ask any question! "
                    text = GoogleTranslator(source='en',target=dict_translator['English']).translate(warn_output)   
                    print(warn_output,1)
                    breaking_point='Break'
                    return text
                else:
                    warn_output= "Please ask any question! "
                    text = GoogleTranslator(source='en',target=dict_translator[language_input]).translate(warn_output)   
                    print(warn_output,2) 
                    breaking_point='Break'        
                    return text
                
            if (language_input is None and breaking_point==''):
                warn_output= "Please select your language!"   
                print(warn_output) 
                breaking_point='Break'                    
                return warn_output

            try:
                text
                
            except:
                text="Please ask question properly, I can't understand what you are typing."
                text = GoogleTranslator(source='en',target=dict_translator[language_input]).translate(text)  
                breaking_point='Break'
                print(text)          
                return text                
            
            try:
                #cache_input=fetch_data(input,mydb)
                cache_input='No Record'
                print(cache_input)
                print('cache')         

            except:
                print('error in fetching data from db')
                cache_input='No Record'

        
            
            try:    
                if cache_input=='No Record':           
                    if language_input !="English" :
                        text = GoogleTranslator(source=dict_translator[language_input], target='en').translate(text)  
                        print(text)          
                        if 'yes'in input_classifying(text):
                            text1 =({'chat_history':[],"input":text})
                            if 'yes' in tool_classify(text):
                                text=chatbot(text1)
                                print(text)
                                text=GoogleTranslator(source='en',target=dict_translator[language_input]).translate(text)
                            else:
                                text = geminigpt(text1)
                                print(text)
                        else:
                            text='Please ask agriculture-related questions.'
                    else:
                         if 'yes'in input_classifying(text):
                            text1 =({'chat_history':[],"input":text})
                            if 'yes' in tool_classify(text):
                                text=chatbot(text1)
                                print(text)
                                text=GoogleTranslator(source='en',target=dict_translator[language_input]).translate(text)
                            else:
                                text = geminigpt(text1)
                                print(text) 
                         else:
                            text='Please ask agriculture-related questions.'            
                    return text    
                elif breaking_point=='':
                    breaking_point='Break'
                    return cache_input            

            except Exception as e:
                    text='We are experiencing high traffic. Please try again later.'
                    text = GoogleTranslator(source='en',target=dict_translator[language_input]).translate(text)
                    print('server error') 
                    print(e) 
                    return text         
        except Exception as e:
                    text='We are experiencing high traffic. Please try again later.'
                    text = GoogleTranslator(source='en',target=dict_translator[language_input]).translate(text)
                    print('server error') 
                    print(e)  
                    return text   


if __name__ == '__main__':
    app.debug = True
    app.run(port=9800)    
