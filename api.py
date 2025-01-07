from typing import List
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import llm
from langchain.chains import ConversationChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant  # Pastikan Anda menginstal qdrant
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import matplotlib.pyplot as plt
from typing import Dict

app = FastAPI()

load_dotenv()

data = pd.read_csv('fixgames.csv') 
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
data_subset = data.head(1000)

documents = []
for idx, row in data_subset.iterrows():
    page_content = (
        f"Game Name: {str(row['Name'])}\n"
        f"About the Game: {str(row['About the game'])}\n"
        f"Genres: {str(row['Genres'])}\n"
        f"Categories: {str(row['Categories'])}\n"
        f"Tags: {str(row['Tags'])}\n"
        f"Supported Languages: {str(row['Supported languages'])}"
    )
    
    metadata = {
        "AppID": str(row['AppID']),
        "Release Date": str(row['Release date']),
        "Required Age": str(row['Required age']),
        "Price": str(row['Price']),
        "DLC Count": str(row['DLC count']),
        "Header Image": str(row['Header image']),
        "Website": str(row['Website']),
        "Windows Support": str(row['Windows']),
        "Mac Support": str(row['Mac']),
        "Linux Support": str(row['Linux']),
        "Positive Reviews": str(row['Positive']),
        "Negative Reviews": str(row['Negative']),
        "Developers": str(row['Developers']),
        "Publishers": str(row['Publishers']),
    }
    
    document = Document(page_content=page_content, metadata=metadata)

    documents.append(document)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large") 

qdrant = Qdrant.from_documents(
    documents,
    embeddings,
    path="./qdrant_data15",
    collection_name="ondisk_documents",
)

retriever = qdrant.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

your response rules:
- The response must be in the English.
- The response must be based on the context.
- If the context doesn't answer the question, suggest more relevant options (e.g., ask the user for more preferences like genre, platform, etc.).
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@tool
def recommend_game(input: str) -> str:
    """
    Recommend a game based on the user's preferences such as Name, Genre, Platform, Playstyle, and Similarity from other games.

    When a user requests a game recommendation, the system will:
    - If the user asks for a game similar to a specific name, 
      and that Name is not available in the database, the system will NOT provide a recommendation. 
      Instead, it will ask for further clarification from the user, such as:
      - I can't provide a recommendation
      - "Could you provide a preferred genre?"
      - "Do you have any specific playstyle in mind?"
      - "Would you like to explore a platform like PC, Mac, or Linux?"
      
    The system will NOT offer a recommendation based on similarity if the name is not in the database, 
    even if similar games exist in the database, because name of unavailable game cannot be directly analyzed.
    """

    context = retriever.get_relevant_documents(input)

    if not context:
        return {
            "category": "game_recommendation",
            "input": input,
            "recommendation": "No relevant games found. Could you provide more details such as preferred genre, platform, or other preferences?"
        }

    return {
        "category": "game_recommendation",
        "input": input,
        "recommendation": context,
    }

@tool
def get_game_details(input: str) -> str:
    """
    Retrieve details of a game such as Name, Genre, Platforms,Price, Release date, and Developer.

    - If the input is not found in the `documents` or is highly irrelevant, the system will:
      - Reply with a message such as:
        - "I couldn't find any game with the title. Could you please provide more specific details?"
        - "The title you entered doesn't match any game in the documents. Could you try a more specific game name?"
        - "Would you like to provide more details such as the game genre or platform?"

    The system will NOT return any game details if the Name is unavailable in the `documents`, and it will NOT attempt to provide a recommendation for a different game or make assumptions about the user's intent. 
    """
   
    context = retriever.get_relevant_documents(input)

    return {
        "category": "game_details",
        "input": input,
        "game_info": context
    }

@tool
def compare_games(input: str) -> str:
    """
    Compare two or more games based on metrics such as Name, Price, Positive, Negative, Developers, Publishers, Categories, Genres.
    
    If the input is not found in the `documents` or is highly irrelevant, the system will:
      - Not return any game details.
      - Reply with a message such as:
        - "I couldn't find any game with the title. Could you please provide more specific details?"
        - "The title you entered doesn't match any game in the documents. Could you try a more specific game name?"
    """
    context = retriever.get_relevant_documents(input)

    return {
    "category": "game_comparison",
    "game_compare": context,
    "input": input
    }

@tool
def visualize_review(positive: int, negative: int) -> dict:
    """
    Visualize reviews based on the number of positive and negative reviews.
    Generates a bar chart and a pie chart and displays them directly.
    
    Args:
    - positive (int): Number of positive reviews.
    - negative (int): Number of negative reviews.
    
    Returns:
    - A dictionary indicating that the visualizations have been shown, with a message about the reviews.
    """

    if positive == 0 and negative == 0:
        return {
            "category": "review_visualization",
            "positive_reviews": 0,
            "negative_reviews": 0,
            "message": "No reviews available for this game."
        }

    review_data = {'Positive': positive, 'Negative': negative}
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  
    
    ax[0].bar(review_data.keys(), review_data.values(), color=['purple', 'gold'])
    ax[0].set_title('Review Bar Chart')
    ax[0].set_ylabel('Number of Reviews')
    ax[0].set_xlabel('Review')

    ax[1].pie(review_data.values(), labels=review_data.keys(), autopct='%1.1f%%', startangle=90, colors=['purple', 'gold'])
    ax[1].set_title('Review Pie Chart')

    plt.tight_layout()

    plt.show()

    return {
        "category": "review_visualization",
        "positive_reviews": positive,
        "negative_reviews": negative,
        "message": "Review visualizations have been displayed."
    }

@tool
def get_game_price(input: str) -> str:
    """
    Retrieve the current price information for a specified game.

    This function:
    - Searches for the game using the provided input.
    - If the game is found, returns the current price information.
    - If not found, prompts the user for clarification (e.g., correct title or platform).

    If the input is not found in the `documents` or is highly irrelevant, the system will:
      - Not return any game details.
      - Reply with a message such as:
        - "I couldn't find any game with the title. Could you please provide more specific details?"
        - "The title you entered doesn't match any game in the documents. Could you try a more specific game name?"
    """
    context = retriever.get_relevant_documents(input)

    if context:
        price = context[0].metadata.get("Price", "Price information not available")
        
        return {
            "category": "game_price",
            "input": input,
            "Price": price,
        }
    else:
        return {
            "category": "game_price",
            "input": input,
            "message": f"I couldn't find details for the game '{input}'. It seems the game is not available in the database. "
                       "Could you provide more information such as the game name or platform?"
        }


class ChatRequest(BaseModel):
    message: str
    history_input: List[str]
    history_output: List[str]

memory = ConversationBufferMemory()

@app.post("/ff_chatbot")
async def ff_chatbot(request: ChatRequest):
    try:
        message = request.message
        history_input = request.history_input
        history_output = request.history_output

        prompt_template = """
        System: You are FuturesFinest, a Game expert and enthusiast. 
        Answer game-related questions with insightful and informative responses. 
        
        If the user ask about game recommendation you have to use the recommend_game tool.
        If the user ask information about game details you have to use the get_game_details tool.
        If the user ask to compare two or more games you have to use compare_games tool.
        If the user ask to visualize reviews or sentiment of a game you have to use visualize_review tool.
        If the user ask about game price you have to use the get_game_price tool.
        
        If the user asks about non-game-related topics, politely refuse to answer


        history to remember: {history}
        User: {input}
        
        """
        history = memory.load_memory_variables({})
        
        tools = [recommend_game, get_game_details, compare_games, visualize_review, get_game_price]

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "Question: {input})\nHistory Chat: {history}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

        print(message)
        print(history)
        agent_response = agent_executor.invoke({
            "input": message,
            "history": history,
        })

        print(agent_response)

        return {"response": agent_response["output"]}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))