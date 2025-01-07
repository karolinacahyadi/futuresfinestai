import streamlit as st
import requests
import csv
import datetime

API_URL = "http://127.0.0.1:8000/ff_chatbot"

def send_message_to_api(message, history_input, history_output):
    payload = {
        "message": message,
        "history_input": history_input,
        "history_output": history_output,
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}")
        return None

if 'history_input' not in st.session_state:
    st.session_state.history_input = []
if 'history_output' not in st.session_state:
    st.session_state.history_output = []

st.markdown(
    """
    <style>
        .title {
            font-size: 50px;
            color: transparent;
            background: linear-gradient(45deg, #6a0dad, #1e90ff); /* Purple-blue gradient */
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold;
            text-align: center;
            font-family: 'Orbitron', sans-serif; /* Futuristic font */
            text-shadow: 2px 2px 8px rgba(106, 13, 173, 0.7), 0 0 25px rgba(30, 144, 255, 0.6); /* Modern glowing effect */
        }
        .stTextInput input {
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">FUTURES FINEST🎮</div>', unsafe_allow_html=True)

message = st.chat_input("Ask me anything about games!")

if message:
    api_response = send_message_to_api(
        message,
        st.session_state.history_input,
        st.session_state.history_output,
    )

    if api_response:
        response_text = api_response.get("response", "")

        st.session_state.history_input.append(message)
        st.session_state.history_output.append(response_text)


if st.session_state.history_input:
    st.write("### Conversation:")
    for i in range(len(st.session_state.history_input)):

        input_message = f'<span style="font-size: 2em;">🧒🏻</span>: {st.session_state.history_input[i]}'
        output_message = f'<span style="font-size: 2em;">🎮</span>: {st.session_state.history_output[i]}'

        st.markdown(input_message, unsafe_allow_html=True)
        st.markdown(output_message, unsafe_allow_html=True)


with st.sidebar:
    with st.expander("Frequently Asked Questions", expanded=False):
        st.markdown(
            """
            <style>
                .faq-title {
                    font-size: 25px;
                    color: transparent;
                    background: linear-gradient(45deg, #6a0dad, #1e90ff); /* Purple-blue gradient */
                    -webkit-background-clip: text;
                    background-clip: text;
                    font-weight: bold;
                    text-align: left;
                    font-family: 'Orbitron', sans-serif; /* Futuristic font */
                    text-shadow: 2px 2px 8px rgba(106, 13, 173, 0.7), 0 0 25px rgba(30, 144, 255, 0.6); /* Glowing effect */
                }
            </style>
            """, unsafe_allow_html=True
        )

        st.markdown("""
        <div class="faq-title">Q: What is Futures Finest?</div>
        A: Futures Finest is an advanced chatbot designed to help gamers discover the perfect games based on their preferences. It analyzes data from a wide range of games and provides personalized recommendations, game details, comparisons, pricing, and reviews to enhance your gaming experience.

        <div class="faq-title">Q: What can I do with Futures Finest?</div>
        - Get Game Recommendations: Receive tailored game suggestions based on your genre, platform, and price preferences.  
        - Explore Game Details: Learn more about specific games, including name, genre, release date, price, and developer.  
        - Compare Games: Compare two or more games based on various metrics like price, reviews, genre, and platform.  
        - Check Prices: Find out the current price of a game across different platforms.  
        - Visualize Reviews: See visual representations of positive and negative reviews in the form of charts.

        <div class="faq-title">Q: What do I get from using Futures Finest?</div>
        - Personalized Recommendations: Get game suggestions that match your exact preferences.  
        - Comprehensive Game Information: Access detailed data about each game to make informed decisions.  
        - Informed Purchasing Decisions: Compare game prices, reviews, and other features to choose the best option for you.  
        - Visualized Review: Easily understand game reviews through Pie and Bar Chart representations.

        <div class="faq-title">Q: Can I get game suggestions based on specific preferences?</div>
        Yes, you can specify your preferred genre, platform (PC, Mac, Linux), price range, and more, and the chatbot will tailor recommendations based on those criteria.

        <div class="faq-title">Q: How does Futures Finest work?</div>
        Futures Finest uses a powerful AI system that analyzes a vast database of game data (including genres, reviews, prices, and platforms). The chatbot matches your preferences to the most relevant games and provides real-time, accurate information.

        <div class="faq-title">Q: Can I ask for game comparisons?</div>
        Yes, you can compare two or more games based on various metrics such as genre, price, reviews, and more. The chatbot will help you understand the differences and make a better choice.

        <div class="faq-title">Q: What if my game isn't found?</div>
        If your game isn't found, the chatbot will suggest alternative options or ask for more details, such as the exact game name, platform, or genre, to refine the search.
        """, unsafe_allow_html=True)

    st.header("Feedback")

    feedback = st.text_area("We'd love to hear your thoughts!", placeholder="Enter your feedback here...")

    if st.button("Submit Feedback"):
        if feedback:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                if file.tell() == 0:
                    writer.writerow(["Timestamp", "Feedback"])
                
                writer.writerow([timestamp, feedback])
            
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter your feedback before submitting.")