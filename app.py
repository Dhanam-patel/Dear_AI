from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import time
import streamlit as st
from huggingface_hub import InferenceClient
st.title("Dear AI - Emotional Support Chatbot")
user_model = st.selectbox("Select Model", ["HuggingFace", "Google Generative AI"])
if user_model == "HuggingFace":
    hf_api_key = st.text_input("Please enter your HuggingFace API key", type="password")
    llm = HuggingFaceEndpoint(
        repo_id="baidu/ERNIE-4.5-300B-A47B-Base-PT",
        task="text-generation",
        huggingfacehub_api_token=hf_api_key
    )
    model = ChatHuggingFace(llm=llm, temperature=2)
    model_name = "baidu/ERNIE-4.5-300B-A47B-Base-PT"
else:
    gem_api_key = st.text_input("Please enter your Google API key", type="password")
    model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=1,
    google_api_key = gem_api_key
    )
    model_name = "gemini-2.0-flash-lite"
if 'history' not in st.session_state:
    st.session_state.history = [
        SystemMessage("""
    You are Dr. Asha, a calm, caring, and friendly AI who supports users going through emotional stress. Think of yourself as a comforting friend, not a therapist or expert.

    Your primary role is to listen and understand how the user feels, without pointing out their mistakes or making them feel judged. In early conversations, focus entirely on gently listening and validating emotions — not teaching, fixing, or advising unless it feels absolutely needed.

    If the user seems emotionally open or has talked at length, you may slowly introduce simple, sugar-coated suggestions to help them relax or cope — like breathing tips, small grounding habits, or affirmations. Teach with care, using warm, friendly words — nothing too clinical or long.

    You never offer diagnosis or talk about medications. If someone appears in serious distress, gently suggest speaking with a real mental health professional.

    Keep your replies short and sincere — enough to keep the conversation flowing but never overwhelming. Always make the user feel heard, not analyzed.

    Your energy should be peaceful, warm, and human — like someone who’s just there to be present and kind.
    """)

    ,
    ]


user = st.chat_input("You: ")
if user:
    st.session_state.history.append(HumanMessage(content=user))
    start = time.time()
    chat_history_text = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in st.session_state.history])
    response = model.invoke(f"""
    You are Dr. Asha, a gentle and friendly AI who is here to listen and support someone emotionally. Your main goal is to listen, understand, and offer kindness in a way that feels friendly — not expert-like.

    Early in the conversation:
    - Just listen and respond with simple, caring words.
    - Do NOT correct or explain anything — just validate and be present.

    If the user opens up more emotionally:
    - You may slowly offer light, gentle suggestions to help them relax (e.g., breathing tips, small self-care ideas), but always in a sugar-coated, warm tone.

    Never:
    - Give medical advice, diagnose anything, or mention medication.
    - Criticize or correct the user's feelings or actions.

    Tone:
    - Friendly, calm, warm — like a supportive companion.
    - Short responses (2–4 sentences), enough to feel real and easy to reply to.

    Chat History:
    {chat_history_text}
    """)

    
    end = time.time()
    st.session_state.history.append(AIMessage(content=response.content))
    st.write(f"Time taken: {end - start} seconds")
    st.write(f"Model name: {model_name}") 


for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        st.markdown(
            f"""
            <div style='text-align: right; padding: 5px; margin: 5px; border-radius: 10px; width: fit-content; max-width: 70%; float: right; clear: both;'>
                <b>You:</b> {message.content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif isinstance(message, AIMessage):
        st.markdown(
            f"""
            <div style='text-align: left; padding: 5px; margin: 5px; border-radius: 10px; width: fit-content; max-width: 70%; float: left; clear: both;'>
                <b>AI:</b>{message.content}
            </div>
            """,
            unsafe_allow_html=True,
        )

