from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# import pyttsx3
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import time
import streamlit as st
# engine = pyttsx3.init()
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="baidu/ERNIE-4.5-300B-A47B-Base-PT",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm,temprature=2)
chat_history = [
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

st.title("Dear AI - AI Based Mental health companion")

while True:
    # user = input("You: ")
    user = st.text_input("Your Message: ")
    chat_history.append(HumanMessage(content=user))
    st.write(f"You: {user}")
    if user.lower() == "exit":
        break
    if st.button("Send"):
        start = time.time()
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
        {chat_history}
        """)




        # print(f'AI:  {response.content}')
        st.write(f"AI: {response.content}")
        end = time.time()
        st.write(f"Time taken: {end - start} seconds")
    # print(f"Time taken: {end - start} seconds")
    # engine.say(response.content)
    chat_history.append(AIMessage(content=response.content))

    # engine.runAndWait()

print(chat_history)
