from hugchat import hugchat
from hugchat.login import Login

# from dotenv import load_dotenv
# import os

# load_dotenv()


class llm_class:
    def __init__(self, email, passwd):
        self.email = email
        self.passwd = passwd
        try:
            # print(email, passwd)
            sign = Login(self.email, self.passwd)
            cookies = sign.login()
            self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
            self.chatbot.switch_llm(3) # for mistral 8*7B
            # self.chatbot.switch_llm(2) # for hfH4
            # self.chatbot.switch_llm(5)  # for gemma
        except Exception as e:
            print("Invalid credentials huggingchat", e)

    async def ask(self, query: str, context: str):
        prompt = f"""
        You are given a detailed image description of multiple images as context and you have to answer the asked question:
        ----------
        Rules:
        1) Do not hallucinate. If you don't know the answer just tell that you don't have the record for that question.
        2) Just return the answer of the asked question do not give any excess information.
        3) The answer should be precise and to the point.
        ----------
        Context: {context}
        Question: {query}
        Answer:
        """
        res = self.chatbot.query(prompt)
        return res["text"] or ""
