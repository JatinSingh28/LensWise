# import cv2
import time
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import asyncio
from transformers import BertModel, BertTokenizer
from vector_db import vector_db_class

# from img_description import gen_description
from llava import gen_description

# from answer_query import answer_query
import os
from llm_hf import llm_class
from dotenv import load_dotenv


class LensWise:
    def __init__(
        self,
        user_id=1,
        email=os.getenv("hf_email"),
        passwd=os.getenv("hf_passwd"),
        vec_db_key=os.getenv("vec_db_key"),
    ):
        # self.cap = cv2.VideoCapture(0)

        load_dotenv()

        self.db = vector_db_class(vec_db_key)
        self.user_id = user_id

        self.model_name = "bert-base-uncased"
        self.embedding_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.embedding_model = BertModel.from_pretrained(self.model_name)

        self.llm = llm_class(email, passwd)

    async def text_to_embedding(self, text: str) -> np.ndarray:
        inputs = self.embedding_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

        embeddings = outputs.last_hidden_state
        sentence_embedding = torch.mean(embeddings, dim=1).squeeze().cpu().numpy()

        await asyncio.sleep(0)

        return sentence_embedding

    # async def capture_img(self, time_interval: int = 30):

    #     if not self.cap.isOpened():
    #         print("Could not access camera")
    #         return

    #     tasks = []

    #     try:
    #         while True:
    #             ret, frame = self.cap.read()
    #             if not ret:
    #                 print("Error: Could not read frame.")
    #                 break

    #             current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #             img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    #             if not os.path.exists(f"./images/{self.user_id}"):
    #                 os.makedirs(f"./images/{self.user_id}")
    #             img_path = f"./images/{self.user_id}/{current_time}.jpg"
    #             img.save(img_path)
    #             print(f"Captured and saved image: ./images/{current_time}.jpg")

    #             # img_caption = self.generate_caption(img)
    #             # img_caption = gen_description(img_path)

    #             task = asyncio.create_task(gen_description(img_path))
    #             task.add_done_callback(
    #                 lambda t: asyncio.create_task(
    #                     self.embed_and_upload(t.result(), current_time)
    #                 )
    #             )
    #             tasks.append(task)

    #             # Check for completed tasks and remove them from the list
    #             tasks = [t for t in tasks if not t.done()]

    #             await asyncio.sleep(time_interval)
    #             # time.sleep(time_interval)

    #     except KeyboardInterrupt:
    #         print("Stopped by user.")

    #     finally:
    #         self.cap.release()
    #         cv2.destroyAllWindows()

    async def embed_and_upload(self, img_caption, current_time):
        print(img_caption)
        img_caption_embedding = await self.text_to_embedding(img_caption)
        await self.db.upload(
            img_caption_embedding,
            user_id=self.user_id,
            embedding_id=str(current_time),
            img_caption=img_caption,
        )
        print("Image uploaded to DB")

    async def gen_answer(self, query):
        try:
            print("Generating Embedding")
            query_embedding = await self.text_to_embedding(query)
            query_embedding = query_embedding.tolist()
            # print(query_embedding)
            print("Searching DB")
            result = await self.db.search(query_embedding, user_id=self.user_id)
            # print(result)
            print("Data retrived from DB")

            context = ""
            for match in result.matches:
                context += match.metadata["img_caption"] + " "

            print(context)
            print("LLM generating response")
            # answer = await answer_query(query, context)
            answer = await self.llm.ask(query, context)
            return answer

        except Exception as e:
            print(f"Error in gen_answer: {e}")
            return None


async def main():
    lenswise = LensWise(4)
    await lenswise.capture_img(20)
    # print(await lenswise.gen_answer("What is the man wearing"))
    # print(await lenswise.gen_answer("What is the person doing"))


# if __name__ == "__main__":
#     asyncio.run(main())
    # cap.gen_answer("What is the color of my clothes?")
