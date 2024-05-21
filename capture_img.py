import cv2
import time
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from sentence_transformers import SentenceTransformer
from vector_db import vector_db_class


class LensWise:
    def __init__(self, user_id=1):
        self.cap = cv2.VideoCapture(0)
        self.caption_model_name = "Salesforce/blip-image-captioning-base"
        self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            self.caption_model_name
        )

        self.vqa_model_name = "Salesforce/blip-vqa-base"
        self.vqa_processor = BlipProcessor.from_pretrained(self.vqa_model_name)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(self.vqa_model_name)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = vector_db_class()
        self.user_id = user_id

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.caption_processor(images=image, return_tensors="pt")
        outputs = self.caption_model.generate(**inputs)
        caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
        print(caption)
        return caption

    def text_to_embedding(self, text: str) -> np.ndarray:
        embedding = self.embedding_model.encode(text)
        return embedding

    def capture_img(self, time_interval: int = 30):

        if not self.cap.isOpened():
            print("Could not access camera")
            return

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                img.save(f"./images/{current_time}.jpg")
                print(f"Captured and saved image: ./images/{current_time}.jpg")

                img_caption = self.generate_caption(img)
                img_caption_embedding = self.text_to_embedding(img_caption)
                self.db.upload(
                    img_caption_embedding,
                    user_id=self.user_id,
                    embedding_id=str(current_time),
                )

                np.save(f"./embeddings/{current_time}", img_caption_embedding)

                # await asyncio.sleep(time_interval)
                time.sleep(time_interval)
        except KeyboardInterrupt:
            print("Stopped by user.")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
        
    def vqa(self, image_path: str, question: str):
        image = Image.open(image_path).convert("RGB")

        inputs = self.vqa_processor(text=question, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.vqa_model(**inputs)

        answer = self.vqa_processor.decode(outputs.logits[0], skip_special_tokens=True)

        return answer

    def gen_answer(self, query):
        try:
            query_embedding = self.text_to_embedding(query).tolist()
            result = self.db.search(query_embedding, user_id=self.user_id)
            print(result)
            img_path = f"./images/{result.matches[0].id}"
            answer = self.vqa(img_path, query)
            return answer
        except Exception as e:
            print(f"Error in gen_answer: {e}")
            return None


if __name__ == "__main__":
    cap = LensWise()
    cap.capture_img(2)
    # cap.gen_answer("What is the color of my clothes?")
