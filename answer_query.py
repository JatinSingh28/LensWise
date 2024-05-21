import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from sentence_transformers import SentenceTransformer
from vector_db import vector_db_class


class answer_query_class:
    def __init__(self, user_id=1):
        self.vqa_model_name = "Salesforce/blip-vqa-base"
        self.vqa_processor = BlipProcessor.from_pretrained(self.vqa_model_name)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(self.vqa_model_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = vector_db_class()
        self.user_id = user_id

    def query(self, query: str):
        embedding = self.embedding_model(query, convert_to_tensor=True).tolist()
        result = db.search(embedding)

    def vqa_gen_answer(self, image_path: str, question: str) -> str:
        image = Image.open(image_path).convert("RGB")

        inputs = self.vqa_processor(text=question, images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.vqa_model(**inputs)

        answer = self.vqa_processor.decode(outputs.logits[0], skip_special_tokens=True)

        return answer


if __name__ == "__main__":
    vqa = answer_query_class()
    image_path = ""
    question = ""
    answer = vqa.vqa_gen_answer(image_path, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
