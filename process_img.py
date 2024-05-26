from lensWise import LensWise
import os
# from img_description import gen_description
from llava import gen_description
from vector_db import vector_db_class
import asyncio

db = vector_db_class()


async def process(user_id=5):
    lenswise = LensWise(user_id)
    img_path = f"./images/{user_id}"
    file_names = os.listdir(f"./images/{user_id}")
    print(file_names)

    for file_name in file_names:
        description = await gen_description(os.path.join(img_path, file_name))
        print(description)
        embedding = await lenswise.text_to_embedding(description)
        await db.upload(embedding, user_id, file_name, description)


if __name__ == "__main__":
    asyncio.run(process("v1"))
