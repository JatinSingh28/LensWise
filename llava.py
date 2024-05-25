import replicate
from dotenv import load_dotenv
import time
import asyncio
import base64

load_dotenv()

async def gen_description(img_path: str):
    # start_time = time.time()
    print("Gen description called")

    with open(img_path, "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
        image = f"data:application/octet-stream;base64,{data}"
        input = {
            "image": image,
            "prompt": "Generate a detailed and precise description on given image. Transcribe any text you see.",
        }

        output = replicate.run(
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            input=input,
        )
        # elapsed_time = time.time() - start_time
        response = "".join(output)
        # print(response)
        # print(f"Time taken: {elapsed_time: .2f} seconds")

        return response


if __name__ == "__main__":
    # print(gen_description(img_path="./kids.png"))
    asyncio.run(gen_description(img_path="./kids.png"))
