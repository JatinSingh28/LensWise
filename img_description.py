import ollama
import asyncio
import time


def gen_description(img_path: str):
    with open(img_path, "rb") as file:
        print("Request sent to LLAVA")
        start_time = time.time()
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": "Generate a detailed and precise description on given image. Transcribe any text you see.",
                    "images": [file.read()],
                }
            ],
        )
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time: .2f} seconds")

    return response["message"]["content"]


if __name__ == "__main__":
    print(gen_description(img_path="./kids.png"))
