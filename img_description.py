import ollama

def gen_description(img_path: str):
    with open(img_path, "rb") as file:
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
    return response["message"]["content"]


if __name__ == "__main__":
    print(gen_description(img_path="./kids.png"))
