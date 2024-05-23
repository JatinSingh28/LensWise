import ollama

def answer_query(query: str, context: str):
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
    response = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
