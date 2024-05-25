import streamlit as st
from lensWise import LensWise
import os
from PIL import Image
import asyncio
import time


async def main():
    st.title("Lens Wise")
    user_id = "v1"

    if "started" not in st.session_state:
        st.session_state.started = False

    if "loading" not in st.session_state:
        st.session_state.loading = False

    if not st.session_state.started:
        st.session_state.lenswise = LensWise(user_id)
        st.session_state.started = True

    IMAGE_FOLDER = f"./images/{user_id}"
    image_files = [f for f in os.listdir(IMAGE_FOLDER)]

    with st.expander("Click to view Images"):
        cols = st.columns(3)  # Adjust the number of columns as needed
        for i, image_file in enumerate(image_files):
            img = Image.open(os.path.join(IMAGE_FOLDER, image_file))
            cols[i % 3].image(img, caption=image_file)

    st.title("Ask your query")
    query = st.text_input("Enter your question: ")
    if query and not st.session_state.loading:
        st.session_state.loading = True

        with st.spinner("Generating answer..."):
            start_time = time.time()
            answer = await st.session_state.lenswise.gen_answer(query)
            elapsed_time = time.time() - start_time
            # with st.expander("View context"):
            #     st.write(context)
            print(f"Time taken to answer query: {elapsed_time: .2f} seconds")
            st.success(answer)
        st.session_state.loading = False
        query = ""  


if __name__ == "__main__":
    asyncio.run(main())
