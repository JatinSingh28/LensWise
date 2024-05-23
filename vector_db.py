from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()


class vector_db_class:
    def __init__(self) -> None:
        self.pc = Pinecone(api_key=os.getenv("vec_db_key"))
        self.index = self.pc.Index("lens-wise-bert")

    async def upload(self, embeddings, user_id, embedding_id, img_caption):
        try:
            await self.index.upsert(
                vectors=[
                    {
                        "id": embedding_id,
                        "values": embeddings,
                        "metadata": {
                            "user_id": str(user_id),
                            "img_caption": img_caption,
                        },
                    }
                ]
            )
            return True
        except Exception as e:
            print(f"Error uploading embedding: {e}")
            return False

    async def search(self, embedding, user_id):
        try:
            results = await self.index.query(
                vector=embedding,
                top_k=5,
                include_metadata=True,
                filter={"user_id": str(user_id)},
            )
            return results
        except Exception as e:
            print(f"Error searching for embedding: {e}")
            return None
