from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
model.save("./models/bge-m3")