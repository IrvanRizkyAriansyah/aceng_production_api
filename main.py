from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
import torch
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan domain yang sesuai jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_path = './metadata_kamera_augmented.json'
my_model_path = './my_models_2'
my_token_path = './my_token_models'

# Instansiasi model
model = BertForSequenceClassification.from_pretrained(my_model_path)
tokenizer = BertTokenizer.from_pretrained(my_token_path) #ini ganti dengan token yg sudah dibuat

# Tambahkan tokenizer ke model
model.resize_token_embeddings(len(tokenizer))

import json

# Load metadata list from the JSON file
with open(dataset_path, 'r') as file:
  my_dataset = json.load(file)

topics = my_dataset['Topics']
products = {product['Product_id']: product for product in my_dataset['Products']}

my_products = []
for topic in topics:
    for product_id in topic['Products_ID']:
        product_info = {
            "Product_id": product_id,
            "Product_name": products[str(product_id)]['Product_name'],
            "Product_brand": products[str(product_id)]['Product_brand'],
            "Product_images": products[str(product_id)]['Product_images'],
            "Product_price": products[str(product_id)]['Product_price'],
            "Topic_id": topic['Topic_ID'],
        }
        my_products.append(product_info)


def search_json(query: str) -> dict:
    # Tokenisasi query
    tokenized_query = tokenizer(query, return_tensors="pt")

    # Inferensi dengan model
    with torch.no_grad():
        outputs = model(**tokenized_query)

    # Ambil informasi produk berdasarkan label
    predicted_label = torch.argmax(outputs.logits).item()
    found_products = list(filter(lambda product: product.get('Topic_id') == predicted_label, my_products))

    # Hitung akurasi relatif
    accuracy = 0.0
    for topic in topics:
        if predicted_label == topic['Topic_ID']:
            accuracy = 1.0

    # Menyiapkan informasi produk yang ditemukan untuk respons
    products_info = []
    if found_products:
        for product in found_products:
            products_info.append({
                "Product ID": product['Product_id'],
                "Name": product['Product_name'],
                "Brand": product['Product_brand'],
                "Images": product['Product_images'],
                "Price": product['Product_price'],
            })

    result = {
        "status": 200,
        "messages": "Success",
        "query": query,
        "predicted_topic_id": predicted_label,
        "accuracy": accuracy,
        "data": products_info
    }


    return result

class QueryItem(BaseModel):
    query: str

@app.post("/search")
async def perform_search(query_item: QueryItem):
    try:
        # Memanggil fungsi search dan mengembalikan hasilnya
        result = search_json(query_item.query)
        return result
    except Exception as e:
        # Menangani kemungkinan error
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    try:
        # Kode yang mungkin menghasilkan kesalahan
        return {"Hello": "World"}
    except Exception as e:
        # Tangani kesalahan dengan HTTPException
        raise HTTPException(status_code=500, detail=str(e))
