import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

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
    # total_topic_products = sum(1 for product in my_products if product.get('Topic_id') == predicted_label)
    # accuracy = len(found_products) / total_topic_products if total_topic_products > 0 else 0.0

    # Menyiapkan informasi produk yang ditemukan untuk respons
    products_info = []
    if found_products:
        for product in found_products:
            products_info.append({
                "Product_id": product['Product_id'],
                "Product_name": product['Product_name'],
                "Product_brand": product['Product_brand'],
                "Product_images": product['Product_images'],
                "Product_price": product['Product_price'],
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

class QueryInput(BaseModel):
    query: str

def find_product(search_query):
    result = []

    # Asumsi bahwa my_dataset['Products'] adalah sebuah list dari dictionary
    for product in my_dataset['Products']:
        # Periksa apakah 'Product_name' ada dan cocok dengan query pencarian
        if search_query.lower() in product.get('Product_name', '').lower():
            # Jika cocok, tambahkan seluruh objek produk ke dalam hasil
            result_product = {
                'Product_id': product.get('Product_id', ''),  # Gunakan nilai default '' jika kunci tidak ditemukan
                'Product_name': product.get('Product_name', ''),
                'Product_brand': product.get('Product_brand', ''),
                'Product_images': product.get('Product_images', ''),
                'Product_price': product.get('Product_price', '')
            }
            result.append(result_product)
    
    results = {
        "status": 200,
        "messages": "Success",
        "data": result
    }

    return results

def get_random_products():
    # Pastikan dataset produk Anda sudah terdefinisi sebagai my_dataset
    products = my_dataset['Products']

    # Mengacak urutan dari produk-produk tersebut
    random.shuffle(products)

    # Mengambil 20 produk pertama dari list yang sudah diacak
    random_products = products[:20]

    results = {
        "status": 200,
        "messages": "Success",
        "data": random_products
    }

    return results

class QueryItem(BaseModel):
    query: str

# Endpoint untuk mendapatkan produk acak
@app.get("/product")
def random_products():
    try:
        # Panggil fungsi get_random_products dan kembalikan hasilnya
        result = get_random_products()
        return result
    except Exception as e:
        # Tangani kemungkinan kesalahan
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint untuk melakukan pencarian produk
@app.post("/product/search_product")
async def perform_search(query_item: QueryItem):
    try:
        # Panggil fungsi find_product dan kembalikan hasilnya
        result = find_product(query_item.query)
        return result
    except Exception as e:
        # Tangani kemungkinan kesalahan
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/product/search")
async def search(query_item: QueryItem):
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)