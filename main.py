import os
import uvicorn
import requests
import json
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Env Vars
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found!")
else:
    genai.configure(api_key=api_key)

app = FastAPI()

class InvoiceRequest(BaseModel):
    document: str

# --- LOGIC ENGINE (Pure Python - No Numpy) ---
ITEM_TOLERANCE = 1.0

def classify_row(description):
    desc = description.lower()
    if any(x in desc for x in ['discount', 'rebate', 'less', 'coupon', 'off']): return "DISCOUNT"
    if any(x in desc for x in ['tax', 'gst', 'vat', 'cgst', 'sgst', 'igst', 'cess']): return "TAX"
    if any(x in desc for x in ['round', 'adjustment', 'fee', 'charge', 'deposit', 'balance']): return "ADJUSTMENT"
    return "ITEM"

def clean_money(val):
    if isinstance(val, (int, float)): return float(val)
    try:
        return float(str(val).replace(",", "").replace("Rs.", "").replace("$", "").strip())
    except:
        return 0.0

def verify_and_reconcile(extracted_data):
    lines = extracted_data.get("pagewise_line_items", [])
    total_items_sum = 0.0
    total_tax_sum = 0.0
    total_discount_sum = 0.0
    total_adjustment_sum = 0.0
    total_item_count = 0
    
    for page in lines:
        valid_items = []
        for item in page.get("bill_items", []):
            try:
                desc = item.get("item_name", "")
                q = clean_money(item.get("item_quantity", 0))
                r = clean_money(item.get("item_rate", 0))
                a = clean_money(item.get("item_amount", 0))
            except: q, r, a = 0.0, 0.0, 0.0

            row_type = classify_row(desc)

            # Filter Headers
            if row_type == "ITEM" and q <= 1.0 and r == 0.0 and a > 100.0 and any(k in desc for k in ["charges", "services", "care", "particulars"]):
                continue

            if row_type == "ITEM":
                if q == 0 and r == 0 and a > 0: q, r = 1.0, a
                
                if q > 0 and r > 0:
                    math_normal = round(q * r, 2)
                    if abs(math_normal - a) > ITEM_TOLERANCE:
                        if a > 0 and abs((q * a) - r) <= ITEM_TOLERANCE:
                            temp = r; r = a; a = temp
                        else:
                            a = math_normal
                
                total_items_sum += a
                total_item_count += 1
            
            elif row_type == "TAX": total_tax_sum += a
            elif row_type == "DISCOUNT": total_discount_sum += abs(a)
            elif row_type == "ADJUSTMENT": total_adjustment_sum += a

            item['item_quantity'] = q
            item['item_rate'] = r
            item['item_amount'] = a
            valid_items.append(item)
            
        page['bill_items'] = valid_items

    final_reconciled_sum = (total_items_sum + total_tax_sum + total_adjustment_sum - total_discount_sum)

    return {
        "pagewise_line_items": lines,
        "total_item_count": total_item_count,
        "reconciled_amount": round(final_reconciled_sum, 2)
    }

# --- GEMINI CALLER ---
def call_gemini_safe(file_ref, prompt):
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest"]
    
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "pagewise_line_items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page_no": {"type": "STRING"},
                            "page_type": {"type": "STRING", "enum": ["Bill Detail", "Final Bill", "Pharmacy"]},
                            "bill_items": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "item_name": {"type": "STRING"},
                                        "item_quantity": {"type": "NUMBER"},
                                        "item_rate": {"type": "NUMBER"},
                                        "item_amount": {"type": "NUMBER"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for model_name in models_to_try:
        try:
            logger.info(f"Attempting model: {model_name}...")
            model = genai.GenerativeModel(model_name, generation_config=generation_config)
            response = model.generate_content([file_ref, prompt])
            return response 
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            continue 
    raise Exception("All models failed.")

# --- API ENDPOINTS ---
@app.get("/")
def health_check(): return {"status": "online", "service": "Simple BFHL Invoice Extractor"}

@app.post("/extract-bill-data")
async def extract_bill_data(request: InvoiceRequest):
    temp_filename = "temp_invoice.jpg"
    try:
        # Download with Browser Headers
        logger.info(f"Downloading {request.document}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
        response = requests.get(request.document, headers=headers, stream=True, timeout=20)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Download failed: Status {response.status_code}")
            
        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        
        # Upload to Gemini
        file_ref = genai.upload_file(temp_filename)
        
        prompt = """
        Extract the invoice line items strictly.
        1. Classify the page_type.
        2. Extract item_name, item_quantity, item_rate, item_amount.
        3. If Quantity is missing but Amount exists, default Qty to 1.
        Ignore 'Total' rows at the bottom.
        """
        
        gemini_resp = call_gemini_safe(file_ref, prompt)
        
        try: raw_data = json.loads(gemini_resp.text)
        except: raw_data = {"pagewise_line_items": []}
            
        try:
            usage = gemini_resp.usage_metadata
            token_data = {"total_tokens": usage.total_token_count, "input_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count}
        except:
            token_data = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        clean_data = verify_and_reconcile(raw_data)
        
        return {
            "is_success": True,
            "token_usage": token_data,
            "data": {
                "pagewise_line_items": clean_data["pagewise_line_items"],
                "total_item_count": clean_data["total_item_count"],
                "reconciled_amount": clean_data["reconciled_amount"]
            }
        }

    except Exception as e:
        logger.error(f"Final Error: {e}")
        return {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {"pagewise_line_items": [], "total_item_count": 0, "reconciled_amount": 0.0}
        }
    finally:
        if file_ref:
             try: genai.Client.files.delete(file_ref)
             except: pass
        if os.path.exists(temp_filename): os.remove(temp_filename)
    
