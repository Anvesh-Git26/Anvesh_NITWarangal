import os
import uvicorn
import requests
import json
import logging
import cv2
import numpy as np
import re
import shutil
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
    logger.warning("GOOGLE_API_KEY not found in environment variables!")
else:
    genai.configure(api_key=api_key)

# Initialize App
app = FastAPI()

# Input Model
class InvoiceRequest(BaseModel):
    document: str

# --- 1. FRAUD ENGINE LOGIC (Inside main.py) ---
def detect_fraud_advanced(image_path):
    try:
        orig = cv2.imread(image_path)
        if orig is None: return {"fraud_score": 0, "tampering_detected": False}
        
        # ELA Simulation
        cv2.imwrite("temp_ela.jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread("temp_ela.jpg")
        
        diff = cv2.absdiff(orig, compressed)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        flattened = diff.flatten()
        flattened.sort()
        # Top 5% pixels
        top_5_percent = int(len(flattened) * 0.05)
        if top_5_percent > 0:
            robust_score = np.mean(flattened[-top_5_percent:])
        else:
            robust_score = 0
        
        return {
            "fraud_score": round(float(robust_score), 2),
            "tampering_detected": robust_score > 15.0
        }
    except Exception as e:
        logger.error(f"Fraud check failed: {e}")
        return {"fraud_score": 0, "tampering_detected": False}

# --- 2. LOGIC ENGINE LOGIC (Inside main.py) ---
ITEM_TOLERANCE = 1.0

def verify_and_reconcile(extracted_data):
    lines = extracted_data.get("pagewise_line_items", [])
    total_count = 0
    calculated_total = 0.0
    
    for page in lines:
        valid_items = []
        for item in page.get("bill_items", []):
            try:
                # Clean strings to floats
                q = float(str(item.get("item_quantity", 0)).replace(",", "").strip())
            except: q = 0.0
            try:
                r = float(str(item.get("item_rate", 0)).replace(",", "").strip())
            except: r = 0.0
            try:
                a = float(str(item.get("item_amount", 0)).replace(",", "").strip())
            except: a = 0.0

            # Trap 1: Lump Sum (Qty=0, Rate=0, Amount>0)
            if q == 0 and r == 0 and a > 0:
                q = 1
                r = a
            
            # Trap 2: Math Verification
            if q > 0 and r > 0:
                math_a = round(q * r, 2)
                if abs(math_a - a) > ITEM_TOLERANCE:
                    a = math_a  # Trust math
            
            # Update item
            item['item_quantity'] = q
            item['item_rate'] = r
            item['item_amount'] = a
            
            valid_items.append(item)
            calculated_total += a
            total_count += 1
            
        page['bill_items'] = valid_items

    return {
        "pagewise_line_items": lines,
        "total_item_count": total_count,
        "reconciled_amount": round(calculated_total, 2)
    }

# --- 3. API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "service": "BFHL Invoice Extractor"}

@app.post("/extract-bill-data")
async def extract_bill_data(request: InvoiceRequest):
    temp_filename = "temp_invoice.jpg"
    
    try:
        # A. Download
        logger.info(f"Downloading {request.document}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.document, headers=headers, stream=True, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")
            
        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # B. Fraud Check
        fraud_data = detect_fraud_advanced(temp_filename)
        
        # C. Gemini Extraction
        logger.info("Calling Gemini...")
        model = genai.GenerativeModel("gemini-1.5-pro", generation_config={
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
        })
        
        file_ref = genai.upload_file(temp_filename)
        prompt = """
        Extract the invoice line items strictly.
        1. Classify the page_type.
        2. Extract item_name, quantity, rate, amount.
        3. If Quantity is missing but Amount exists, default Qty to 1.
        Ignore 'Total' rows at the bottom.
        """
        gemini_resp = model.generate_content([file_ref, prompt])
        
        try:
            raw_data = json.loads(gemini_resp.text)
        except:
            raw_data = {"pagewise_line_items": []}
            
        # Token Usage
        try:
            usage = gemini_resp.usage_metadata
            token_data = {
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count
            }
        except:
            token_data = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # D. Logic Engine
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
        logger.error(f"Error: {e}")
        return {
            "is_success": False,
            "token_usage": {"total_tokens": 0},
            "data": {"pagewise_line_items": [], "total_item_count": 0, "reconciled_amount": 0.0}
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

