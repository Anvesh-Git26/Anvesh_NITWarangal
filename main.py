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
    logger.warning("GOOGLE_API_KEY not found!")
else:
    genai.configure(api_key=api_key)

app = FastAPI()

# Input Model (Matches submission format)
class InvoiceRequest(BaseModel):
    document: str

# --- 1. FRAUD ENGINE LOGIC (ELA) ---
def detect_fraud_advanced(image_path):
    # ELA (Error Level Analysis) checks for digital tampering
    try:
        orig = cv2.imread(image_path)
        if orig is None: return {"fraud_score": 0, "tampering_detected": False}
        
        cv2.imwrite("temp_ela.jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread("temp_ela.jpg")
        
        diff = cv2.absdiff(orig, compressed)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        flattened = diff.flatten()
        flattened.sort()
        top_5_percent = int(len(flattened) * 0.05)
        
        if top_5_percent > 0:
            robust_score = np.mean(flattened[-top_5_percent:])
        else:
            robust_score = 0
            
        # Tampering is detected if the robust score exceeds a threshold (15.0)
        return {"fraud_score": round(float(robust_score), 2), "tampering_detected": bool(robust_score > 15.0)}
    except Exception as e:
        logger.error(f"Fraud check failed: {e}")
        return {"fraud_score": 0, "tampering_detected": False}

# --- 2. LOGIC ENGINE LOGIC (Math Verification) ---
ITEM_TOLERANCE = 1.0

def verify_and_reconcile(extracted_data):
    # This logic handles item filtering, math verification, and total calculation
    lines = extracted_data.get("pagewise_line_items", [])
    total_count = 0
    calculated_total = 0.0
    
    # Helper to clean strings into floats
    def clean_money(val):
        if isinstance(val, (int, float)): return float(val)
        try:
            return float(str(val).replace(",", "").replace("Rs.", "").replace("$", "").strip())
        except:
            return 0.0
    
    for page in lines:
        valid_items = []
        for item in page.get("bill_items", []):
            try:
                q = clean_money(item.get("item_quantity", 0))
                r = clean_money(item.get("item_rate", 0))
                a = clean_money(item.get("item_amount", 0))
            except: q, r, a = 0.0, 0.0, 0.0

            # Trap 1: Lump Sum (Qty=0, Rate=0, Amount>0)
            if q == 0 and r == 0 and a > 0:
                q = 1.0
                r = a
            
            # Trap 2: Math Verification (Checks for normal error AND column swap error)
            if q > 0 and r > 0:
                math_normal = round(q * r, 2)
                if abs(math_normal - a) > ITEM_TOLERANCE:
                    # Check for Swap Trap (Qty * Amount == Rate)
                    if a > 0 and abs((q * a) - r) <= ITEM_TOLERANCE:
                        temp = r
                        r = a
                        a = temp
                    else:
                        a = math_normal # Trust Math
            
            # Update item with final reconciled values
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

# --- 3. ROBUST GEMINI CALLER (FINAL WORKING MODEL) ---
def call_gemini_safe(file_ref, prompt):
    # This list prioritizes the model confirmed to work in Colab (2.0-flash)
    models_to_try = [
        "gemini-2.0-flash",       # Primary Winner (Confirmed in user's environment)
        "gemini-2.5-flash",       # Secondary (Next generation)
        "gemini-flash-latest"     # Safe Fallback
    ]
    
    # Standard generation configuration for strict JSON output
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
    
    raise Exception("All models failed. Check API key/service enablement.")

# --- 4. API ENDPOINTS ---

@app.get("/")
def health_check():
    # Simple health check endpoint for Render
    return {"status": "online", "service": "BFHL Invoice Extractor"}

@app.post("/extract-bill-data")
async def extract_bill_data(request: InvoiceRequest):
    temp_filename = "temp_invoice.jpg"
    
    try:
        # A. Download (Added User-Agent for robustness against servers like HackRx blob storage)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.document, headers=headers, stream=True, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Download failed: Status {response.status_code}")
            
        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # B. Fraud Check
        fraud_data = detect_fraud_advanced(temp_filename)

        # C. Gemini Extraction
        file_ref = genai.upload_file(temp_filename)
        prompt = """
        Extract the invoice line items strictly.
        1. Classify the page_type as 'Bill Detail', 'Final Bill', or 'Pharmacy'.
        2. Extract item_name, item_quantity, item_rate, item_amount.
        3. If Quantity is missing but Amount exists, default Qty to 1.
        Ignore 'Total' rows at the bottom.
        """
        
        # Call the robust function
        gemini_resp = call_gemini_safe(file_ref, prompt)
        
        # D. Token Usage & Parsing
        try:
            raw_data = json.loads(gemini_resp.text)
        except:
            raw_data = {"pagewise_line_items": []}
            
        try:
            usage = gemini_resp.usage_metadata
            token_data = {"input_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count, "total_tokens": usage.total_token_count}
        except:
            token_data = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        # E. Logic Engine
        clean_data = verify_and_reconcile(raw_data)
        
        # F. Final Response Construction
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
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
