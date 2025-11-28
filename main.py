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

# --- 2. LOGIC ENGINE LOGIC (Math Verification & Tax/Discount) ---

ITEM_TOLERANCE = 1.0

def classify_row(description):
    """Classifies a row into semantic types: ITEM, TAX, DISCOUNT, or ADJUSTMENT."""
    desc = description.lower()
    if any(x in desc for x in ['discount', 'rebate', 'less', 'coupon']):
        return "DISCOUNT"
    if any(x in desc for x in ['tax', 'gst', 'vat', 'cgst', 'sgst', 'igst', 'cess']):
        return "TAX"
    if any(x in desc for x in ['round', 'adjustment', 'fee', 'charge']):
        return "ADJUSTMENT"
    return "ITEM"

def verify_and_reconcile(extracted_data):
    lines = extracted_data.get("pagewise_line_items", [])
    
    # Financial Trackers
    total_items_sum = 0.0
    total_tax_sum = 0.0
    total_discount_sum = 0.0
    total_adjustment_sum = 0.0

    total_item_count = 0
    
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
                # Load values
                desc = item.get("item_name", "")
                q = clean_money(item.get("item_quantity", 0))
                r = clean_money(item.get("item_rate", 0))
                a = clean_money(item.get("item_amount", 0))
            except: q, r, a = 0.0, 0.0, 0.0

            row_type = classify_row(desc)

            # --- HEADER/TOTAL FILTER (Sample 1 Filter) ---
            if row_type == "ITEM" and q <= 1.0 and r == 0.0 and a > 100.0 and any(k in desc for k in ["charges", "services", "care", "particulars"]):
                # This is likely a category header. Skip.
                continue

            # --- ACCOUNTING & MATH LOGIC ---

            if row_type == "ITEM":
                # Trap 1: Lump Sum
                if q == 0 and r == 0 and a > 0:
                    q = 1.0
                    r = a
                
                # Trap 2: Math Verification (Corrects OCR mistakes in Amount)
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
                
                total_items_sum += a
                total_item_count += 1
            
            # Summing Components (For Final Reconciliation)
            elif row_type == "TAX":
                total_tax_sum += a
            elif row_type == "DISCOUNT":
                total_discount_sum += abs(a) # Subtract absolute value later
            elif row_type == "ADJUSTMENT":
                total_adjustment_sum += a

            # Update item with final reconciled values
            item['item_quantity'] = q
            item['item_rate'] = r
            item['item_amount'] = a
            item['row_type'] = row_type # Add debug info
            
            valid_items.append(item)
            
        page['bill_items'] = valid_items

    # --- FINAL RECONCILIATION ---
    final_reconciled_sum = (
        total_items_sum
        + total_tax_sum
        + total_adjustment_sum
        - total_discount_sum
    )

    return {
        "pagewise_line_items": lines,
        "total_item_count": total_item_count,
        "reconciled_amount": round(final_reconciled_sum, 2)
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
    return {"status": "online", "service": "BFHL Invoice Extractor"}

@app.post("/extract-bill-data")
async def extract_bill_data(request: InvoiceRequest):
    temp_filename = "temp_invoice.jpg"
    
    try:
        # A. Download (Added User-Agent for robustness against servers like HackRx blob storage)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'}
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

        # E. Logic Engine (Now includes Tax/Discount handling)
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
            

