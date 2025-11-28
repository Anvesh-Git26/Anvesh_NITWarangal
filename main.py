import os
import uvicorn
import requests
import json
import logging
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Custom modules
from fraud_engine import detect_fraud_advanced
from logic_engine import verify_and_reconcile

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Env Vars
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize App
app = FastAPI()

# Input Model
class InvoiceRequest(BaseModel):
    document: str

@app.get("/")
def health_check():
    return {"status": "online", "service": "BFHL Invoice Extractor"}

@app.post("/extract-bill-data")
async def extract_bill_data(request: InvoiceRequest):
    temp_filename = "temp_invoice.jpg"
    
    try:
        # 1. Download File from URL
        logger.info(f"Downloading from {request.document}")
        # Add headers to avoid 403 blocks from some servers
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(request.document, headers=headers, stream=True, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")
            
        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # 2. Fraud Check (Forensic Layer)
        logger.info("Running Fraud Check...")
        try:
            fraud_data = detect_fraud_advanced(temp_filename)
        except:
            fraud_data = {"fraud_score": 0, "tampering_detected": False}
        
        # 3. Gemini Extraction (Visual Layer)
        logger.info("Sending to Gemini...")
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={
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
        1. Classify the page_type as 'Bill Detail', 'Final Bill', or 'Pharmacy'.
        2. Extract item_name, item_quantity, item_rate, item_amount.
        3. If Quantity is missing but Amount exists, default Quantity to 1.
        Ignore 'Total' rows at the bottom.
        """
        gemini_resp = model.generate_content([file_ref, prompt])
        
        # Robust JSON Parsing
        try:
            raw_data = json.loads(gemini_resp.text)
        except:
            raw_data = {"pagewise_line_items": []}
            
        # Extract Token Usage (NEW REQUIREMENT)
        # Handle cases where usage metadata might be missing
        try:
            usage = gemini_resp.usage_metadata
            token_data = {
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count
            }
        except:
            token_data = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        
        # 4. Logic Engine (Math Verification Layer)
        logger.info("Running Logic Engine...")
        # The logic engine passes extra fields (like page_type) through automatically
        clean_data = verify_and_reconcile(raw_data)
        
        # 5. Final Response (Strictly matching the NEW Schema)
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
        logger.error(f"Error: {str(e)}")
        return {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0,
                "reconciled_amount": 0.0
            }
        }
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
