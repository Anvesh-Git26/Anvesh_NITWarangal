import os
import json
import tempfile
import mimetypes
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai


# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="Datathon Bill Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request model
# -----------------------------
class InvoiceRequest(BaseModel):
    document: Any


# -----------------------------
# Extract URL helper
# -----------------------------
def extract_url(document: Any) -> str:
    if isinstance(document, str) and document.startswith(("http://", "https://")):
        return document

    if isinstance(document, dict):
        for k in ("document", "url", "link", "href"):
            v = document.get(k)
            if isinstance(v, str) and v.startswith(("http://", "https://")):
                return v
        for v in document.values():
            if isinstance(v, (dict, list, str)):
                try:
                    return extract_url(v)
                except Exception:
                    pass

    if isinstance(document, list):
        for v in document:
            try:
                return extract_url(v)
            except Exception:
                pass

    raise HTTPException(status_code=400, detail="No valid URL in document")


# -----------------------------
# Download file helper
# -----------------------------
def download_file(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Download failed: {r.status_code}")

    content_type = r.headers.get("Content-Type", "").split(";")[0].lower()

    if content_type == "application/pdf":
        suffix = ".pdf"
    elif "png" in content_type:
        suffix = ".png"
    elif "jpeg" in content_type or "jpg" in content_type:
        suffix = ".jpg"
    else:
        suffix = mimetypes.guess_extension(content_type) or ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in r.iter_content(8192):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return tmp.name


# -----------------------------
# Normalize item floats
# -----------------------------
def f(x):
    try:
        if x is None:
            return 0.0
        return float(str(x).replace(",", "").strip())
    except:
        return 0.0


# -----------------------------
# Gemini setup - Gemini 2.0 Flash
# -----------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set!")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={
        "temperature": 0.0,
        "max_output_tokens": 3000,
        "response_mime_type": "application/json",
    }
)


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "up"}


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/extract-bill-data")
def extract_bill_data(req: InvoiceRequest):
    filename = None
    try:
        url = extract_url(req.document)
        filename = download_file(url)

        file_ref = genai.upload_file(filename)

        prompt = """
Extract bill items exactly in this format:
{
  "pagewise_line_items": [
    {
      "page_number": 1,
      "items": [
        {
          "item_name": "string",
          "item_amount": number,
          "item_rate": number,
          "item_quantity": number
        }
      ]
    }
  ]
}
Rules:
- Only chargeable line items
- No totals / subtotals / taxes
- If quantity missing → 1
- If only amount shown → quantity=1, rate=amount
Return valid JSON only.
"""

        resp = model.generate_content([file_ref, prompt])
        data = json.loads(resp.text)
        pages_raw = data.get("pagewise_line_items", [])

        # Convert to required Datathon schema
        pages_final = []
        total_items = 0

        for p in pages_raw:
            pn = str(p.get("page_number", 1))
            items = p.get("items", [])
            clean_items = []

            for it in items:
                clean_items.append({
                    "item_name": it.get("item_name", "").strip(),
                    "item_amount": f(it.get("item_amount")),
                    "item_rate": f(it.get("item_rate")),
                    "item_quantity": f(it.get("item_quantity")),
                })
            total_items += len(clean_items)

            pages_final.append({
                "page_no": pn,
                "page_type": "Bill Detail",
                "bill_items": clean_items
            })

        usage = getattr(resp, "usage_metadata", None)
        token_usage = {
            "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
            "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0
        }

        return {
            "is_success": True,
            "token_usage": token_usage,
            "data": {
                "pagewise_line_items": pages_final,
                "total_item_count": total_items,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
