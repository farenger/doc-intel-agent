# ============================================================
# DOC INTEL AGENT - Main File
# ============================================================

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import fitz
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

load_dotenv()

# ============================================================
# MODELS
# ============================================================

llm = ChatOpenAI(
    model="openrouter/auto",
    temperature=0,
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# ============================================================
# TOOLS
# ============================================================

@tool
def read_document_tool(file_path: str) -> str:
    """Read a document from file path. Handles both digital and scanned PDFs.
    Returns JSON string with content and source type."""
    
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ""
        source = "digital"
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text().strip()
            if page_text:
                text += page_text
            else:
                source = "ocr"
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.medianBlur(img, 3)
                img = cv2.adaptiveThreshold(img, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2)
                pil_image = Image.fromarray(img).convert("RGB")
                pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = ocr_model.generate(pixel_values)
                ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text += ocr_text + "\n"
        
        doc.close()
    else:
        with open(file_path, 'r') as f:
            text = f.read()
        source = "digital"
    
    return json.dumps({"content": text, "source": source})

@tool
def classify_document_tool(content: str) -> str:
    """Classify document into: invoice, bank_statement, form, or unknown."""
    
    response = llm.invoke(f"""
Classify this document into exactly one of these categories:
- invoice
- bank_statement
- form
- unknown

Document:
{content[:300]}

Return ONLY the category name, nothing else.
""")
    return response.content.strip()

@tool
def extract_fields_tool(content: str) -> str:
    """Extract all relevant fields from document content.
    Returns JSON string with extracted fields."""
    
    response = llm.invoke(f"""
You are a financial document parser.
Rules:
- Do NOT guess values
- If unsure, return null
- Extract only explicitly present values

Look at this document and extract ALL relevant fields.
Return a JSON object with field names and values.

Document:
{content}

Return ONLY valid JSON. No markdown, no explanation.
""")
    clean = response.content.strip().replace("```json","").replace("```","").strip()
    return clean

@tool
def validate_document_tool(extracted_json: str, source: str) -> str:
    """Validate extracted fields and add review flags.
    Returns JSON string with validation results."""
    
    result = json.loads(extracted_json)
    errors = []
    warnings = []
    
    # Check numeric fields
    amount_fields = ["total_amount", "closing_balance", "amount", "total", "balance"]
    for field in amount_fields:
        if result.get(field):
            amount = str(result[field]).replace(",","").replace("Rs","").strip()
            if not amount.replace(".","").isdigit():
                errors.append(f"{field} is not a valid number")
    
    # Check vendor
    if not result.get("vendor") and not result.get("bank_name") and not result.get("company_name"):
        warnings.append("No vendor or bank name found")
    
    # Add source flags
    result["source"] = source
    result["requires_human_review"] = source == "ocr"
    if source == "ocr":
        result["review_reason"] = "Handwritten/scanned document - amounts need human verification"
    
    # Add validation status
    if errors:
        result["validation_status"] = "failed"
        result["errors"] = errors
    elif warnings:
        result["validation_status"] = "passed_with_warnings"
        result["warnings"] = warnings
    else:
        result["validation_status"] = "passed"
    
    return json.dumps(result)

@tool
def check_consistency_tool(extracted_json: str) -> str:
    """Check if document totals are consistent with line items.
    Returns JSON string with consistency check results."""
    
    result = json.loads(extracted_json)
    
    # Check if line items sum matches total
    if result.get("description") or result.get("items") or result.get("transactions"):
        items = result.get("description") or result.get("items") or result.get("transactions")
        
        try:
            amounts = []
            for item in items:
                if isinstance(item, dict):
                    amount = item.get("amount") or item.get("debit") or 0
                    if amount:
                        clean = str(amount).replace(",","").replace("Rs","").strip()
                        if clean.replace(".","").isdigit():
                            amounts.append(float(clean))
            
            if amounts and result.get("total_amount"):
                calculated_total = sum(amounts)
                stated_total = float(str(result["total_amount"]).replace(",","").replace("Rs","").strip())
                
                if abs(calculated_total - stated_total) > 1:
                    result["consistency_check"] = "failed"
                    result["consistency_note"] = f"Line items sum ({calculated_total}) doesn't match total ({stated_total})"
                else:
                    result["consistency_check"] = "passed"
            else:
                result["consistency_check"] = "skipped"
                
        except:
            result["consistency_check"] = "skipped"
    else:
        result["consistency_check"] = "skipped"
    
    return json.dumps(result, indent=2)

# ============================================================
# AGENT
# ============================================================

tools = [
    read_document_tool,
    classify_document_tool,
    extract_fields_tool,
    validate_document_tool,
    check_consistency_tool
]

agent = create_react_agent(
    llm,
    tools,
    prompt="""You are an intelligent document processing agent.

When given a file path, follow these steps IN ORDER:
1. Call read_document_tool with the file path
2. Call classify_document_tool with the content from step 1
3. Call extract_fields_tool with the content from step 1
4. Call validate_document_tool with the extracted JSON from step 3 AND the source from step 1
5. Call check_consistency_tool with the validated JSON from step 4
6. Return ONLY the raw JSON from step 5. No text, no explanation.
7. Final JSON Should Have all the checks status as given in the below json for example.

{"consistency_check":"passed","customer":"Anupam Manas","date":"20th April 2025","due_date":"30th April 2025"
,"invoice_number":"1234",
"items":[{"amount":"Rs 65,000","name":"Laptop"},{"amount":"Rs 1,500","name":"Mouse"},
{"amount":"Rs 2,000","name":"Keyboard"}],
"payment_status":"Unpaid","requires_human_review":false,"source":"digital",
"total_amount":"Rs 68,500","validation_status":"passed","vendor":"ABC Electronics"}


""")

# ============================================================
# RUN
# ============================================================
