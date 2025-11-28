import re

ITEM_TOLERANCE = 1.0

def clean_money(val):
    if isinstance(val, (int, float)): return float(val)
    try:
        clean = str(val).replace(",", "").replace("Rs.", "").replace("$", "").strip()
        return float(clean) if clean else 0.0
    except:
        return 0.0

def is_header_or_total(desc, qty, rate, amount):
    desc = desc.lower()
    if any(x in desc for x in ["total", "sum", "gross", "net", "amount in words"]): return True
    
    # Heuristic for Category Headers
    if qty <= 1 and rate == 0 and amount > 100:
        if any(x in desc for x in ["charges", "services", "care", "fee", "particulars"]):
            return True
    return False

def verify_and_reconcile(extracted_data):
    lines = extracted_data.get("pagewise_line_items", [])
    
    total_count = 0
    calculated_total = 0.0
    
    for page in lines:
        valid_items = []
        for item in page.get("bill_items", []):
            desc = item.get("item_name", "")
            q = clean_money(item.get("item_quantity", 0))
            r = clean_money(item.get("item_rate", 0))
            a = clean_money(item.get("item_amount", 0))
            
            # --- TRAP FIX 1: Headers & Totals ---
            if is_header_or_total(desc, q, r, a):
                continue

            # --- TRAP FIX 2: Column Swap ---
            if q > 0 and r > 0 and a > 0:
                math_normal = round(q * r, 2)
                if abs(math_normal - a) > ITEM_TOLERANCE:
                    if abs((q * a) - r) <= ITEM_TOLERANCE:
                        temp = r
                        r = a
                        a = temp

            # --- LOGIC & MATH RECONCILIATION ---
            if q > 0 and r > 0:
                math_a = round(q * r, 2)
                if abs(math_a - a) > ITEM_TOLERANCE:
                    a = math_a 
            elif q > 0 and r == 0 and a > 0:
                r = round(a / q, 2)
            elif q == 0 and r == 0 and a > 0:
                q = 1
                r = a
            
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
