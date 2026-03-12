from flask import Flask, request, jsonify
import pandas as pd
import re
from functools import lru_cache

app = Flask(__name__)

# ─────────────────────────────────────────
# LOAD ONCE AT STARTUP
# ─────────────────────────────────────────

print("Loading dataset...")
df = pd.read_excel("gov_schemes_full_with_age_gender_caste.xlsx")
df.columns = df.columns.str.strip().str.lower()
df["combined_text"] = df.astype(str).agg(" ".join, axis=1).str.lower()
SCHEMES = df.to_dict("records")
print(f"Ready. {len(SCHEMES)} schemes loaded.")


# ─────────────────────────────────────────
# ALIAS MAPS
# ─────────────────────────────────────────

GENDER_ALIASES = {
    "female": ["female", "girl", "woman", "women", "lady", "mahila", "she", "her"],
    "male":   ["male", "boy", "man", "men", "he", "him", "gents", "purush"],
}

CASTE_ALIASES = {
    "sc":      ["sc", "scheduled caste", "dalit", "harijan"],
    "st":      ["st", "scheduled tribe", "tribal", "adivasi"],
    "obc":     ["obc", "other backward", "backward class", "bc"],
    "general": ["general", "open", "unreserved", "ur", "forward"],
    "ews":     ["ews", "economically weaker"],
    "minority":["minority", "muslim", "christian", "sikh", "buddhist"],
}

OCCUPATION_ALIASES = {
    "student":     ["student", "studying", "school", "college", "university", "education", "scholarship"],
    "farmer":      ["farmer", "farming", "agriculture", "kisan", "crop"],
    "worker":      ["worker", "labour", "labor", "mazdoor", "daily wage", "construction"],
    "disabled":    ["disabled", "divyang", "handicapped", "pwd"],
    "widow":       ["widow", "widowed", "single mother"],
    "pregnant":    ["pregnant", "pregnancy", "maternity"],
    "senior":      ["senior", "elderly", "old age", "retired", "pension"],
    "unemployed":  ["unemployed", "jobless", "no job"],
    "entrepreneur":["entrepreneur", "startup", "business", "self employed", "msme"],
    "bpl":         ["bpl", "below poverty", "poor", "low income", "garib"],
}

INTENT_MAP = {
    "scholarship": ["scholarship", "study", "tuition"],
    "health":      ["health", "medical", "hospital", "medicine", "treatment"],
    "housing":     ["house", "home", "shelter", "awas", "housing"],
    "loan":        ["loan", "credit", "finance", "mudra"],
    "pension":     ["pension", "retirement"],
    "employment":  ["job", "employment", "skill", "training", "rozgar"],
    "food":        ["food", "ration", "nutrition", "subsidy"],
    "agriculture": ["agriculture", "farming", "kisan", "crop"],
}

# Flat list of all keywords with weights — built once at startup
ALL_KEYWORDS: list[tuple[str, int]] = []
for aliases in OCCUPATION_ALIASES.values():
    for a in aliases:
        ALL_KEYWORDS.append((a, 3))
for keywords in INTENT_MAP.values():
    for k in keywords:
        ALL_KEYWORDS.append((k, 2))


# ─────────────────────────────────────────
# AGE
# ─────────────────────────────────────────

AGE_PATTERNS = [
    r'\b(?:i\s+am|i\'m|age\s+is|aged?|my\s+age)\s*(\d{1,2})\b',
    r'\b(\d{1,2})\s*(?:years?(?:\s+old)?|yr\.?s?)\b',
    r'\b(\d{1,2})\b',
]

def extract_age(msg):
    for p in AGE_PATTERNS:
        m = re.search(p, msg, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 5 <= v <= 100:
                return v
    return None

def age_match(age_range, user_age):
    r = str(age_range).strip().lower()
    if r in ("", "nan", "all", "any", "no limit", "none"):
        return True
    if "+" in r:
        try: return user_age >= int(r.replace("+","").strip())
        except: return True
    if "-" in r:
        parts = r.split("-")
        try: return user_age <= int(parts[1].strip())
        except: return True
    try: return user_age <= int(r)
    except: return True


# ─────────────────────────────────────────
# EXTRACT USER INFO  (cached)
# ─────────────────────────────────────────

@lru_cache(maxsize=512)
def extract_user_info(message: str):
    msg = message.lower()
    info = {"age": extract_age(msg), "gender": None, "caste": None,
            "occupation": [], "intents": [], "keywords": []}

    for gender, aliases in GENDER_ALIASES.items():
        if any(re.search(rf'\b{re.escape(a)}\b', msg) for a in aliases):
            info["gender"] = gender
            break

    for caste, aliases in CASTE_ALIASES.items():
        if any(re.search(rf'\b{re.escape(a)}\b', msg) for a in aliases):
            info["caste"] = caste
            break

    for occ, aliases in OCCUPATION_ALIASES.items():
        if any(re.search(rf'\b{re.escape(a)}\b', msg) for a in aliases):
            info["occupation"].append(occ)

    for intent, kws in INTENT_MAP.items():
        if any(re.search(rf'\b{re.escape(k)}\b', msg) for k in kws):
            info["intents"].append(intent)

    # Simple word extraction — no spaCy, no overhead
    words = re.findall(r'\b[a-z]{3,}\b', msg)
    stopwords = {"the","and","for","are","you","give","want","show","need",
                 "please","get","me","my","am","is","in","of","to","a","i",
                 "have","has","what","how","can","will","this","that","with"}
    info["keywords"] = [w for w in words if w not in stopwords]

    return info


# ─────────────────────────────────────────
# SCORE  (pure string contains — very fast)
# ─────────────────────────────────────────

def score_scheme(text: str, info: dict) -> float:
    score = 0.0
    # Keyword weights
    for kw, weight in ALL_KEYWORDS:
        if kw in text:
            score += weight
    # Raw user words
    for w in info["keywords"]:
        if w in text:
            score += 1
    return score


# ─────────────────────────────────────────
# CHAT ENDPOINT
# ─────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Send JSON with a 'message' key."}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"reply": "Please enter a message.", "schemes": []}), 400

    info = extract_user_info(message.lower())

    # Score all schemes
    scored = sorted(
        ((score_scheme(s["combined_text"], info), s) for s in SCHEMES),
        key=lambda x: x[0],
        reverse=True
    )

    # Use top scored or fallback to all
    candidates = [s for sc, s in scored if sc > 0] or SCHEMES

    # Apply filters
    filtered = []
    for s in candidates:
        if info["age"] is not None:
            if not age_match(s.get("age_requirement", "all"), info["age"]):
                continue
        if info["gender"]:
            req = str(s.get("gender_requirement", "all")).lower()
            if req not in ("all","any","nan","") and info["gender"] not in req:
                continue
        if info["caste"]:
            req = str(s.get("caste_community_requirement", "all")).lower()
            if req not in ("all","any","nan",""):
                if not any(a in req for a in CASTE_ALIASES.get(info["caste"], [info["caste"]])):
                    continue
        filtered.append(s)
        if len(filtered) == 5:
            break

    if not filtered:
        return jsonify({
            "reply": f"No schemes found. I understood: {info_summary(info)}",
            "schemes": [], "understood": info_summary(info)
        })

    return jsonify({
        "reply": f"Found {len(filtered)} scheme(s) for you.",
        "understood": info_summary(info),
        "schemes": [{
            "name":               s.get("name", "N/A"),
            "category":           s.get("scheme_category", "N/A"),
            "age_requirement":    s.get("age_requirement", "All"),
            "gender_requirement": s.get("gender_requirement", "All"),
            "caste_requirement":  s.get("caste_community_requirement", "All"),
            "benefits":           s.get("benefits", "N/A"),
        } for s in filtered]
    })


def info_summary(info):
    parts = []
    if info["age"]:        parts.append(f"Age: {info['age']}")
    if info["gender"]:     parts.append(f"Gender: {info['gender']}")
    if info["caste"]:      parts.append(f"Caste: {info['caste'].upper()}")
    if info["occupation"]: parts.append(f"Occupation: {', '.join(info['occupation'])}")
    if info["intents"]:    parts.append(f"Looking for: {', '.join(info['intents'])}")
    return " | ".join(parts) if parts else "General query"


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=False)