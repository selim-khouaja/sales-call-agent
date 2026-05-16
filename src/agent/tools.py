import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_company",
            "description": "Look up company info (industry, size, description) by name using Clearbit Autocomplete",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "The company name to look up"}
                },
                "required": ["company_name"],
            },
        },
    }
]


def lookup_company(company_name: str) -> dict:
    """Fetch basic company info from Clearbit Autocomplete API."""
    try:
        resp = httpx.get(
            "https://autocomplete.clearbit.com/v1/companies/suggest",
            params={"query": company_name},
            timeout=5.0,
        )
        results = resp.json()
    except Exception:
        results = []

    if not results:
        return {"name": company_name, "domain": None, "description": None}

    top = results[0]
    return {
        "name": top.get("name", company_name),
        "domain": top.get("domain"),
        "description": top.get("description") or top.get("text"),
    }


TOOL_MAP = {"lookup_company": lookup_company}
