"""
demo/demo_queries.py
--------------------
Sample queries per domain to showcase the SME-PLUG capabilities.
"""

DEMO_QUERIES = {
    "structural_engineering": [
        {
            "query": "Is a W24x84 steel beam safe for a 500 kN uniformly distributed load on a 6m span?",
            "description": "Beam capacity check — structural analysis with AISC code compliance",
        },
        {
            "query": "What is the minimum reinforcement required for a 300mm x 600mm concrete beam per IS 456?",
            "description": "Reinforcement design — concrete code clause verification",
        },
        {
            "query": "Calculate the effective length factor for a fixed-pinned column and check for buckling.",
            "description": "Column stability — buckling analysis with code references",
        },
    ],
    "cybersecurity": [
        {
            "query": "Our SIEM detected multiple failed login attempts from IP 192.168.1.105 followed by a successful login and PowerShell execution. Triage this alert.",
            "description": "Incident triage — brute force + lateral movement detection",
        },
        {
            "query": "Map the following attack pattern to MITRE ATT&CK: attacker used spearphishing email, downloaded a malicious DLL, escalated privileges, and exfiltrated data via DNS tunneling.",
            "description": "MITRE ATT&CK mapping — full kill chain analysis",
        },
        {
            "query": "What are the recommended NIST 800-53 controls for preventing unauthorized remote access?",
            "description": "Compliance guidance — NIST control identification",
        },
    ],
    "legal": [
        {
            "query": "Review this NDA clause: 'The Receiving Party shall hold all Confidential Information in perpetuity and shall not disclose it under any circumstances.' Is this enforceable?",
            "description": "NDA review — enforceability analysis with jurisdiction awareness",
        },
        {
            "query": "A service agreement has an indemnification clause that is one-sided with no liability cap. What are the risks?",
            "description": "Contract risk assessment — indemnification clause analysis",
        },
        {
            "query": "Does a non-compete clause with a 5-year duration and worldwide scope hold up in California?",
            "description": "Non-compete enforceability — jurisdiction-specific analysis",
        },
    ],
}


def get_query(domain: str, index: int = 0) -> dict:
    """Get a specific demo query by domain and index."""
    return DEMO_QUERIES.get(domain, [{}])[index]


def get_all_domains() -> list[str]:
    """Return all domains with demo queries."""
    return list(DEMO_QUERIES.keys())
