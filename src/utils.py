"""
utils.py

Helper functions for highlighting and error handling.
"""

import re
import streamlit as st

def highlight_text(text, substrings):
    """
    Returns HTML with substrings highlighted. (Case-insensitive)
    """
    if not substrings:
        return text
    pattern = "|".join(map(re.escape, [s.strip() for s in substrings if s]))
    if not pattern:
        return text
    return re.sub(
        f"({pattern})",
        r'<mark style="background-color: #ffe066">\1</mark>',
        text,
        flags=re.IGNORECASE
    )

def show_error(msg):
    st.error(msg)
