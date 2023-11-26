import numpy as np
import pandas as pd
import streamlit as st
import datetime
import json
from pyvis.network import Network
from streamlit_markmap import markmap

def styled_text(text, background_color='yellow'):
    # span 태그를 사용하여 배경색 적용
    html_str = f"<span style='background-color:{background_color};'>{text}</span>"
    st.markdown(html_str, unsafe_allow_html=True)

def styled_hashtag(text, bg_color='#0d6efd', text_color='white'):
    # 스타일링된 해시태그를 위한 HTML 코드
    html_str = f"""
    <span style='
        background-color:{bg_color};
        color: {text_color};
        padding: 0.2em 0.5em;
        border-radius: 0.3em;
        font-size: 1em;
        margin: 0.1em;'>#{text}</span>
    """
    st.markdown(html_str, unsafe_allow_html=True)

def render_markdown(data, level=0):
    md_string = ""
    for key, value in data.items():
        md_string += '    ' * level + f"- **{key}**\n"
        if isinstance(value, list):
            for item in value:
                for item_key in item:
                    md_string += '    ' * (level + 1) + f"- {item_key}\n"
        else:
            md_string += render_markdown(value, level+1)
    return md_string

def chunks(lst, num):
    """Yield successive num-sized chunks from lst."""
    for i in range(0, len(lst), num):
        yield lst[i:i + num]



    

