import re
import html
import emoji

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG = re.compile(r"#[A-Za-z0-9_]+")
PUNCT = re.compile(r"[^\w\s]")

def clean_text(t: str) -> str:
    t = html.unescape(str(t or ""))
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = HASHTAG.sub(" ", t)
    t = emoji.replace_emoji(t, replace=" ")
    t = PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

