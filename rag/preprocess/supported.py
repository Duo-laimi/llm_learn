import re
import PyPDF2
import markdown
from bs4 import BeautifulSoup

from .register import register


@register.register(".pdf")
def read_pdf(path):
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
    return "".join(text)

@register.register(".md")
def read_markdown(path):
    with open(path, "r", encoding="utf-8") as file:
        md_text = file.read()
        html_text = markdown.markdown(md_text)
        soup = BeautifulSoup(html_text, "html.parser")
        plain_text = soup.get_text()
        text = re.sub(r"http\S+", "", plain_text)
    return text

@register.register(".txt")
def read_text(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text