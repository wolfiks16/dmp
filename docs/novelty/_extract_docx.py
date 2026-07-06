# -*- coding: utf-8 -*-
"""Извлечь текст из .docx (zipfile + strip XML) в .txt (консоль не печатает UTF-8)."""
import html
import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "Двигатель", "Электродвигатель описание.docx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_motor_desc.txt")

with zipfile.ZipFile(SRC) as z:
    xml = z.read("word/document.xml").decode("utf-8", "replace")

# структура: ячейки | , строки/абзацы -> перевод строки
xml = xml.replace("</w:tc>", " | ")
xml = re.sub(r"</w:tr>", "\n", xml)
xml = re.sub(r"</w:p>", "\n", xml)
xml = xml.replace("<w:tab/>", "\t").replace("<w:br/>", "\n")
text = re.sub(r"<[^>]+>", "", xml)
text = html.unescape(text)
# почистить лишние пустые строки
lines = [ln.rstrip() for ln in text.splitlines()]
text = "\n".join(ln for ln in lines if ln.strip())

with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)

print("OK extracted")
print("chars:", len(text))
print("lines:", len(text.splitlines()))
