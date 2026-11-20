import urllib.request
import re
import json

url = 'https://www.kaggle.com/code/asteriospanoras/santa-2026-best-public-score-dataset-fails'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

req = urllib.request.Request(url, headers=headers)
html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')

with open('data/kaggle_notebook.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Saved HTML, length:', len(html))

# Look for embedded JSON state or notebook content
# Kaggle usually embeds bootstrap data in `<script class="kaggle-render-data">` or similar
render_data = re.findall(r'<script[^>]*class="[^"]*render-data[^"]*"[^>]*>(.*?)</script>', html, re.DOTALL)
print('Render data matches:', len(render_data))

# Also search for "body" or notebook JSON
for m in re.finditer(r'\{[^{}]*"title"[^{}]*"totalVotes"[^{}]*\}', html):
    print('Kernel meta snippet:', m.group(0)[:200])

# Search for dataset references
datasets = set(re.findall(r'asteriospanoras/[a-zA-Z0-9_\-]+', html))
print('Datasets/kernels referenced:', datasets)
