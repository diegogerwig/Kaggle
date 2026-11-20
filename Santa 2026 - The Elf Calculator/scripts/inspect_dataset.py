import urllib.request
import re

url = 'https://www.kaggle.com/datasets/asteriospanoras/santa-2026-best-public-score-dataset-fails'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

req = urllib.request.Request(url, headers=headers)
html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
print('Dataset HTML length:', len(html))

files = set(re.findall(r'[\w\-./]+\.(?:csv|json|py|txt|cpp)', html))
print('Files mentioned:', files)

for m in re.finditer(r'\{[^{}]*"name"[^{}]*"totalBytes"[^{}]*\}', html):
    print('File meta:', m.group(0))

descs = re.findall(r'"description":\s*"([^"]+)"', html)
print('Descriptions:', descs[:5])
