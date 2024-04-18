
from urllib import request, parse

def queue_prompt(prompt):
    p = f'{{"prompt": {prompt}}}'
    req =  request.Request("http://localhost:8188/prompt", data=p.encode('utf-8'), method='POST')
    request.urlopen(req).read()
