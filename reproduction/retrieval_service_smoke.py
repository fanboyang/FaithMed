"""CPU MiniCPM exact retrieval over an explicitly bounded five-source smoke corpus.

This service is for integration tests only: it does not reproduce the paper's
full-corpus sharded DiskANN retrieval. Model pooling follows the pinned OpenBMB
MiniCPM-Embedding-Light implementation (adapt_mean_pooling=True).
"""
import argparse
import base64
import json
from pathlib import Path
import threading
import numpy as np

SCOPE='integration_smoke_subset_exact_search'


def passage_text(document):
    text=document['content']
    return text if document['data_source'].lower()=='pubmed' else document['title']+'\n'+text


def ranked_results(documents,embeddings,query_embedding,k):
    if type(k) is not int or not 1<=k<=len(documents):
        raise ValueError('k must be within the corpus size')
    if embeddings.ndim!=2 or embeddings.shape[0]!=len(documents) or query_embedding.shape!=(embeddings.shape[1],):
        raise ValueError('embedding shape mismatch')
    if not np.isfinite(embeddings).all() or not np.isfinite(query_embedding).all():
        raise ValueError('nonfinite embeddings')
    scores=embeddings@query_embedding
    order=np.argsort(-scores,kind='stable')[:k]
    results=[]
    for i in order:
        d=documents[i]
        payload={'id':d['id'],'title':d['title'],'Clean-Text':passage_text(d),
                 'data_source':d['data_source'],'source_url':d['source_url']}
        results.append(base64.b64encode(json.dumps(payload,ensure_ascii=False).encode()).decode())
    return {'results':results,'scores':[float(scores[i]) for i in order],'scope':SCOPE}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--port',type=int,default=18601)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel,AutoTokenizer
    from fastapi import FastAPI,HTTPException
    import uvicorn
    torch.set_num_threads(args.threads)
    documents=[json.loads(line) for line in args.corpus.read_text().splitlines()]
    sources={d['data_source'].lower() for d in documents}
    if len(sources)!=5 or not 6<=len(documents)<=100:
        raise ValueError('smoke corpus must contain five sources and 6-100 documents')
    for d in documents:
        if any(not isinstance(d.get(key),str) or not d[key].strip() for key in ('id','title','content','data_source','source_url')):
            raise ValueError('incomplete document metadata')
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    model=AutoModel.from_pretrained(args.model,trust_remote_code=True,local_files_only=True,
                                    torch_dtype=torch.float32).eval()
    def encode(texts,query=False):
        texts=['Query: '+text for text in texts] if query else texts
        batch=tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
        with torch.inference_mode():
            outputs=model(**batch,return_dict=True,adapt_mean_pooling=True)
            mask=batch['attention_mask'].unsqueeze(-1)
            reps=(outputs.last_hidden_state*mask).sum(1)/mask.sum(1)
            reps=F.normalize(reps,p=2,dim=-1)
        if reps.shape[1]!=1024 or not torch.isfinite(reps).all():
            raise ValueError('invalid MiniCPM embeddings')
        return reps.numpy()
    embeddings=np.concatenate([encode([passage_text(d) for d in documents[i:i+4]]) for i in range(0,len(documents),4)])
    lock=threading.Lock()
    app=FastAPI()
    @app.get('/health')
    def health():return {'status':'ready','scope':SCOPE,'documents':len(documents),'sources':sorted(sources),'dimension':1024}
    @app.get('/search')
    def search(query:str,k:int=6):
        if not query.strip() or not 1<=k<=len(documents):raise HTTPException(400,'invalid query/k')
        with lock:
            vector=encode([query],query=True)[0]
        return ranked_results(documents,embeddings,vector,k)
    print(json.dumps(health()),flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port)

if __name__=='__main__':main()
