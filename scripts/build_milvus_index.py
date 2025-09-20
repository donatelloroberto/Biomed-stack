#!/usr/bin/env python3
"""Simple script to build a Milvus collection from a JSONL of passages.
Input JSONL format: {"id": "<unique>", "text": "<passage>", "meta": {...}}
"""
import argparse
import json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--collection', default='pubmed')
args = parser.parse_args()

embed = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
connections.connect(host='localhost', port='19530')

# create collection if not exists
if not Collection.exists(args.collection):
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description='PubMed passages')
    col = Collection(args.collection, schema)
else:
    col = Collection(args.collection)

batch = []
texts = []
ids = []
with open(args.input, 'r') as f:
    for line in f:
        rec = json.loads(line)
        txt = rec.get('text')
        vec = embed.encode(txt)
        texts.append(txt)
        batch.append(vec)

# convert to columns and insert
col.insert([texts, batch])
col.flush()
print('Inserted')
