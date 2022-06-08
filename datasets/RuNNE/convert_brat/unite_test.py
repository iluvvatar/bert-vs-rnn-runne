import json


def unite_test(test_texts_file, test_entities_file, file_out):
    docs = {}   # id: dict
    with open(test_texts_file, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            docs[doc['id']] = doc
    with open(test_entities_file, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            docs[doc['id']]['ners'] = doc['ners']
    ids = sorted(docs.keys())
    with open(file_out, 'w', encoding='utf-8') as f:
        for doc_id in ids:
            doc = docs[doc_id]
            print(json.dumps(doc, ensure_ascii=False), file=f)


if __name__ == '__main__':
    unite_test('../../../../../../../datasets/RuNNE/_test_texts.jsonl', '_test_entities.jsonl', '_test.jsonl')
