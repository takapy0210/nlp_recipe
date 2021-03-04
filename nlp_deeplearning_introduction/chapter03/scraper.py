import json


def load_json(filename):
    with open(filename) as f:
        return json.loads(f.read())


def extract_data(response):
    for key in response['response'].keys():
        if not key.isdigit():
            continue
        d = response['response'][key]['photo']
        if d.get('comment') and d.get('total_score'):
            comment = d['comment']
            score = d['total_score']
            data = {
                'comment': comment,
                'score': score
            }
            yield data


def save_as_json(save_file, record):
    with open(save_file, mode='a') as f:
        f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    file_name = 'data/response.json'
    save_file = 'data/dataset.jsonl'
    response = load_json(file_name)
    records = extract_data(response)
    for record in records:
        save_as_json(save_file, record)
