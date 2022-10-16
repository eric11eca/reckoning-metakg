import json


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_jsonl(path, mode="r", **kwargs):
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path, mode="w"):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines) + "\n", path, mode=mode)
