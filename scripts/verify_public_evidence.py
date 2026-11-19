"""Check the curated evidence bundle without executing logs, models or agents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def strict_json(text):
    def invalid(value):
        raise ValueError('non-finite JSON number: ' + value)
    return json.loads(text, parse_constant=invalid)


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = strict_json((root / 'MANIFEST.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'curated_read_only_experiment_evidence':
        raise ValueError('unsupported evidence manifest')
    records = manifest.get('files')
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty evidence file list required')
    seen = set()
    total = 0
    for record in records:
        name = record['path']
        path = PurePosixPath(name)
        if (not isinstance(name, str) or '\\' in name or path.is_absolute()
                or '..' in path.parts or path.as_posix() != name or name in seen):
            raise ValueError('unsafe or duplicate evidence path: ' + str(name))
        if record['source'] != 'runs/' + name:
            raise ValueError('source identity differs: ' + name)
        for key in ('source_sha256', 'published_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', record[key]):
                raise ValueError('invalid digest: ' + name)
        file = root / name
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError('evidence escapes bundle: ' + name)
        data = file.read_bytes()
        if len(data) != record['published_bytes'] or hashlib.sha256(data).hexdigest() != record['published_sha256']:
            raise ValueError('published bytes changed: ' + name)
        text = data.decode('utf-8')
        if file.suffix == '.json':
            strict_json(text)
        elif file.suffix == '.jsonl':
            for line in text.splitlines():
                if line.strip():
                    strict_json(line)
        elif file.suffix not in ('.md', '.txt', '.csv'):
            raise ValueError('unexpected evidence format: ' + name)
        seen.add(name)
        total += len(data)
    # README files are editorial guides; the manifest covers extracted artifacts.
    unexpected = [p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and p.name not in ('README.md', 'MANIFEST.json')
                  and p.relative_to(root).as_posix() not in seen]
    if unexpected:
        raise ValueError('unlisted evidence files: ' + ', '.join(sorted(unexpected)))
    return {'files': len(seen), 'published_bytes': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'experiments')
    args = parser.parse_args()
    try:
        result = verify_bundle(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Evidence verification failed: ' + str(error) + '\n')
    print(json.dumps({'status': 'verified', **result}))
    print('Scope: published bytes and parseability; no model execution or performance certification.')


if __name__ == '__main__':
    main()
