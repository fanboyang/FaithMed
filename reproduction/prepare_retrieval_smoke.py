#!/usr/bin/env python3
"""仅准备五源integration smoke子集（每源最多8段），不能作为论文正式检索库。

HF源固定revision，按固定首文件顺序流式读取少量真实记录。StatPearls直接
流式读取NCBI官方归档的首个NXML，不下载整个归档；保留原XML、许可与出处。
不生成embedding或索引，不运行GPU，不以其他来源替代失败来源。
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import xml.etree.ElementTree as ET

SOURCE_ORDER = ('pubmed', 'textbooks', 'wikipedia', 'guidelines', 'statpearls')
SCOPE = 'integration_smoke_subset'
FIELDS = {'id', 'title', 'content', 'data_source', 'source_url'}
HF_SOURCES = {
    'pubmed': {'repo': 'MedRAG/pubmed', 'revision': '33da3593d5756bc04c8909f170003c0b14197957',
               'file': 'chunk/pubmed23n0001.jsonl', 'license': 'MedRAG card does not declare a license grant; underlying article rights retained'},
    'textbooks': {'repo': 'MedRAG/textbooks', 'revision': '9c72838920a1323ffa867467d3f7aa7b36b0f994',
                  'file': 'chunk/Anatomy_Gray.jsonl', 'license': 'MedRAG card does not declare a license grant; underlying textbook rights retained'},
    'wikipedia': {'repo': 'MedRAG/wikipedia', 'revision': 'd76b9ad82135e352235d17e75921c49b68fd07b2',
                  'file': 'chunk/wiki20220301en000.jsonl', 'license': 'Underlying Wikipedia attribution/share-alike terms; retain source attribution'},
    'guidelines': {'repo': 'epfl-llm/guidelines', 'revision': 'a8f0269471088e4c8aafe2319f30c14b2fad82bc',
                   'file': 'open_guidelines.jsonl', 'license': 'Dataset declares Common Crawl terms; original guideline source rights retained'},
}
STATPEARLS_URL = 'https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz'
STATPEARLS_LAST_MODIFIED = 'Sun, 25 Oct 2026 07:25:05 GMT'
STATPEARLS_ARCHIVE_BYTES = 2048277776
STATPEARLS_LICENSE = 'https://creativecommons.org/licenses/by-nc-nd/4.0/'


def require_text(value, field, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f'{field} must be a {"string" if allow_empty else "nonempty string"}')
    return value


def medrag_record(source, row, source_url):
    if source not in ('pubmed', 'textbooks', 'wikipedia'):
        raise ValueError(f'not a MedRAG source: {source}')
    original_id = require_text(row.get('id'), 'id')
    return {'id': f'{source}:{original_id}', 'title': require_text(row.get('title'), 'title', allow_empty=True),
            'content': require_text(row.get('content'), 'content'), 'data_source': source,
            'source_url': require_text(source_url, 'source_url')}


def guideline_records(row, limit=8):
    original_id = require_text(row.get('id'), 'id')
    title = require_text(row.get('title'), 'title', allow_empty=True)
    content = require_text(row.get('clean_text'), 'clean_text')
    source_url = require_text(row.get('url'), 'url')
    # 仅按原文空行切段；保留段内文字、标点及换行，不改写、不补全标题。
    paragraphs = [part.strip() for part in re.split(r'\n\s*\n', content) if part.strip()]
    return [{'id': f'guidelines:{original_id}:paragraph:{i}', 'title': title, 'content': paragraph,
             'data_source': 'guidelines', 'source_url': source_url}
            for i, paragraph in enumerate(paragraphs[:limit])]


def statpearls_records(xml_bytes, archive_member, limit=8):
    root = ET.fromstring(xml_bytes)
    title_node = root.find('.//title')
    if title_node is None:
        raise ValueError('StatPearls NXML has no title')
    title = require_text(''.join(title_node.itertext()), 'StatPearls title')
    permissions = root.find('.//permissions')
    if permissions is None:
        raise ValueError('StatPearls original permissions missing')
    license_node = permissions.find('license')
    license_url = license_node.get('{http://www.w3.org/1999/xlink}href') if license_node is not None else None
    if license_url != STATPEARLS_LICENSE:
        raise ValueError(f'StatPearls first-article license differs from verified source: {license_url}')
    paragraphs = []
    for section in root.iter('sec'):
        for paragraph in section.findall('p'):
            value = ''.join(paragraph.itertext()).strip()
            if value:
                paragraphs.append(value)
    if not paragraphs:
        raise ValueError('StatPearls first NXML contains no original section paragraphs')
    source_url = STATPEARLS_URL+'#'+archive_member
    records = [{'id': f'statpearls:{Path(archive_member).name}:paragraph:{i}', 'title': title,
                'content': value, 'data_source': 'statpearls', 'source_url': source_url}
               for i, value in enumerate(paragraphs[:limit])]
    metadata = {'source_url': STATPEARLS_URL, 'archive_member': archive_member,
                'revision': f'HTTP Last-Modified: {STATPEARLS_LAST_MODIFIED}; bytes: {STATPEARLS_ARCHIVE_BYTES}',
                'license': license_url, 'selection': 'first NXML, first nonempty direct section paragraphs',
                'license_scope': 'local noncommercial smoke; retain attribution and do not redistribute adapted text',
                'archive_is_historical_medrag_snapshot': False,
                'original_xml': 'raw/statpearls/original.nxml', 'permissions': 'raw/statpearls/permissions.xml'}
    artifacts = {'raw/statpearls/original.nxml': xml_bytes,
                 'raw/statpearls/permissions.xml': ET.tostring(permissions, encoding='utf-8')}
    return records, metadata, artifacts


def fetch_hf_records(source, limit):
    import requests
    specification = HF_SOURCES[source]
    relative = f'/datasets/{specification["repo"]}/resolve/{specification["revision"]}/{specification["file"]}'
    canonical_url = 'https://huggingface.co'+relative
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    records = []
    with requests.get(endpoint+relative, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        for line_number, raw_line in enumerate(response.iter_lines(chunk_size=65536), 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f'{source} fixed source JSONL line {line_number} is invalid') from error
            if not isinstance(row, dict):
                raise ValueError(f'{source} fixed source JSONL line {line_number} is not an object')
            if source == 'guidelines':
                records.extend(guideline_records(row, limit-len(records)))
            else:
                records.append(medrag_record(source, row, canonical_url+'#'+require_text(row.get('id'), 'id')))
            if len(records) >= limit:
                break  # 立即关闭连接，不遍历/缓存完整大文件。
    if not records:
        raise ValueError(f'{source}: fixed source yielded no real content')
    metadata = {**specification, 'source_url': canonical_url,
                'selection': 'fixed first file; first original clean_text paragraphs' if source == 'guidelines'
                             else 'fixed first chunk file; first original records',
                'download': 'HTTP stream closed after selected records; not a complete source download'}
    return records, metadata


class LimitedReader:
    def __init__(self, raw, maximum):
        self.raw, self.maximum, self.bytes_read = raw, maximum, 0

    def read(self, size=-1):
        remaining = self.maximum-self.bytes_read
        if remaining <= 0:
            raise ValueError('StatPearls smoke compressed-read budget exhausted before first NXML')
        data = self.raw.read(remaining if size < 0 else min(size, remaining))
        self.bytes_read += len(data)
        return data


def fetch_statpearls(limit):
    import requests
    with requests.get(STATPEARLS_URL, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        if (response.headers.get('Last-Modified') != STATPEARLS_LAST_MODIFIED
                or response.headers.get('Content-Length') != str(STATPEARLS_ARCHIVE_BYTES)):
            raise ValueError('StatPearls current official archive changed; review and pin new metadata explicitly')
        reader = LimitedReader(response.raw, 8*1024*1024)
        with tarfile.open(fileobj=reader, mode='r|gz') as archive:
            for member in archive:
                if member.isfile() and member.name.endswith('.nxml'):
                    if not 0 < member.size <= 4*1024*1024:
                        raise ValueError('StatPearls first NXML exceeds smoke article budget or is empty')
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError('StatPearls first NXML cannot be read')
                    xml_bytes = stream.read()
                    if len(xml_bytes) != member.size:
                        raise ValueError('StatPearls first NXML is incomplete')
                    records, metadata, artifacts = statpearls_records(xml_bytes, member.name, limit)
                    metadata['compressed_bytes_read'] = reader.bytes_read
                    metadata['whole_archive_downloaded'] = False
                    return records, metadata, artifacts
    raise ValueError('StatPearls archive has no NXML within smoke read budget')


def validate_records(records):
    counts = Counter()
    ids = set()
    for record in records:
        if set(record) != FIELDS:
            raise ValueError(f'invalid smoke record fields: {set(record)}')
        for key, value in record.items():
            require_text(value, key, allow_empty=key=='title')
        if record['id'] in ids:
            raise ValueError(f'duplicate smoke record id: {record["id"]}')
        ids.add(record['id'])
        counts[record['data_source']] += 1
    if set(counts) != set(SOURCE_ORDER) or any(not 1 <= n <= 8 for n in counts.values()):
        raise ValueError(f'require all five original sources and 1..8 real chunks per source, got {dict(counts)}')
    return dict(counts)


def prepare_smoke(output_dir, per_source=8, hf_loader=None, stat_loader=None):
    if type(per_source) is not int or not 1 <= per_source <= 8:
        raise ValueError('per_source must be an integer in 1..8')
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f'output directory is not empty: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}.partial-', dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        records, sources = [], {}
        for source in SOURCE_ORDER:
            print(f'fetching integration smoke source: {source}', flush=True)
            if source == 'statpearls':
                selected, metadata, artifacts = (stat_loader or fetch_statpearls)(per_source)
                for relative, content in artifacts.items():
                    path = staging/relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
            else:
                selected, metadata = (hf_loader or fetch_hf_records)(source, per_source)
            if not selected or len(selected) > per_source or any(r.get('data_source') != source for r in selected):
                raise ValueError(f'{source}: missing/mismatched source or exceeded requested subset size')
            records.extend(selected)
            sources[source] = metadata
        counts = validate_records(records)
        manifest = {'status': 'complete', 'scope': SCOPE, 'source_counts': counts, 'sources': sources,
                    'total_chunks': len(records), 'maximum_per_source': per_source,
                    'schema': sorted(FIELDS), 'selection_seed': None,
                    'not_a_full_corpus': True, 'not_a_paper_result': True,
                    'purpose': 'retrieval/reward/one-update integration only; never report as full-corpus evaluation'}
        with (staging/'corpus.jsonl').open('w', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        (staging/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (staging/'raw/statpearls/source.json').write_text(
            json.dumps(sources['statpearls'], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if output_dir.exists():
            output_dir.rmdir()
        staging.replace(output_dir)
    print(json.dumps({'output_dir': str(output_dir), 'scope': SCOPE, 'source_counts': counts}, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputdir', '--output-dir', required=True, type=Path)
    parser.add_argument('--per-source', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args(argv)
    prepare_smoke(args.outputdir, args.per_source)


if __name__ == '__main__':
    main()
