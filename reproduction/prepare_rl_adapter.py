#!/usr/bin/env python3
"""仅为已有FaithMed parquet补齐env_kwargs，原数据目录不改、不重下载。

复用medmix_preparation.add_env_kwargs；不重排、过滤或重抽样任何行。
输出train/test parquet、保留原manifest并注明schema adapter，整目录原子发布。
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preparation.medmix_preparation import add_env_kwargs


def adapt_parquet(source, target):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features
    original = pq.ParquetFile(source)
    schema = original.schema_arrow
    for field in ('data_source', 'reward_model', 'extra_info'):
        if field not in schema.names:
            raise ValueError(f'{source}: required column {field} missing')
    environment_type = pa.struct([
        pa.field('question', pa.string()), pa.field('question_id', pa.string()),
        pa.field('ground_truth', pa.string()), pa.field('data_source', pa.string()),
        pa.field('extra_info', schema.field('extra_info').type),
    ])
    already_present = 'env_kwargs' in schema.names
    output_schema = schema if already_present else schema.append(pa.field('env_kwargs', environment_type))
    # HF读取parquet会使用嵌入features，需同步新增列描述，保留原有其他feature元数据。
    metadata = dict(output_schema.metadata or {})
    if b'huggingface' in metadata:
        hf = json.loads(metadata[b'huggingface'])
        hf['info']['features']['env_kwargs'] = Features.from_arrow_schema(
            pa.schema([pa.field('env_kwargs', environment_type)])).to_dict()['env_kwargs']
        metadata[b'huggingface'] = json.dumps(hf).encode()
        output_schema = output_schema.with_metadata(metadata)
    count = 0
    with pq.ParquetWriter(target, output_schema) as writer:
        for batch in original.iter_batches(batch_size=128):
            rows = batch.to_pylist()
            environments = [add_env_kwargs(row)['env_kwargs'] for row in rows]
            table = pa.Table.from_batches([batch])
            if not already_present:
                table = table.append_column('env_kwargs', pa.array(environments, type=environment_type))
            table = table.replace_schema_metadata(output_schema.metadata)
            writer.write_table(table)
            count += len(rows)
    if count != original.metadata.num_rows or pq.ParquetFile(target).metadata.num_rows != count:
        raise ValueError(f'{source}: adapter changed row count')
    if count <= 0:
        raise ValueError(f'{source}: empty source parquet')
    return {'rows': count, 'columns_added': [] if already_present else ['env_kwargs'],
            'original_columns': schema.names, 'output_columns': output_schema.names}


def prepare_adapter(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError('output must be a new directory outside the immutable source directory')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f'output directory is not empty: {output_dir}')
    manifest_bytes = (input_dir/'manifest.json').read_bytes()
    original_manifest = json.loads(manifest_bytes)
    if original_manifest.get('status') != 'complete':
        raise ValueError('source manifest does not declare a complete preparation')
    for split in ('train', 'test'):
        if not (input_dir/f'{split}.parquet').is_file():
            raise ValueError(f'source split missing: {split}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}.partial-', dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        counts = {split: adapt_parquet(input_dir/f'{split}.parquet', staging/f'{split}.parquet')
                  for split in ('train', 'test')}
        adapter = {'name': 'medrm_env_kwargs_schema_adapter', 'source_directory': str(input_dir),
                   'source_manifest': 'source_manifest.json', 'splits': counts,
                   'mapping': {'question': 'extra_info.question', 'question_id': 'extra_info.uid',
                               'ground_truth': 'reward_model.ground_truth', 'data_source': 'data_source',
                               'extra_info': 'extra_info'},
                   'data_protocol_changed': False, 'rows_reordered_or_filtered': False,
                   'original_directory_modified': False, 'downloaded': False}
        manifest = {**original_manifest, 'schema_adapter': adapter}
        (staging/'source_manifest.json').write_bytes(manifest_bytes)
        (staging/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if output_dir.exists():
            output_dir.rmdir()
        staging.replace(output_dir)
    print(json.dumps({'output_dir': str(output_dir), 'schema_adapter': adapter}, ensure_ascii=False, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', '--inputdir', required=True, type=Path)
    parser.add_argument('--outputdir', '--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    prepare_adapter(args.input_dir, args.outputdir)


if __name__ == '__main__':
    main()
