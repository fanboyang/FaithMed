"""Offline recovery of exact <escape>JSON</escape> provider responses.

SOURCE is opened read-only and copied to a NEW output database. No HTTP request
is made. Original raw responses/attempts are retained; every status conversion is
appended to replay_history. Unknown pending requests remain pending/blocked.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from reproduction.judge_client import parse_scores

DECODER='exact_provider_escape_json_v1'


def replay_saved(source,output):
    source=Path(source).resolve(strict=True)
    output=Path(output).resolve()
    if source==output:
        raise ValueError('output must be a new database, never the source')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb'):
        pass
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as original:
        with closing(sqlite3.connect(output)) as replay:
            original.backup(replay)
            replay.executescript('''
              CREATE TABLE IF NOT EXISTS replay_history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                request_key TEXT NOT NULL, source_database TEXT NOT NULL,
                source_attempt_sequence INTEGER NOT NULL, decoder TEXT NOT NULL,
                old_status TEXT NOT NULL, old_result TEXT, old_error_type TEXT,
                new_status TEXT NOT NULL, new_result TEXT NOT NULL,
                created_at REAL NOT NULL);
            ''')
            rows=replay.execute('SELECT request_key,status,result,error_type FROM requests WHERE status=?',
                                ('terminal_failure',)).fetchall()
            report={'source_database':str(source),'output_database':str(output),'decoder':DECODER,
                    'terminal_examined':len(rows),'replayed':0,'not_replayable':0,
                    'pending_blocked':replay.execute('SELECT count(*) FROM requests WHERE status="pending"').fetchone()[0],
                    'skips':[]}
            with replay:
                for key,old_status,old_result,old_error in rows:
                    successes=replay.execute('SELECT sequence,raw_response FROM attempts WHERE request_key=? AND http_status=200 ORDER BY sequence',
                                             (key,)).fetchall()
                    attempt_sequence=None
                    try:
                        if len(successes)!=1:
                            raise ValueError('require exactly one saved successful HTTP response')
                        attempt_sequence,raw=successes[0]
                        content=json.loads(raw)['choices'][0]['message']['content']
                        if not isinstance(content,str):
                            raise ValueError('saved message content is not text')
                        content=content.strip()
                        if not(content.startswith('<escape>') and content.endswith('</escape>')):
                            raise ValueError('not the exact supported provider wrapper')
                        request=json.loads(key)
                        if request['body']['model']!='gpt-6-luna':
                            raise ValueError('unexpected requested judge model')
                        result=parse_scores(raw,request['rubrics'])
                    except (ValueError,KeyError,TypeError,IndexError) as exc:
                        report['not_replayable']+=1
                        report['skips'].append({'source_attempt_sequence':attempt_sequence,'error_type':type(exc).__name__})
                        continue
                    encoded=json.dumps(result)
                    now=time.time()
                    replay.execute('''INSERT INTO replay_history
                       (request_key,source_database,source_attempt_sequence,decoder,old_status,old_result,old_error_type,new_status,new_result,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)''',
                       (key,str(source),attempt_sequence,DECODER,old_status,old_result,old_error,'ok',encoded,now))
                    replay.execute('UPDATE requests SET status=?,result=?,error_type=NULL,updated_at=? WHERE request_key=?',
                                   ('ok',encoded,now,key))
                    report['replayed']+=1
            return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(replay_saved(args.source,args.output),indent=2))

if __name__=='__main__':main()
