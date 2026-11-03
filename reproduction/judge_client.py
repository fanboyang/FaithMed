"""Audited GPT-6-Luna judge, preserving successful/terminal responses without resampling.

The canonical full request, endpoint and rubric types form the SQLite cache key.
No request hashes or credentials are written. Pending requests after interruption
are deliberately ambiguous: they fail explicitly rather than being resampled.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import urllib.request


class JudgeResponseError(RuntimeError):
    """A received judge result was invalid or a cached failure is terminal."""


class JudgeTransportError(RuntimeError):
    """A request failed at the network/HTTP layer; consult the local audit."""


class JudgeInFlightError(RuntimeError):
    """Identical request is running or was interrupted with unknown outcome."""


def post_raw(endpoint, body, api_key, timeout):
    request=urllib.request.Request(endpoint,data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+api_key})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return response.read().decode('utf-8',errors='replace')


def parse_scores(raw, rubrics):
    response=json.loads(raw)
    content=response['choices'][0]['message']['content']
    if not isinstance(content,str):
        raise ValueError('judge message content is not text')
    content=re.sub(r'^```json\s*','',content.strip())
    content=re.sub(r'\s*```$','',content)
    parsed=json.loads(content)
    if not isinstance(parsed,dict):
        raise ValueError('rubric result must be an object')
    result={}
    for rubric in rubrics:
        entry=parsed[rubric['id']]  # Missing fields are failures, never N/A.
        score=entry['score'] if isinstance(entry,dict) else entry
        if isinstance(score,str) and re.fullmatch(r'-?\d+',score):
            score=int(score)
        maximum=1 if rubric['type']=='binary' else 2
        if type(score) is not int or score not in range(-1,maximum+1):
            raise ValueError('rubric score is outside its declared scale')
        result[rubric['id']]=score
    return result


class JudgeClient:
    def __init__(self, api_key, audit_db, endpoint='https://api.zcloudapi.com/v1/chat/completions',
                 model='gpt-6-luna', max_attempts=3, timeout=180):
        if not api_key:
            raise ValueError('OPENAI_API_KEY is required')
        if model != 'gpt-6-luna':
            raise ValueError('this reproduction fixes the judge to gpt-6-luna')
        parsed=urlsplit(endpoint)
        if parsed.scheme not in ('https','http') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('judge URL must be a credential-free Chat Completions endpoint')
        if max_attempts < 1 or timeout <= 0:
            raise ValueError('positive retry count and timeout are required')
        self.api_key=api_key
        self.endpoint=endpoint
        self.model=model
        self.max_attempts=max_attempts
        self.timeout=timeout
        self.audit_db=Path(audit_db)
        self.audit_db.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS requests (
                  request_key TEXT PRIMARY KEY, config TEXT NOT NULL,
                  status TEXT NOT NULL, result TEXT, error_type TEXT,
                  created_at REAL NOT NULL, updated_at REAL NOT NULL, owner_pid INTEGER, owner_host TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                  request_key TEXT NOT NULL, attempt INTEGER NOT NULL,
                  http_status INTEGER, raw_response TEXT, usage TEXT,
                  error_type TEXT, created_at REAL NOT NULL);
            ''')
            columns={row[1] for row in connection.execute('PRAGMA table_info(requests)')}
            for column,kind in [('owner_pid','INTEGER'),('owner_host','TEXT')]:
                if column not in columns:
                    connection.execute(f'ALTER TABLE requests ADD COLUMN {column} {kind}')

    @contextmanager
    def connect(self):
        connection=sqlite3.connect(self.audit_db,timeout=60)
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin(self,key,config):
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row=connection.execute('SELECT status,result,owner_pid,owner_host FROM requests WHERE request_key=?',(key,)).fetchone()
            if row and row[0]=='ok':
                return json.loads(row[1])
            if row and row[0]=='terminal_failure':
                raise JudgeResponseError('cached terminal judge failure; inspect audit database')
            if row and row[0]=='pending':
                raise JudgeInFlightError('request pending; never resample an ambiguous request',row[2],row[3])
            now=time.time()
            connection.execute('''INSERT INTO requests(request_key,config,status,result,error_type,created_at,updated_at,owner_pid,owner_host) VALUES(?,?,?,NULL,NULL,?,?,?,?)
              ON CONFLICT(request_key) DO UPDATE SET status='pending',result=NULL,
              error_type=NULL,updated_at=excluded.updated_at,owner_pid=excluded.owner_pid,owner_host=excluded.owner_host''', (key,config,'pending',now,now,os.getpid(),socket.gethostname()))
        return None

    def finish(self,key,status,result=None,error_type=None):
        with self.connect() as connection:
            connection.execute('UPDATE requests SET status=?,result=?,error_type=?,updated_at=? WHERE request_key=?',
                (status,json.dumps(result) if result is not None else None,error_type,time.time(),key))

    def audit_attempt(self,key,attempt,status,raw,error_type=None):
        usage=None
        if raw is not None:
            raw=raw.replace(self.api_key,'[REDACTED_API_KEY]')
            try:
                decoded=json.loads(raw)
                if isinstance(decoded,dict) and 'usage' in decoded:
                    usage=json.dumps(decoded['usage'])
            except (ValueError,TypeError):
                pass
        with self.connect() as connection:
            connection.execute('INSERT INTO attempts(request_key,attempt,http_status,raw_response,usage,error_type,created_at) VALUES(?,?,?,?,?,?,?)',
                               (key,attempt,status,raw,usage,error_type,time.time()))

    def score(self,prompt,rubrics,max_tokens=512):
        if self.api_key in prompt:
            raise ValueError('judge prompt contains the API credential')
        if not rubrics or len({r['id'] for r in rubrics}) != len(rubrics):
            raise ValueError('distinct nonempty rubric IDs are required')
        body={'model':self.model,'messages':[{'role':'user','content':prompt}],
              'temperature':0,'reasoning_effort':'none','max_completion_tokens':max_tokens,
              'response_format':{'type':'json_object'}}
        config={'protocol_version':1,'endpoint':self.endpoint,'body':body,
                'rubrics':[{'id':r['id'],'type':r['type']} for r in rubrics],
                'timeout':self.timeout,'max_attempts':self.max_attempts}
        request_identity={k:v for k,v in config.items() if k not in ('timeout','max_attempts')}
        canonical=json.dumps(request_identity,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        deadline=time.monotonic()+self.timeout*self.max_attempts+30
        while True:
            try:
                cached=self.begin(canonical,json.dumps(config,ensure_ascii=False,sort_keys=True))
                break
            except JudgeInFlightError as exc:
                owner_pid,owner_host=exc.args[1:]
                if owner_pid is None or owner_host is None:
                    raise
                if owner_host==socket.gethostname():
                    try:
                        os.kill(owner_pid,0)
                    except ProcessLookupError:
                        raise JudgeInFlightError('prior owner died with ambiguous request; inspect audit') from None
                if time.monotonic()>=deadline:
                    raise JudgeInFlightError('timed out awaiting identical in-flight judge request') from None
                time.sleep(.1)
        if cached is not None:
            return cached
        for attempt in range(1,self.max_attempts+1):
            try:
                raw=post_raw(self.endpoint,body,self.api_key,self.timeout)
            except HTTPError as exc:
                error_type=type(exc).__name__
                self.audit_attempt(canonical,attempt,exc.code,exc.read().decode('utf-8',errors='replace'),error_type)
                transient=exc.code==429 or 500<=exc.code<600
            except (URLError,TimeoutError,ConnectionError,OSError) as exc:
                error_type=type(exc).__name__
                self.audit_attempt(canonical,attempt,None,None,error_type)
                transient=True
            except Exception as exc:
                self.audit_attempt(canonical,attempt,None,None,type(exc).__name__)
                self.finish(canonical,'terminal_failure',error_type=type(exc).__name__)
                raise JudgeTransportError('unexpected judge transport failure; inspect audit database') from None
            else:
                # Persist every valid HTTP response BEFORE attempting semantic parsing.
                self.audit_attempt(canonical,attempt,200,raw)
                try:
                    scores=parse_scores(raw,rubrics)
                except Exception as exc:
                    self.finish(canonical,'terminal_failure',error_type=type(exc).__name__)
                    raise JudgeResponseError('invalid judge response; original saved, no resampling') from None
                self.finish(canonical,'ok',result=scores)
                return scores
            if not transient or attempt==self.max_attempts:
                self.finish(canonical,'retryable_failure' if transient else 'terminal_failure',error_type=error_type)
                raise JudgeTransportError('judge request failed; inspect audit database') from None
            time.sleep(min(2**(attempt-1),4))
        raise AssertionError('unreachable retry state')


def from_environment():
    audit_dir=os.environ.get('FAITHMED_JUDGE_AUDIT_DIR')
    if not audit_dir:
        raise ValueError('FAITHMED_JUDGE_AUDIT_DIR is required for raw-response auditing')
    return JudgeClient(os.environ.get('OPENAI_API_KEY'),Path(audit_dir)/'judge.sqlite',
                       endpoint=os.environ.get('OPENAI_API_URL','https://api.zcloudapi.com/v1/chat/completions'))
