"""Development-only, synthetic coding evaluation. Never part of the app update.

Run --validate-fixtures without an Ollama model. Live runs write private reports
under data/ and repair disposable copies. This is not an OS security sandbox.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
VALIDATION_COMMAND = "python -m unittest -q"

CASES = {
    "ledger": {
        "goal": "Repair cents(text) in main.py. Convert positive ordinary decimal amounts into exact integer cents. Strip surrounding whitespace. Allow one or two fractional digits or integers. Reject zero, negatives, empty/invalid strings, NaN, infinity, scientific notation, more than two fractional digits and non-string inputs with ValueError. Do not round excess decimal places.",
        "broken": '''def cents(text):
    return int(float(text) * 100)
''',
        "reference": '''from decimal import Decimal
import re
def cents(text):
    if not isinstance(text, str):
        raise ValueError('Expected an amount string')
    text = text.strip()
    if not re.fullmatch(r'[0-9]+(?:\\.[0-9]{1,2})?', text):
        raise ValueError('Invalid amount')
    value = Decimal(text) * 100
    if value <= 0:
        raise ValueError('Amount must be positive')
    return int(value)
''',
        "tests": '''from main import cents
class Checks(unittest.TestCase):
    def test_exact_cents(self):
        for text, expected in [('0.29',29),('1.10',110),('19.99',1999),(' 12.5 ',1250),('2',200),('90071992547409.93',9007199254740993)]:
            with self.subTest(text=text): self.assertEqual(cents(text),expected)
    def test_invalid_amounts(self):
        for text in ['',' ','0','0.00','-1','NaN','Infinity','inf','1e2','1.005','1.000','abc','$2',None,True,3]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError): cents(text)
    def test_integer_output(self):
        self.assertIs(type(cents('4.20')),int)
''',
    },
    "stable_ids": {
        "goal": "Repair Store in main.py. Store.items is a list of dictionaries with id and text. add(text) appends a task and returns its unique persistent string ID. remove(id) removes only the matching task and returns True, or False if absent. IDs must remain stable for surviving tasks and never be recycled after deletion, including after to_json()/from_json() reload. to_json() returns JSON text; Store.from_json(text) restores a new Store preserving the IDs and text.",
        "broken": '''import json
class Store:
    def __init__(self): self.items = []
    def add(self, text):
        identifier = str(len(self.items) + 1)
        self.items.append({'id': identifier, 'text': text})
        return identifier
    def remove(self, identifier):
        for index, item in enumerate(self.items):
            if item['id'] == identifier:
                self.items.pop(index)
                return True
        return False
    def to_json(self): return json.dumps(self.items)
    @classmethod
    def from_json(cls, text):
        result = cls()
        result.items = json.loads(text)
        return result
''',
        "reference": '''import json
from uuid import uuid4
class Store:
    def __init__(self): self.items = []
    def add(self, text):
        identifier = uuid4().hex
        self.items.append({'id': identifier, 'text': text})
        return identifier
    def remove(self, identifier):
        for index, item in enumerate(self.items):
            if item['id'] == identifier:
                self.items.pop(index)
                return True
        return False
    def to_json(self): return json.dumps(self.items)
    @classmethod
    def from_json(cls, text):
        result = cls()
        result.items = json.loads(text)
        return result
''',
        "tests": '''from main import Store
class Checks(unittest.TestCase):
    def test_delete_add_and_remove_correct_task(self):
        store=Store(); first=store.add('one'); second=store.add('two')
        self.assertTrue(store.remove(first)); third=store.add('three')
        self.assertNotEqual(second,third)
        self.assertEqual(store.items[0]['id'],second)
        self.assertTrue(store.remove(second))
        self.assertEqual(store.items,[{'id':third,'text':'three'}])
    def test_reload_does_not_recycle_deleted_id(self):
        store=Store(); removed=store.add('removed'); store.remove(removed)
        restored=Store.from_json(store.to_json()); new=restored.add('new')
        self.assertNotEqual(new,removed)
        self.assertIsInstance(new,str)
    def test_round_trip_preserves_data(self):
        store=Store(); identifier=store.add('日本語 task')
        restored=Store.from_json(store.to_json())
        self.assertEqual(restored.items,[{'id':identifier,'text':'日本語 task'}])
        self.assertFalse(restored.remove('not-found'))
        restored.add('second'); self.assertEqual(len(store.items),1)
''',
    },
    "atomic_save": {
        "goal": "Repair save_records(path, records) in main.py. Accept only a list of dictionaries, each having a nonempty string id and a string value, with unique IDs. Reject invalid data with ValueError before changing the existing file. Write valid records as UTF-8 JSON with an atomic replacement using a temporary file in the same directory. Return None. If os.replace fails, propagate OSError, preserve the old file and remove temporary files. Do not create missing parent directories.",
        "broken": '''import json
def save_records(path, records):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(records, stream)
''',
        "reference": '''import json
import os
from pathlib import Path
import tempfile
def save_records(path, records):
    if not isinstance(records, list): raise ValueError('Expected records list')
    seen = set()
    for record in records:
        if not isinstance(record,dict) or not isinstance(record.get('id'),str) or not record['id'] or not isinstance(record.get('value'),str) or record['id'] in seen:
            raise ValueError('Invalid record')
        seen.add(record['id'])
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(records,stream,ensure_ascii=False)
        os.replace(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
''',
        "tests": '''from main import save_records
class Checks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path.cwd()); self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name); self.file=self.folder/'records.json'
        self.old=b'[{"id":"old","value":"preserve"}]'; self.file.write_bytes(self.old)
    def test_valid_round_trip(self):
        records=[{'id':'one','value':'日本語'}]
        self.assertIsNone(save_records(self.file,records))
        self.assertEqual(json.loads(self.file.read_text(encoding='utf-8')),records)
    def test_invalid_preserves_old_file(self):
        for records in [None,{},[None],[{'id':'','value':'x'}],[{'id':'one','value':3}],[{'id':'same','value':'a'},{'id':'same','value':'b'}]]:
            with self.subTest(records=records):
                with self.assertRaises(ValueError): save_records(self.file,records)
                self.assertEqual(self.file.read_bytes(),self.old)
    def test_replace_failure_preserves_old_file_and_cleans_temporary(self):
        with patch('os.replace',side_effect=OSError('simulated replacement failure')):
            with self.assertRaises(OSError): save_records(self.file,[{'id':'new','value':'value'}])
        self.assertEqual(self.file.read_bytes(),self.old)
        self.assertEqual(list(self.folder.iterdir()),[self.file])
''',
    },
}

TEST_HEADER = '''import sys, os, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
workspace=Path(sys.argv.pop()).resolve()
sys.path.insert(0,str(workspace))
# An audit guard catches accidental process/network access and writes outside
# the fixture. It is defense in depth, not an adversarial OS sandbox.
def within(path):
    if isinstance(path,int): return True
    try: return Path(path).resolve().is_relative_to(workspace)
    except (TypeError,ValueError): return False
def audit(event,args):
    if event in {'subprocess.Popen','os.system','os.exec','os.posix_spawn','socket.__new__','socket.connect','ctypes.dlopen'}:
        raise PermissionError('Evaluation prohibits process/network access')
    if event=='open':
        mode=args[1] or ''
        flags=args[2] or 0
        writing=any(letter in str(mode) for letter in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and not within(args[0]): raise PermissionError('Evaluation writes must stay in the fixture')
    if event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate'} and not within(args[0]):
        raise PermissionError('Evaluation mutation must stay in the fixture')
    if event in {'os.rename','os.link','os.symlink'} and (not within(args[0]) or not within(args[1])):
        raise PermissionError('Evaluation move must stay in the fixture')
sys.addaudithook(audit)
'''


def prepare_case(case: str, directory: Path, *, reference: bool = False):
    directory = directory.resolve()
    workspace = directory / "project"
    workspace.mkdir(parents=True)
    source = CASES[case]["reference" if reference else "broken"]
    (workspace / "main.py").write_text(source, encoding="utf-8")
    tests = directory / "fixed_tests.py"
    tests.write_text(TEST_HEADER + CASES[case]["tests"] + "\nif __name__=='__main__': unittest.main()\n", encoding="utf-8")
    # Show the exact checks named by tracebacks without giving the agent write
    # access to the authoritative runner. Validation always uses the sibling
    # file outside this editable project, never this reference copy.
    reference = workspace / "fixed_tests.py"
    reference.write_bytes(tests.read_bytes())
    reference.chmod(0o444)
    return workspace, tests


def validate(workspace: Path, tests: Path, *, timeout: float = 20) -> dict:
    before = hashlib.sha256(tests.read_bytes()).hexdigest()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    try:
        process = subprocess.run([sys.executable, "-I", str(tests), "-q", str(workspace)],
            cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        result = {"exit_code":process.returncode,"output":(process.stdout+process.stderr)[-12000:]}
    except subprocess.TimeoutExpired:
        result = {"exit_code":1,"output":"External validation exceeded its time limit."}
    if hashlib.sha256(tests.read_bytes()).hexdigest() != before:
        return {"exit_code":1,"output":"Fixed tests were altered; evaluation invalid."}
    return result


def load_core(core_path: Path | None = None):
    path = (core_path or ROOT / "agent_core.py").resolve()
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location("_benchmark_agent_core", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the requested agent core.")
    module = importlib.util.module_from_spec(spec)
    # Compile exactly the bytes being fingerprinted, including frozen baselines.
    # Concurrent source saves cannot relabel the loaded implementation later.
    exec(compile(source, str(path), "exec"), module.__dict__)
    module._benchmark_source_sha256 = hashlib.sha256(source).hexdigest()
    return module


def evaluation_agent(core, workspace, tests, emit, cancel, model, think):
    class EvaluationAgent(core.Agent):
        def _inference_options(self):
            return {**super()._inference_options(), "seed": 42}

        def _execute(self, name, arguments):
            if name not in {"list_files","read_file","write_file","edit_file","search_files","run_command","set_plan","read_working_history"}:
                raise PermissionError("This development evaluation permits only project file tools and the fixed test command.")
            if name in {"write_file","edit_file"} and self._path(arguments.get("path","")) != self._path("main.py"):
                raise PermissionError("Only main.py is editable in this evaluation.")
            return super()._execute(name,arguments)

        def _run_command(self, command, timeout_seconds=90):
            if command.strip() != VALIDATION_COMMAND:
                self._denied = True
                raise PermissionError("Only python -m unittest -q is approved for this evaluation; no shell is executed.")
            self._check_cancelled()
            result = validate(workspace,tests,timeout=min(20,max(1,int(timeout_seconds))))
            self._record_command_result(result)
            return result
    return EvaluationAgent(workspace,emit,lambda *_:False,cancel,model=model,think=think,vision_supported=False)


@contextmanager
def run_limits(core, steps, seconds):
    previous=(core.MAX_RUN_STEPS,core.MAX_RUN_SECONDS)
    core.MAX_RUN_STEPS,core.MAX_RUN_SECONDS=steps,seconds
    try: yield
    finally: core.MAX_RUN_STEPS,core.MAX_RUN_SECONDS=previous


def evaluate(case, core, directory, *, model, think, max_steps, max_seconds):
    workspace,tests=prepare_case(case,directory)
    events=[]
    def emit(kind,payload):
        events.append({"event":kind,"payload":payload})
        if kind in {"tool","error","done"}:
            print(json.dumps({"task":case,"event":kind,"name":payload.get("name"),"text":payload.get("text")}),flush=True)
    cancel=threading.Event()
    agent=evaluation_agent(core,workspace,tests,emit,cancel,model,think)
    timer=threading.Timer(max_seconds,cancel.set)
    timer.daemon=True
    start=time.monotonic()
    before=validate(workspace,tests)
    timer.start()
    try:
        with run_limits(core,max_steps,max_seconds):
            agent.run(CASES[case]["goal"]+" Work directly on main.py using the file tools. Run python -m unittest -q to execute the fixed external checks. Fix failures and verify before claiming success. Only main.py may be edited; use standard-library Python. Do not install dependencies, launch apps, or access the network.")
    finally: timer.cancel()
    after=validate(workspace,tests)
    elapsed=round(time.monotonic()-start,2)
    errors=[item["payload"] for item in events if item["event"]=="error"]
    report={"task":case,"model":model,"think":think,"seed":42,
            "core_sha256":getattr(core,"_benchmark_source_sha256",None) or hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "max_steps":max_steps,"max_seconds":max_seconds,
            "passed":after["exit_code"]==0,"baseline_failed":before["exit_code"]!=0,
            "elapsed_seconds":elapsed,"tool_calls":sum(item["event"]=="tool" for item in events),
            "timed_out":cancel.is_set(),"errors":errors,"validation":after,"history":agent.history,"events":events}
    (directory/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return {key:value for key,value in report.items() if key not in {"history","events"}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--think",action="store_true")
    parser.add_argument("--tasks",nargs="+",choices=list(CASES),default=list(CASES))
    parser.add_argument("--max-seconds",type=int,default=180)
    parser.add_argument("--max-steps",type=int,default=8)
    parser.add_argument("--core-path",type=Path)
    parser.add_argument("--output",type=Path,help="New report directory inside private data/; never overwritten.")
    parser.add_argument("--validate-fixtures",action="store_true")
    args=parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 3600 or not 1 <= args.max_steps <= 120:
        parser.error("Use 1-3600 seconds and 1-120 steps per task.")
    private=ROOT/"data"
    private.mkdir(exist_ok=True)
    directory=args.output or Path(tempfile.mkdtemp(prefix="coding-evaluation-",dir=private))
    directory=directory.resolve()
    if not directory.is_relative_to(private.resolve()) or directory==private.resolve():
        parser.error("Reports must be inside the private data directory.")
    if args.output:
        directory.mkdir(parents=True,exist_ok=False)
    if args.validate_fixtures:
        results=[]
        for case in args.tasks:
            broken=validate(*prepare_case(case,directory/(case+"-broken")))
            reference=validate(*prepare_case(case,directory/(case+"-reference"),reference=True))
            results.append({"task":case,"broken_fails":broken["exit_code"]!=0,"reference_passes":reference["exit_code"]==0,"reference_validation":reference})
        print(json.dumps(results,ensure_ascii=False,indent=2))
        return 0 if all(item["broken_fails"] and item["reference_passes"] for item in results) else 1
    core=load_core(args.core_path)
    core.runtime.ensure_server()
    results=[]
    for case in args.tasks:
        results.append(evaluate(case,core,directory/case,model=args.model,think=args.think,max_steps=args.max_steps,max_seconds=args.max_seconds))
        (directory/"summary.json").write_text(json.dumps({"model":args.model,"think":args.think,"core_path":str(args.core_path) if args.core_path else None,"results":results},ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    print("Private report: "+str(directory/"summary.json"),flush=True)
    return 0 if all(item["passed"] for item in results) else 1


if __name__=="__main__":
    raise SystemExit(main())
