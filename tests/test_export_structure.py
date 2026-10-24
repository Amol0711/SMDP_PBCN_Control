"""Standard-library export fixtures. No scientific models or numerical packages run."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pd_smac import runtime
from pd_smac.analysis import gate
from pd_smac import cli

class ExportStructure(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'companion';self.root.mkdir()
        self.f=self.root/'payload.json';self.f.write_text('{"value":1}\n')
        self.record={'files':{'payload.json':{'bytes':self.f.stat().st_size,'sha256':hashlib.sha256(self.f.read_bytes()).hexdigest()}}}
        (self.root/'MANIFEST_SHA256.json').write_text(json.dumps(self.record))
    def verify(self):
        with patch.object(runtime,'ROOT',self.root):return runtime.verify_package()
    def test_01_valid_manifest(self):self.assertEqual(self.verify()['file_count'],2)
    def test_02_altered_payload_rejected(self):
        self.f.write_text('{"value":2}\n')
        with self.assertRaises(ValueError):self.verify()
    def test_03_missing_payload_rejected(self):
        self.f.unlink()
        with self.assertRaises(ValueError):self.verify()
    def test_04_extra_payload_rejected(self):
        (self.root/'extra').write_text('unexpected')
        with self.assertRaises(ValueError):self.verify()
    def test_05_symlink_rejected(self):
        (self.root/'alias').symlink_to(self.f)
        with self.assertRaises(ValueError):self.verify()
    def test_06_unsafe_manifest_rejected(self):
        for name in ['../outside','/absolute','a\\b','C:escape','a/../escape']:
            with self.subTest(name=name):
                (self.root/'MANIFEST_SHA256.json').write_text(json.dumps({'files':{name:self.record['files']['payload.json']}}))
                with self.assertRaises(ValueError):self.verify()
    def test_07_git_metadata_not_scientific_payload(self):
        (self.root/'.git').mkdir();(self.root/'.git/HEAD').write_text('ref: refs/heads/main')
        self.assertEqual(self.verify()['status'],'passed')
    def test_08_output_inside_payload_rejected(self):
        with patch.object(runtime,'ROOT',self.root):
            with self.assertRaises(ValueError):runtime.output_dir(self.root/'new')
    def test_09_existing_output_rejected(self):
        with patch.object(runtime,'ROOT',self.root):
            with self.assertRaises(FileExistsError):runtime.output_dir(Path(self.tmp.name))
    def test_10_external_new_output(self):
        q=Path(self.tmp.name)/'new-output'
        with patch.object(runtime,'ROOT',self.root):self.assertEqual(runtime.output_dir(q),q)
    def test_11_mock_environment_mismatch_stops_before_inputs(self):
        with patch.object(gate,'environment_check',return_value={'differences':{'kernel':'fixture'},'status':'blocked_environment'}),patch.object(gate,'verify_external') as verify:
            with self.assertRaises(PermissionError):gate.require_continuation('/unread','not-a-run',False)
            verify.assert_not_called()
    def test_12_guard_first_statement(self):
        tree=ast.parse((ROOT/'pd_smac/analysis/gate.py').read_text())
        f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='require_continuation')
        self.assertIsInstance(f.body[0],ast.Assign);self.assertEqual(f.body[0].value.func.id,'require_environment')
    def test_13_python_sources_parse(self):
        for p in ROOT.rglob('*.py'):
            if '.git' not in p.relative_to(ROOT).parts:ast.parse(p.read_text(),filename=str(p))
    def test_14_all_output_routes_resolve(self):
        data=json.loads((ROOT/'manifests/paper_outputs.json').read_text())['outputs']
        self.assertEqual(len(data),23);self.assertEqual(len({x['id'] for x in data}),23)
        for row in data:
            for n in row['export_files']:
                p=(ROOT/n).resolve();self.assertTrue(p.is_relative_to(ROOT));self.assertTrue(p.is_file(),n)
    def test_15_current_six_plots(self):
        names=json.loads((ROOT/'figures/index.json').read_text());self.assertEqual(len(names),6)
        for name in names:
            for ext in ('.tex','.tikz'):self.assertTrue((ROOT/'figures'/(name+ext)).is_file())
        self.assertEqual(len(list((ROOT/'figures/data').glob('*.csv'))),29)
    def test_16_study_paths_resolve(self):
        data=json.loads((ROOT/'manifests/studies.json').read_text())['studies'];self.assertEqual(len(data),7)
        for row in data:
            for k in ['configuration','implementation','saved_results','guide']:self.assertTrue((ROOT/row[k]).exists(),row[k])
    def test_17_external_inputs_not_bundled(self):
        d=json.loads((ROOT/'manifests/external_inputs.json').read_text());self.assertFalse(d['bundled_inputs'])
        for group in d['studies'].values():
            for item in group:self.assertFalse(list(ROOT.rglob(Path(item['path']).name)))
    def test_18_permission_checks_are_retained(self):
        with self.assertRaises(PermissionError):runtime.verify_inputs('melanoma28',None,False)
        with self.assertRaises(PermissionError):gate.verify_external('/unread',False)
    def test_19_cli_no_authorization_override(self):
        sub=next(x for x in cli.parser()._actions if hasattr(x,'choices') and isinstance(x.choices,dict))
        for p in sub.choices.values():
            self.assertFalse(any('override' in o or 'bypass' in o for a in p._actions for o in a.option_strings))
    def test_20_no_model_import_for_parser(self):
        # Test in a separate interpreter so numerical-suite discovery cannot
        # contaminate this standard-library-only import check.
        code="import sys;sys.dont_write_bytecode=True;sys.path.insert(0, "+repr(str(ROOT))+");from pd_smac.cli import parser;parser();assert not any(n in sys.modules for n in ['numpy','torch','pd_smac.operations','pd_smac.analysis.biological.extension','pd_smac.analysis.continuing.model'])"
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_21_publication_and_license_not_granted(self):
        r=json.loads((ROOT/'manifests/rights_scope.json').read_text())
        self.assertFalse(r['public_release_authorized']);self.assertFalse(r['project_license_selected'])
    def test_22_manifest_transparency(self):
        r=self.verify();self.assertFalse(r['public_release']);self.assertIn('not numerical reproduction',r['scope'])

if __name__=='__main__':
    unittest.main(verbosity=2)
