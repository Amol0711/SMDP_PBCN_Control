"""Synthetic contract tests. Mocked negative gates never authorize a real continuation."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pd_smac.runtime import ROOT,load,dump
from pd_smac.analysis import gate,operations as op
from pd_smac.analysis.continuing.contracts import load_protocol,debug_limit,execution_keys,verify_protocol_object,verify_production_authorization
from pd_smac.analysis.continuing.learner import initialize,advance,validate_state
from pd_smac.analysis.continuing.model import Plant
from pd_smac.analysis.continuing.io import save,restore,from_serial
from pd_smac.analysis.continuing.evaluator import evaluate_checkpoint

class ContinuingContracts(unittest.TestCase):
 def setUp(self):self.p=load_protocol();self.s=initialize(self.p,900);self.plant=Plant.from_protocol(self.p)
 def test_production_new_run_refused(self):
  with self.assertRaises(PermissionError):verify_production_authorization({})
 def test_debug_seed_rejects_scientific_seed(self):
  with self.assertRaises(ValueError):debug_limit(0,128)
 def test_debug_seed_ceiling(self):
  with self.assertRaises(ValueError):debug_limit(910,128)
 def test_debug_horizon_ceiling(self):
  with self.assertRaises(ValueError):debug_limit(900,2049)
 def test_debug_bool_rejected(self):
  with self.assertRaises(ValueError):debug_limit(True,1)
 def test_debug_keys_exact(self):self.assertEqual(execution_keys(900,'functional_test')['plant'],[20260919,212,10,3,900,300,0,0,22])
 def test_production_frozen_keys_exact(self):self.assertEqual(execution_keys(0,'production')['actor'],[20260919,212,21,3,0,300,0,0,0])
 def test_protocol_mutation_rejected(self):
  p=copy.deepcopy(self.p);p.config['actor']['bound']=11
  with self.assertRaises(ValueError):verify_protocol_object(p)
 def test_state_roundtrip(self):self.assertEqual(from_serial(self.s.serial(),self.p).identity(),self.s.identity())
 def test_state_rng_corruption(self):
  r=self.s.serial();r['offsets']['actor']=1
  with self.assertRaises(ValueError):from_serial(r,self.p)
 def test_state_seed_corruption(self):
  r=self.s.serial();r['seed_keys']['plant'][4]=901
  with self.assertRaises(ValueError):from_serial(r,self.p)
 def test_evaluation_noninvasive(self):
  before=self.s.serial();r=evaluate_checkpoint(self.s,self.plant,self.p);self.assertEqual(before,self.s.serial());self.assertEqual(r['training_rng_consumption'],0)
 def test_greedy_null_not_zero(self):
  r=evaluate_checkpoint(self.s,self.plant,self.p);self.assertIsNone(r['greedy']['residuals']['R_act']);self.assertIsInstance(r['stochastic']['residuals']['R_act'],float)
 def test_certificate_not_fresh_optimization(self):
  with tempfile.TemporaryDirectory()as d:
   r=op.witness(Path(d));self.assertFalse(r['fresh_optimization']);self.assertEqual(r['saved_active_support_certificates'],2)
 def test_dyadic_float_not_interchangeable(self):
  with self.assertRaises(AssertionError):op.compare_values({'a':None},{'a':0})
 def test_tolerance_rejects_nonfinite(self):
  with self.assertRaises(AssertionError):op.compare_values(float('nan'),0.)
 def test_structure_keys_must_match(self):
  with self.assertRaises(AssertionError):op.compare_values({'a':0},{'a':0,'b':0})
 def test_real_continuation_gate_has_no_override(self):
  import inspect
  self.assertNotIn('override',inspect.signature(gate.require_continuation).parameters)
 def test_negative_host_gate_stops_before_inputs(self):
  with patch.object(gate,'environment_check',return_value={'differences':{'cpu_model':{'actual':'different','expected':'recorded'}}}),patch.object(gate,'verify_external',side_effect=AssertionError('must not touch inputs'))as q:
   with self.assertRaises(PermissionError):gate.require_continuation('/missing','arbitrary',True)
   q.assert_not_called()
 def test_external_rights_not_assumed(self):
  with self.assertRaises(PermissionError):gate.verify_external('/missing')
 def test_external_companion_directory_refused(self):
  with self.assertRaises(ValueError):gate.verify_external(ROOT,True)
 def test_external_missing_state_refused(self):
  with tempfile.TemporaryDirectory()as d:
   with self.assertRaises(ValueError):gate.verify_external(d,True)
 def test_external_inventory_exact_scope(self):
  m=load(ROOT/'manifests/continuation_external.json');self.assertEqual(len(m['files']),113);self.assertEqual(sum(x.endswith('/state.json')for x in m['files']),36)
 def test_saved_policy_inventory(self):
  rows=load(ROOT/'results/diagnostics/continuing/index.json');self.assertEqual(len(rows),130);self.assertEqual(len({r['seed']for r in rows}),10)
 def test_current_plots_are_six_native_wrappers(self):
  names=load(ROOT/'figures/index.json');self.assertEqual(len(names),6)
  for n in names:self.assertIn('{standalone}',(ROOT/'figures'/(n+'.tex')).read_text())

class RestartContracts(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)/'state';self.p=load_protocol();self.s=initialize(self.p,900);self.plant=Plant.from_protocol(self.p);save(self.d,self.s,self.p)
 def test_snapshot_byte_identity(self):self.assertEqual(restore(self.d,self.p,900).serial(),self.s.serial())
 def test_snapshot_wrong_seed(self):
  with self.assertRaises(ValueError):restore(self.d,self.p,901)
 def test_snapshot_mutation(self):
  with(self.d/'state.json').open('a')as f:f.write(' ')
  with self.assertRaises(ValueError):restore(self.d,self.p,900)
 def test_snapshot_extra_file(self):
  (self.d/'extra').write_text('x')
  with self.assertRaises(ValueError):restore(self.d,self.p,900)
 def test_snapshot_overwrite(self):
  with self.assertRaises(FileExistsError):save(self.d,self.s,self.p)
 def test_production_snapshot_write_forbidden(self):
  self.s.mode='production'
  with self.assertRaises(PermissionError):save(self.d.parent/'production',self.s,self.p)
 def test_resume_complete_hold_identity(self):
  for _ in range(64):advance(self.s,self.plant,self.p)
  mid=self.d.parent/'mid';save(mid,self.s,self.p);other=restore(mid,self.p,900)
  for _ in range(64):
   a=advance(self.s,self.plant,self.p);b=advance(other,self.plant,self.p);np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
  self.assertEqual(self.s.serial(),other.serial())
if __name__=='__main__':unittest.main()
