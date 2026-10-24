"""Synthetic numerical and file-isolation tests; no biological input is needed."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pd_smac import runtime as rt
from pd_smac import statistics_core as st
from pd_smac import operations as op
from pd_smac.common.actor_geometry import project_zero_sum_box
from pd_smac.common.actor_scores import softmax_rows,tabular_score,clipped_logistic_audit
from pd_smac.common.actor_steps import multiplier_step
from pd_smac.common.actor_residuals import DiagnosticContext,episodic_diagnostics

class NumericalTests(unittest.TestCase):
    def test_projection_counterexample(self):np.testing.assert_allclose(project_zero_sum_box([3.,0.,0.],1),[1.,-.5,-.5],atol=1e-12)
    def test_projection_zero_sum(self):self.assertLess(abs(project_zero_sum_box(np.arange(15.).reshape(3,5),2).sum(1)).max(),1e-10)
    def test_projection_bounds(self):self.assertLessEqual(abs(project_zero_sum_box([1e3,-4,1],2)).max(),2)
    def test_projection_idempotence(self):
        w=project_zero_sum_box([[4.,9.,-2.],[2.,5.,6.]],3);np.testing.assert_allclose(project_zero_sum_box(w,3),w,atol=1e-10)
    def test_projection_nan_rejected(self):
        with self.assertRaises(ValueError):project_zero_sum_box([np.nan,0.])
    def test_projection_bound_rejected(self):
        with self.assertRaises(ValueError):project_zero_sum_box([0.,0.],0)
    def test_temperature_score_derivative(self):
        w=np.array([.2,-.3,.1]);temp=.4;eps=1e-6
        fd=np.array([(np.log(softmax_rows(w+eps*np.eye(3)[i],temp)[1])-np.log(softmax_rows(w-eps*np.eye(3)[i],temp)[1]))/(2*eps)for i in range(3)])
        np.testing.assert_allclose(tabular_score(w,1,temp),fd,atol=1e-8)
    def test_temperature_rejected(self):
        with self.assertRaises(ValueError):softmax_rows([0.,1.],0)
    def test_score_action_rejected(self):
        with self.assertRaises(ValueError):tabular_score([0.,1.],True)
    def test_softmax_shift_identity(self):np.testing.assert_allclose(softmax_rows([2.,1.]),softmax_rows([5.,4.]),atol=0)
    def test_logistic_saturation_not_exact_score(self):
        a=clipped_logistic_audit([40.],[1.],0);self.assertTrue(a['saturated']);np.testing.assert_array_equal(a['true_composite_score'],[0.]);self.assertNotEqual(a['inherited_surrogate_score'][0],0.)
    def test_multiplier_cap(self):self.assertEqual(multiplier_step(19.,4.,1.,20.),20.)
    def test_multiplier_floor(self):self.assertEqual(multiplier_step(.1,-1.,1.,20.),0.)
    def test_invalid_multiplier(self):
        with self.assertRaises(ValueError):multiplier_step(-1.,0.,1.,20.)
    def test_linear_quantile(self):np.testing.assert_allclose(st.linear_quantile([0.,2.,4.,6.],[.25,.75]),[1.5,4.5])
    def test_population_vs_sample_sd(self):self.assertAlmostEqual(st.summarize([0.,2.],ddof=1)['sd'],np.sqrt(2));self.assertEqual(st.summarize([0.,2.],ddof=0)['sd'],1.)
    def test_bootstrap_units_checked(self):
        with self.assertRaises(ValueError):st.summarize([1.,2.],np.array([[0,1,1]]))
    def test_bootstrap_indices_checked(self):
        with self.assertRaises(ValueError):st.summarize([1.,2.],np.array([[0,2]]))
    def test_constant_bootstrap(self):np.testing.assert_array_equal(st.summarize([3.,3.],np.array([[0,0],[1,1]]))['bootstrap95'],[3.,3.])
    def test_ratio_estimands_distinct(self):self.assertNotEqual(np.mean(np.array([1.,1.])/np.array([1.,9.])),2./10.)
    def test_horizon_ceiling(self):self.assertEqual(int(np.ceil(20/6)),4);self.assertEqual(20/np.ceil(20/6),5.)
    def test_null_not_zero(self):
        with self.assertRaises(AssertionError):op.matched({'a':None},{'a':0})
    def test_scalar_comparison_detects_mutation(self):
        with self.assertRaises(AssertionError):op.matched({'rate':.4},{'rate':.5})
    def test_stochastic_greedy_identity_separate(self):
        r=op.selected_records('four_state','all');self.assertEqual(len(r),80);self.assertEqual({x['mode']for x in r},{'stochastic','greedy'})
    def test_all_evaluation_records(self):self.assertEqual(len(op.selected_records('all','all')),676)
    def test_wrong_training_seed_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):op.train('four_state',Path(d),900,None,'full')
    def test_debug_seed_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):op.train('four_state',Path(d),0,None,'smoke')
    def test_wrong_four_state_method_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):op.train('four_state',Path(d),0,'Macro_Q','full')
    def test_partial_arbitrary_meanfield_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):op.meanfield(Path(d),3)
    def test_zero_evaluation_limit_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):op.evaluate_saved('four_state',Path(d),limit=0)

class FileIsolationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'companion';self.root.mkdir()
    def manifest(self):
        p=self.root/'x.txt';p.write_text('test');rt.dump(self.root/'MANIFEST_SHA256.json',{'files':{'x.txt':{'bytes':4,'sha256':rt.sha(p)}}})
    def test_manifest_valid(self):
        self.manifest()
        with patch.object(rt,'ROOT',self.root):self.assertEqual(rt.verify_package()['file_count'],2)
    def test_manifest_content_mutation(self):
        self.manifest();(self.root/'x.txt').write_text('fail')
        with patch.object(rt,'ROOT',self.root):
            with self.assertRaises(ValueError):rt.verify_package()
    def test_manifest_extra_file(self):
        self.manifest();(self.root/'extra').write_text('x')
        with patch.object(rt,'ROOT',self.root):
            with self.assertRaises(ValueError):rt.verify_package()
    def test_manifest_missing_file(self):
        self.manifest();(self.root/'x.txt').unlink()
        with patch.object(rt,'ROOT',self.root):
            with self.assertRaises(ValueError):rt.verify_package()
    def test_manifest_symlink(self):
        self.manifest();(self.root/'link').symlink_to(self.root/'x.txt')
        with patch.object(rt,'ROOT',self.root):
            with self.assertRaises(ValueError):rt.verify_package()
    def test_output_inside_companion_rejected(self):
        with patch.object(rt,'ROOT',self.root):
            with self.assertRaises(ValueError):rt.output_dir(self.root/'run')
    def test_output_overwrite_rejected(self):
        with patch.object(rt,'ROOT',self.root):
            with self.assertRaises(FileExistsError):rt.output_dir(Path(self.tmp.name))
    def test_new_external_output(self):
        with patch.object(rt,'ROOT',self.root):self.assertTrue(rt.output_dir(Path(self.tmp.name)/'run').is_dir())
    def test_input_acknowledgment_required(self):
        with self.assertRaises(PermissionError):rt.verify_inputs('melanoma28',Path(self.tmp.name),False)
    def test_input_missing_directory_argument(self):
        with self.assertRaises(ValueError):rt.verify_inputs('melanoma28',None,True)
    def test_input_inside_companion_rejected(self):
        with self.assertRaises(ValueError):rt.verify_inputs('melanoma28',rt.ROOT,True)
    def test_required_inputs_not_in_companion(self):
        for items in rt.load(rt.ROOT/'manifests/external_inputs.json')['studies'].values():
            for x in items:self.assertFalse((rt.ROOT/x['path']).exists())
    def test_import_all_checked_before_any_copy(self):
        dest=Path(self.tmp.name)/'imported'
        with self.assertRaises(ValueError):rt.import_inputs('melanoma28',Path(self.tmp.name),dest,True)
        self.assertFalse(dest.exists())
    def test_object_pickle_disabled_for_saved_arrays(self):
        for p in(rt.ROOT/'results').rglob('*.npz'):
            with np.load(p,allow_pickle=False)as a:
                for name in a.files:self.assertFalse(a[name].dtype.hasobject)


class PlotAndInputContractTests(unittest.TestCase):
    def test_active_plot_data_exactly_used(self):
        import re
        used=set()
        for p in(rt.ROOT/'figures').glob('*.tikz'):used.update(re.findall(r'\\FigureDataDir/([^}\s]+\.csv)',p.read_text()))
        self.assertEqual(len(used),29);self.assertEqual(used,{p.name for p in(rt.ROOT/'figures/data').glob('*.csv')})
    def test_six_standalone_wrappers(self):
        wrappers=[p for p in(rt.ROOT/'figures').glob('*.tex')if p.name!='preamble.tex']
        self.assertEqual(len(wrappers),6)
        for p in wrappers:self.assertIn('{standalone}',p.read_text())
    def test_learning_label_separation(self):self.assertEqual((rt.ROOT/'figures/melanoma28_learning.tikz').read_text().count('yshift=-1.00cm'),2)
    def test_biological_input_identities_preserved(self):
        spec=rt.load(rt.ROOT/'manifests/external_inputs.json')
        self.assertFalse(spec['bundled_inputs']);self.assertEqual(sum(len(x)for x in spec['studies'].values()),6)
    def test_input_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'companion';root.mkdir();ext=Path(d)/'local';(ext/'toy').mkdir(parents=True);f=ext/'toy/a.csv';f.write_text('x')
            rt.dump(root/'manifests/external_inputs.json',{'studies':{'melanoma28':[{'path':'toy/a.csv','bytes':1,'sha256':'0'*64}]}})
            with patch.object(rt,'ROOT',root):
                with self.assertRaises(ValueError):rt.verify_inputs('melanoma28',ext,True)
    def test_hash_verified_import(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'companion';root.mkdir();src=Path(d)/'source';src.mkdir();f=src/'a.csv';f.write_text('x');dest=Path(d)/'local'
            rt.dump(root/'manifests/external_inputs.json',{'studies':{'melanoma28':[{'path':'toy/a.csv','bytes':1,'sha256':rt.sha(f)}]}})
            with patch.object(rt,'ROOT',root):
                self.assertEqual(len(rt.import_inputs('melanoma28',src,dest,True)),1)
                with self.assertRaises(FileExistsError):rt.import_inputs('melanoma28',src,dest,True)

if __name__=='__main__':unittest.main()
