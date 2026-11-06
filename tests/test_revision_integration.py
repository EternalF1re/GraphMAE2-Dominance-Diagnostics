"""Publication mapping, CPU-only integration and vector PDF regressions."""
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]


class RevisionIntegrationTests(unittest.TestCase):
    def test_plot_outputs_cannot_overwrite_other_references_or_canonical_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary)/"candidate"
            shutil.copytree(ROOT,copied,ignore=shutil.ignore_patterns("reproduced","__pycache__",".git"))
            for number, script in [("1","plot_factorial_trajectory.py"),("5","plot_maskgae_dose_response.py")]:
                for target in [copied/"figures/figure_2/figure_2_reference.pdf",copied/"processed_records/factorial/checkpoint_validation.csv"]:
                    before = target.read_bytes()
                    run = subprocess.run([sys.executable,"-B",str(copied/"figures"/('figure_'+number)/script),
                                          "--output",str(target)],capture_output=True,text=True,timeout=30)
                    self.assertNotEqual(run.returncode,0,"Unsafe plot output was accepted")
                    self.assertEqual(target.read_bytes(),before)

    def test_approved_graph_ticks_and_mask_connecting_lines_are_retained(self):
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name,path)
            result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
            return result
        renderer = module("revision_style",ROOT/"figures/revision_render.py")
        factorial = module("factorial_style_data",ROOT/"code/analysis/factorial.py")
        maskgae = module("mask_style_data",ROOT/"code/analysis/maskgae_pubmed.py")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            renderer.configure_matplotlib(folder)
            import matplotlib.pyplot as plt
            with patch.object(plt,"close"):
                renderer.graph_figure(factorial.checkpoint_summary(factorial.read_records(ROOT)),folder,60)
                graph = plt.gcf()
                self.assertEqual(graph.axes[0].get_ylim(),graph.axes[1].get_ylim())
                self.assertTrue({"69","70","71","72"}.issubset({t.get_text() for t in graph.axes[1].get_yticklabels()}))
                groups = maskgae.analyze(maskgae.load_records(ROOT))["groups"]
                groups = [dict(r,epoch500_mean_pct=100*r["epoch500_mean"],epoch500_sd_pp=100*r["epoch500_sd"]) for r in groups]
                renderer.mask_figure(groups,folder,60)
                mask = plt.gcf()
                for ax in mask.axes:
                    self.assertEqual(ax.containers[0].lines[0].get_linestyle(),"-")
                    self.assertEqual(list(ax.texts),[])
            plt.close(graph); plt.close(mask)

    def test_legacy_assembler_help_never_opens_files_for_writing(self):
        original_open = Path.open
        def guarded_open(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"): raise AssertionError("--help attempted a write")
            return original_open(path, mode, *args, **kwargs)
        script = ROOT/"tables/assemble_current_method_record.py"
        with patch.object(sys,"argv",[str(script),"--help"]), patch.object(Path,"open",guarded_open), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped: runpy.run_path(str(script),run_name="__main__")
            self.assertEqual(stopped.exception.code,0)

    def test_six_current_figure_identifiers_and_all_paths_exist(self):
        mapping = json.loads((ROOT/"FIGURE_MAP.json").read_text())
        self.assertEqual([r["figure"] for r in mapping["current"]], ["1","2","3","4","5","D.1"])
        for row in mapping["current"]:
            for key in ["script","reference","input"]: self.assertTrue((ROOT/row[key]).is_file(),row[key])
        self.assertIn("factorial", mapping["current"][0]["script"])
        self.assertIn("maskgae", mapping["current"][4]["script"])

    def test_legacy_reference_bytes_and_superseded_figure_preserved(self):
        data = json.loads((ROOT/"figures/reference_provenance.json").read_text())
        for row in data["unchanged_legacy_figures"].values():
            self.assertEqual(hashlib.sha256((ROOT/row["relative_path"]).read_bytes()).hexdigest(),row["sha256"])
        old = ROOT/"figures/historical/three_seed_trajectory"
        for name in ["plot_validation_trajectory.py","figure_1_reference.pdf","input/validation_trajectory.csv"]:
            self.assertTrue((old/name).is_file())

    def test_new_pdfs_are_vector_with_embedded_fonts(self):
        for name in ["figure_1","figure_5"]:
            raw = (ROOT/"figures"/name/(name+"_reference.pdf")).read_bytes()
            self.assertFalse(re.search(rb"/Subtype\s*/Image\b",raw))
            self.assertIn(b"/FontFile2",raw); self.assertIn(b"/ToUnicode",raw)

    def test_figure_input_copies_equal_canonical_records(self):
        for number, source in [("1","factorial/checkpoint_validation.csv"),("5","maskgae_pubmed/checkpoints.csv")]:
            self.assertEqual((ROOT/"figures"/('figure_'+number)/"input/checkpoint_validation.csv").read_bytes(),
                             (ROOT/"processed_records"/source).read_bytes())

    def test_cpu_entry_has_no_model_imports_and_preserves_inputs(self):
        before = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ROOT/"processed_records").rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            code = ("import importlib.abc,sys,runpy; from pathlib import Path\n"
                    "class Block(importlib.abc.MetaPathFinder):\n"
                    " def find_spec(self,fullname,*a,**k):\n"
                    "  if fullname.split('.')[0] in ['torch','dgl','ogb','torch_geometric','wandb']: raise RuntimeError('ML import forbidden: '+fullname)\n"
                    "sys.meta_path.insert(0,Block())\n"
                    "sys.argv=[sys.argv[1],'--skip-figures','--output-dir',sys.argv[2]]\n"
                    "runpy.run_path(sys.argv[0],run_name='__main__')\n")
            out = Path(tmp)/"analysis"
            run = subprocess.run([sys.executable,"-B","-c",code,str(ROOT/"reproduce_analysis.py"),str(out)],
                                 capture_output=True,text=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('"verified_claims": 84',run.stdout)
            result = json.loads((out/"analysis_summary.json").read_text())
            self.assertEqual(result["estimator_reliability"]["execution_disposition"],"FAIL_FORMAL_EXECUTION_GATE")
            self.assertEqual(result["estimator_reliability"]["new_bootstrap_resamples"],0)
        after = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT/"processed_records").rglob("*") if p.is_file()}
        self.assertEqual(before,after)

    def test_source_outputs_cannot_overlap_inputs(self):
        spec = importlib.util.spec_from_file_location("revision_entry",ROOT/"reproduce_analysis.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with self.assertRaises(ValueError): module.run(ROOT/"processed_records"/"unsafe",figures=False)
