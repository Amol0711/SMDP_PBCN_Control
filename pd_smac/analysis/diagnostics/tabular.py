from ...common.actor_geometry import actor_mapping, require_domain, PROJECTION_ID
from ...common.actor_scores import softmax_rows
from ...common.actor_residuals import DiagnosticContext, array_hash, evaluate_tabular_smdp, continuing_diagnostics

from ...runtime import ROOT
COMMON_PARENT = ROOT / "pd_smac"
