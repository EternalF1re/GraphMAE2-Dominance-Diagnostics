"""Non-runnable scientific source excerpts for inspection only.

Study-authored capture/geometry excerpts, under LICENSE-AUTHORS.
Scientific context: THUDM GraphMAE2, MIT; see LICENSE and THIRD_PARTY_NOTICES.md.
The original executed scientific member hashes are in the public provenance
record. Each literal below preserves its source text, including indentation.
No function, model import, asset loader, optimizer, or launcher executes here.

The caller captured three reconstruction criterion outputs and one bare latent
loss from a single native train-mode forward at each fixed state. Both groups
use local auxiliary coefficient 10. Full-Z VJPs are measured first; target
metrics then index ALL ordered target rows, retaining zeros. Routed ratios are
derived, not a new VJP. There are no parameter-gradient measurements here.
"""

CAPTURE_CARDINALITY_EXCERPT = r"""
        need(len(zbox)==1 and len(recs)==3 and len(latents)==1,'NATIVE_COMPONENT_CAPTURE')
"""

COMPONENT_VJP_EXCERPT = r"""
        z=zbox[0];before_vjp=state()
        budget.vjp();a=torch.autograd.grad(sum(recs),z,retain_graph=True)[0]
        budget.completed['vjps']+=1
        need(state()==before_vjp,'FIRST_VJP_PURITY')
        budget.vjp();b=torch.autograd.grad(10.*latents[0],z,retain_graph=True)[0]
        budget.completed['vjps']+=1
        need(state()==before_vjp and value_sha(model.state_dict())==before_weights,'SECOND_VJP_PURITY')
        nonroot=torch.ones(z.shape[0],dtype=torch.bool,device=z.device);nonroot[batch[1]]=False
        need(torch.count_nonzero(b[nonroot]).item()==0,'NON_TARGET_AUXILIARY_LEAKAGE')
        metrics=direction_metrics(a,b,scale)
        # Retain the frozen formal geometry's zero-norm => undefined behavior.
        # A zero target norm is not grounds to replace a batch or filter a seed.
        from secondary import target_metrics
        extra=target_metrics(a,b,batch[1],(metrics['reconstruction_gradient_norm'],metrics['weighted_aux_gradient_norm']))
"""

FULL_Z_METRICS_EXCERPT = r"""
def direction_metrics(rec,weighted_aux,scale):
    a=rec.detach().cpu().double();b=weighted_aux.detach().cpu().double()
    if a.shape!=b.shape or a.ndim!=2:raise RuntimeError('FULL_Z_SHAPE')
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise RuntimeError('NONFINITE_FULL_Z')
    na=float(torch.linalg.vector_norm(a));nb=float(torch.linalg.vector_norm(b))
    defined=na>0 and nb>0
    cosine=max(-1.,min(1.,float((a*b).sum())/(na*nb))) if defined else None
    ar=(a!=0).any(dim=1);br=(b!=0).any(dim=1)
    return dict(defined=defined,undefined_reason=None if defined else 'ZERO_COMPONENT_NORM',
                reconstruction_gradient_norm=na,weighted_aux_gradient_norm=nb,
                weighted_ratio=nb/na if na else None,
                routed_weighted_ratio_derived=scale*nb/na if na else None,
                routed_scale_derived=scale,cosine=cosine,
                angle_degrees=math.degrees(math.acos(cosine)) if defined else None,
                reconstruction_nonzero_row_fraction=float(ar.double().mean()),
                auxiliary_nonzero_row_fraction=float(br.double().mean()),
                shared_nonzero_row_fraction=float((ar&br).double().mean()),
                full_z_rows=a.shape[0],hidden_dimension=a.shape[1])
"""

ALL_TARGET_ROWS_METRICS_EXCERPT = r"""
def target_metrics(a,b,targets,full_norms=None):
    import torch
    need(a.shape==b.shape and a.ndim==2,'TARGET_FULL_Z_SHAPE')
    need(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'NONFINITE_COMPONENT')
    t=targets.detach().cpu()
    need(t.ndim==1 and t.dtype==torch.int64 and t.numel()>0 and
        t.unique().numel()==t.numel() and int(t.min())>=0 and int(t.max())<a.shape[0],'TARGET_IDS')
    ta=a.detach()[targets].cpu().double();tb=b.detach()[targets].cpu().double()
    if full_norms is None:
        na=float(torch.linalg.vector_norm(a.detach().cpu().double()));nb=float(torch.linalg.vector_norm(b.detach().cpu().double()))
    else:na,nb=full_norms
    need(math.isfinite(na) and math.isfinite(nb) and na>=0 and nb>=0,'NONFINITE_FULL_NORMS')
    nta=float(torch.linalg.vector_norm(ta));ntb=float(torch.linalg.vector_norm(tb))
    defined=nta>0 and ntb>0
    cosine=max(-1.,min(1.,float((ta*tb).sum())/(nta*ntb))) if defined else None
    ar=(a.detach()!=0).any(dim=1);br=(b.detach()!=0).any(dim=1)
    atr=(ta!=0).any(dim=1);btr=(tb!=0).any(dim=1)
    return dict(shared_target_cosine=cosine,target_defined=defined,
        target_undefined_reason=None if defined else 'ZERO_TARGET_COMPONENT_NORM',target_count=t.numel(),
        target_reconstruction_gradient_norm=nta,target_weighted_aux_gradient_norm=ntb,
        target_reconstruction_fullZ_norm_fraction=nta/na if na else None,
        target_aux_fullZ_norm_fraction=ntb/nb if nb else None,
        fullZ_reconstruction_nonzero_rows=int(ar.sum()),fullZ_aux_nonzero_rows=int(br.sum()),
        fullZ_shared_nonzero_rows=int((ar&br).sum()),target_reconstruction_nonzero_rows=int(atr.sum()),
        target_aux_nonzero_rows=int(btr.sum()),target_shared_nonzero_rows=int((atr&btr).sum()),
        secondary_status='PREDECLARED_EXPLORATORY_NOT_REPLACING_PRIMARY')
"""

