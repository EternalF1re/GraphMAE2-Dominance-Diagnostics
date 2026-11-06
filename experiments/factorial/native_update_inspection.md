# Native update path (inspection only)

The following method is copied from the frozen executed controller. Its injected native runtime/collator and private assets are intentionally not published here. The runnable routing/clipping primitives preserve their source formula. This is not a claim of one-command training replay.

```python
def step(self, position, run_identity):
        import torch
        import execute as prior
        from adapter import branch_context,clip_gradients,maybe_ema,maybe_scheduler
        started = time.perf_counter()
        batch, entry = prior.collate(self.loader,self.schedule,position,self.modules)
        if entry['global_update_zero_based'] != position or entry['epoch_index_zero_based'] != position//711:
            raise ValueError('schedule position drift')
        self.optimizer.zero_grad(set_to_none=True)
        with branch_context(self.model,run_identity['s']):
            rec,lat,_,metrics = self.modules['frozen'].component_forward(
                self.model,batch,torch.device(self.device),self.modules)
            self.modules['isolated'].validate_schedule_entry(entry,metrics)
            loss = rec + run_identity['lambda']*lat
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite loss')
            ema = maybe_ema(self.model,position//711)
            loss.backward()
        clipping = clip_gradients(self.model,run_identity['protocol'])
        self.optimizer.step()
        boundary = maybe_scheduler(self.scheduler,position+1,711)
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise FloatingPointError('nonfinite model state')
        if torch.device(self.device).type=='cuda':
            torch.cuda.synchronize(self.device)
        return dict(metrics,**clipping,global_update=position+1,ema_updated=ema,
                    scheduler_updated=boundary,learning_rate=self.optimizer.param_groups[0]['lr'],
                    update_seconds=time.perf_counter()-started,weighted_total_loss=float(loss.detach()),
                    validation_accessed=False,test_accessed=False)
```
