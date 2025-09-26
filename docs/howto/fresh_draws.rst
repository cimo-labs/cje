Fresh Draws: Structure and Loading
==================================

Fresh draws are responses sampled from target policies (π′) and scored by the judge. They enable DR estimators.

Directory Layout
----------------

Minimal layout expected by the helpers:

.. code-block:: text

   responses/
     policy_a.jsonl
     policy_b.jsonl

Each line example:

.. code-block:: json

   {"prompt_id": "abc123", "target_policy": "policy_a", "judge_score": 0.86, "draw_idx": 0}

Loading in the API
------------------

- High‑level: pass ``fresh_draws_dir`` to ``analyze_dataset`` with ``estimator="stacked-dr"`` or ``auto``
- Manual: use loaders and add per‑policy datasets to the estimator

.. code-block:: python

   from cje.data.fresh_draws import load_fresh_draws_auto

   # Inside estimator setup
   for policy in sampler.target_policies:
       fd = load_fresh_draws_auto(Path("responses/"), policy, verbose=False)
       estimator.add_fresh_draws(policy, fd)

Best Practices
--------------

- Use consistent ``prompt_id`` across logged data and fresh draws
- Keep ``draw_idx`` contiguous per prompt (0..k-1)
- Store judge scores for fresh draws; responses are optional but helpful for audits

Troubleshooting
---------------

- Missing policies or prompts: verify policy names and prompt IDs match the dataset
- Inconsistent draws per prompt: allowed, but a warning is emitted; prefer consistent k

See also
--------

- DR quickstart: :doc:`../tutorials/dr_quickstart`
- Estimator selection: :doc:`choose_estimator`
