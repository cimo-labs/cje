# Choose the human outcome before combining raters

If the target is the mean vote of a fixed three-person panel, give each sampled response one label equal to the mean of its three preplanned votes. If the target is passing a majority decision, use the majority decision as that label. For votes [1, 1, 0] the two labels are 2/3 and 1. Use the same rater assignment design and aggregation rule on both policies. Pass the response once and keep its prompt ID for paired inference.

Prompt-level uncertainty applies to sampling new prompts with this labeling protocol. When the same raters work across prompts and the target includes variation across new raters, inference must account for the rater assignment as well. Requesting a third opinion only on disputed responses also changes the sampling protocol. Declare what initial votes, tie breakers and skipped ratings mean in the estimand.

A model of latent rater quality defines another outcome. For external posterior samples, compute each policy difference within the same **joint draw** from the model, keeping draw IDs aligned across prompts and policies. Repeated draws of one latent label still describe one response. Report the model, prior, source of rater data and how prompt sampling was handled.

## Keep unscored responses in the audit frame

A missing judge score and a missing human outcome require different treatment. Keep the originally assigned units in the denominator. Human reviews of unscored responses can estimate their contribution. If being blocked or cancelled defines a failure in the intended outcome, specify that rule before looking at assignments and record the failure directly.

If the actual outcome is missing and lies in [0,1], let q be the fraction of the target mass with an observed human label and m its observed mean. Its full-frame mean lies in [q m, q m + (1−q)]. For example, with 80 of 100 equal-weight outcomes observed and 60 successes, the bounds are [.60, .80]. For a policy difference, subtract the baseline upper bound from the candidate lower bound for the lower limit, and subtract the baseline lower from the candidate upper for the upper limit. These fixed-frame identification bounds use the declared endpoint and target weights. Sampling units from a wider population adds a separate source of uncertainty.

For paired documents with known probabilities of entering traffic and then receiving human review, the [weighted paired audit](weighted-paired-audit.md) exposes its observed-outcome bounds and its sampling interval separately. Lee trimming addresses the treatment effect among units whose outcomes would be observed under either arm. It requires a justified monotone effect on observation and an appropriate assignment design. Different aggregate missingness rates alone do not establish that premise. [Lee's selection bounds](https://doi.org/10.1111/j.1467-937X.2009.00536.x)
