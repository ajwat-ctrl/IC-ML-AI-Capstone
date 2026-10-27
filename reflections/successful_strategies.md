# BBO Capstone: Successful Strategies Reflection

## My strongest strategies and why they worked

The main gain came from choosing queries with a Gaussian process surrogate and an acquisition function. A fitted model turns every past result into a prediction with an uncertainty estimate, so each query is chosen deliberately instead of by guesswork.

The second gain came from treating the eight functions differently. The flat Function 1 used the largest exploration weight (kappa 3.0), while Function 8 used one of the smallest (kappa 1.5). From round 2 onward each function had its own settings: UCB for six functions and Expected Improvement for two, with different output transforms, kernel smoothness and noise handling. Function 5 shows the payoff most clearly: its result rose from 150 in round 1 to 1,581 in round 2 and 6,065 in round 3 as queries moved towards a high corner of the input space.

These choices shaped the whole challenge:

- Rounds 2 to 4: keep each function's settings fixed (UCB with kappa 1.5 to 3.0, or EI with xi 0.01), so changes in the queries come from new data.
- Next: consider lowering the exploration weight in later rounds, because a wasted query costs more near the end than at the start.
- Throughout: log every query, result and rationale, which stopped me repeating myself or chasing noise.

## What makes a strategy successful

Outcomes matter, but they are not enough on their own, for three reasons:

- **Noise and luck.** With one query per function per round, a good result can be chance. A strategy that scored well by luck is not necessarily a good one.
- **Adaptability.** The best strategies changed in response to evidence, for example by cutting exploration when returns diminished or abandoning an approach that stopped working.
- **Reasoning and efficiency.** A strategy whose logic I can explain, and which gets results from few queries, is one I can reuse. If I can't say why it worked, I can't repeat it.

I therefore judge success by outcome, quality of reasoning, adaptability and results per query together. A failed experiment that taught me something is still part of a successful process, as the peer example below shows.

## Applying these strategies in professional ML/AI work

The same approach carries over wherever evaluations are expensive and few.

- **Hyperparameter tuning and AutoML.** Each training run is costly, so surrogate-based search (Gaussian process or tree-based Bayesian optimisation, as in tools like Optuna) applies directly.
- **Experiment-limited settings.** A/B testing, simulation-based design and lab experiments share the one-query-at-a-time structure of this task.
- **Process habits.** Logging every experiment with its rationale, treating exploration as a shrinking budget, and tailoring the method to each problem are useful in any ML project. For example, tuning a model under a fixed compute budget follows the same loop: fit a surrogate to past runs, then choose the next run by expected gain.

## Reflecting on a peer's approach

I read a colleague's retrospective (repository: slilienstroem/imperial-ml-capstone). Their pipeline is more sophisticated than mine.

### What worked well and why

- **GP plus SVM spatial filter.** They trained an SVM on historical data binarised at the 75th percentile and used it to prune unpromising regions before the acquisition step. This concentrates a limited query budget on likely-good regions, and they named it one of their two highest-impact changes.
- **ARD kernels for high-dimensional functions.** On Function 7 (6D) and Function 8 (8D) an anisotropic kernel produced a breakthrough on Function 7 in round 8. Their lesson: the true gradient path is often low-dimensional even when the input space is not, which echoes the idea behind PCA.
- **Noise regularisation on stochastic functions.** Raising the GP noise term on Functions 2 and 6 stopped the search chasing transient fluctuations.
- **Honest reporting of a failure.** A sign-inversion experiment on Function 4 in round 10 made the model diverge. They rolled it back, reactivated the SVM filter and recovered over the following rounds.
- **A decaying exploration weight**, linked to decaying epsilon-greedy in reinforcement learning.

### Overlap with my approach

We both used a GP surrogate with a UCB or EI acquisition and tuned strategy per function, with different acquisition functions, output transforms and noise settings for each. They went further in structure, with the SVM filter and explicit per-dimension relevance analysis.

### Suggestions and questions

1. **Stability of the SVM filter.** With roughly 13 to 20 points per function, a classifier trained on data binarised at the 75th percentile has very few positive examples. How did they check it never pruned the true optimum? A safeguard would be to always keep a small share of candidates from outside the filter.
2. **The epsilon-greedy analogy.** UCB's exploration weight is a bonus on posterior uncertainty, not a probability of acting randomly. It is closer to a shrinking optimism term, so the analogy captures the decay but not the mechanism.
3. **Testing bold ideas more cheaply.** The sign-inversion test cost a real query on Function 4. Checking held-out prediction error under the inverted assumption first might have exposed the problem for free.
4. **A public query log.** A CSV of every query and result in the repository would let others reproduce and learn from the work.

### How this broadens my view of success

Their post reinforced that success is more than the final score. It includes the quality of the reasoning, the willingness to test ideas that might fail, and honesty in reporting what happened. The failed experiment still taught them something about the space. It also showed how much structure (filters, per-dimension relevance, explicit noise handling) can improve sample efficiency when queries are scarce, which I would like to try next time.
