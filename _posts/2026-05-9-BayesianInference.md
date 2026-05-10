---
title: "(WIP) Rethinking A/B Testing with Bayesian Inference"
date: 1999-05-09
tags: [Applied Statistics, Python]
excerpt: "Explore Bayesian A/B testing as a principled alternative to traditional frequentist inference"
---

## Background
Running an A/B test sounds straightforward. Split your test data, measure an outcome, and pick a winner. Yet assumptions in the most common A/B tests can break down in ways that are easy to miss and costly to ignore. Luckily, there are alternative approaches that treat belief as something to be updated as evidence arrives, rather than a mere threshold to be crossed. In this post, we'll build our knowledge on the Bayesian statistical framework and examine what it really costs to get decisions wrong.

## The Issue with Frequentist Probability
The **Frequentist** definition of probability is the most widely known in statistics. It states that probability measures the long-run frequency of an event occurring over many repeated trials. A classic example is the coin flip. Flipping a coin millions of times, the proportion of heads gradually stabilizes near 0.5 as an estimate of the true underlying probability. That true probability is always unknown, but with enough repeated trials, we can get reasonably close. 

**Frequentist A/B Testing** is an experiment we run to see if two processes behave significantly differently from one another. Consider a pharmaceutical company testing a new treatment to relieve headaches. Patients are randomly assigned to receive treatment A (an existing medication) or treatment B (the new medication). After exposure to the treatment, each patient reports whether their headache was relieved within one hour. The goal of this test is to determine whether treatment B produces a meaningfully higher relief rate than treatment A. The null hypothesis for this A/B test is: Treatments A and B produce identical relief rates. We’d then define a fixed number of patients, each receiving exactly one of the two treatments. Then, we’d use the test results to compute a single p-value answering the question: What’s the probability of observing data as extreme as ours if the null hypothesis is true? If this probability is very unlikely (< 0.05), we’d reject the null hypothesis and claim that the two treatments produce meaningfully different relief rates, resulting in a winner.

Key takeaways on the Frequentist A/B approach:
- The p-value from this test is the probability of observing our collected data given the null hypothesis is true.
- Frequentist probability treats the true relief rates of treatments A and B as fixed, unknown constants (probabilities that do not change). 
- The false positive rate (0.05) is only controlled in the test if we commit to our sample size in advance and look at the results exactly once.

Let’s say we decide a reasonable sample size for this experiment to be 500 patients. As we carry out this experiment, stakeholders behind the scenes are eager to monitor the results in real time. After the first 50 sessions, the resulting p-value dips to 0.03, indicating we should reject the null hypothesis and favor treatment B. The issue is, every time we glance at the accumulating results before reaching our defined sample size of 500, we risk rejecting the null hypothesis by chance alone. This is known as p-hacking by peeking, which can actually inflate the 0.05 false positive rate we thought we were controlling. Research suggests that peeking at live results just ten times can balloon a 0.05 false positive rate to upwards of 0.25  [(Miller, 2010)](https://www.evanmiller.org/how-not-to-run-an-ab-test.html). So why peek at all? Interim results can be tempting to those who have incentive to act on promising data early, even when the data isn’t fully conclusive. Frequentist A/B testing is meant for a world where we resist that pressure, but that’s not always the case in practice.

After 100 sessions, treatment B has a relief rate of 68% and treatment A has a relief rate of 61%. Is treatment B truly better yet? If we’re playing by the frequentist rules, we’re in a tough spot. We’ve been monitoring the number of collected sessions, but we still haven’t reached our pre-defined sample size of 500 (and frankly, we're questioning whether 100 independent patient sessions could have been a sufficient sample size to begin with). At this point, the p-value still cannot tell us: Given the data, how confident should we be that treatment B is genuinely better? Lucky for us, Bayesian inference can help us answer this question.

## What is Bayesian Statistics?
The Bayesian definition of probability is a measurement of belief. It represents a value of uncertainty about an event, whether it happens once or many times. Our belief in the outcome of an event can change as we observe new evidence. Instead of committing to a single fixed estimate, we maintain a full distribution of beliefs across all possible parameter values, updating these beliefs as new evidence arrives.

Let’s revisit the coin flip for a moment. Under frequentist thinking, we flip a coin millions of times, and the probability of landing on heads will converge to a fixed value around 0.5. With Bayesian probability, we start with a belief about the coin's bias, like “I expect the true probability of landing on heads to fall between 0.4 and 0.6.” We might suspect the coin is fair, but we’re not certain. Well, we can revise that belief with every flip. After 10 flips, our distribution is wide and uncertain. After 10,000 flips, it has narrowed considerably. While both Frequentist and Bayesian estimates in this example might lead to the same result, the Bayesian approach allows us to carry a full picture of our estimates and uncertainty along the way. 

We’ll use this same approach now for the clinical drug experiment. Instead of asking whether the relief rate of treatment B is significantly different than treatment A at some fixed sample size, we’ll construct a distribution of beliefs around both treatments’ true relief rates, updating both in real time as patient sessions are recorded. To understand how that updating works, we need Bayes' Theorem. Let’s recall our favorite conditional probability formula:


$$
\small P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$


In our experiment, we’re not talking about abstract events A and B. We care about parameters and data. Let’s rewrite this formula as:


$$
\small P(\theta \mid X) = \frac{P(X \mid \theta) \cdot P(\theta)}{P(X)}
$$


Where θ is the unknown parameter we want to learn (the true relief rate for a particular treatment) and X is the observed data (the recorded outcomes of relieved and not relieved from real patient sessions).

Next, we need to define a few new terms:


| Term | Name | Definition |
|------|------|---------------|
| $$P(\theta)$$ | Prior | Our belief about θ before seeing any data |
| $$P(X \mid \theta)$$ | Likelihood | How probable is this data, given a particular value of θ? |
| $$P(\theta \mid X)$$ | Posterior | Our updated belief about θ after seeing the data |
| $$P(X)$$ | Marginal Likelihood | A normalizing constant that ensures the posterior is a valid distribution |

The posterior is always what we're after. It reflects our updated belief about θ given everything observed so far.

Let's put this knowledge to the test with a famous example in Bayesian probability.

---
_Consider a disease that affects 1% of the population. A diagnostic test for this disease is 95% accurate: if you have the disease, the test returns positive 95% of the time. If you don’t have the disease, the test returns negative 95% of the time (a 5% false positive rate). You test positive. What is the probability you actually have the disease?_

Here's how we go about answering this question using the formulas above.

$$
\small
\begin{aligned}
P(\theta) &= 0.01 \quad \text{(prior: 1% of the population has the disease)} \\
P(X \mid \theta) &= 0.95 \quad \text{(likelihood: test returns positive 95% of the time given disease is present)}
\end{aligned}
$$

Expanding $P(X)$ using the law of total probability:

$$
\small
\begin{aligned}
P(X) &= P(X \mid \theta)P(\theta) + P(X \mid \neg\theta)P(\neg\theta) \\
&= (0.95)(0.01) + (0.05)(0.99) = 0.0095 + 0.0495 = 0.059
\end{aligned}
$$

Drumroll...

$$
\small P(\theta \mid X) = \frac{0.95 \times 0.01}{0.059} \approx 0.161
$$

There’s actually only a 16% chance you have the disease despite a 95% accurate test. Many people (my initial self included) assume the answer is closer to 95%. The disease is so rare that false positives outnumber true positives across the full population. Our prior belief (only 1% of people have this disease) is powerful enough to override a test that is 95% accurate. Thus, the posterior is a balance between prior belief and observed evidence. If you were to test positive twice in a row, that posterior of 16% becomes your new prior going into the next update. Each result reshapes your belief, which becomes the starting point for the next posterior.

---
We’ll now transition back to the clinical drug experiment.

<div style="background-color: #f5f5f5; border-left: 4px solid #ccc; padding: 12px 16px; margin: 16px 0;">
<strong>A side note on the marginal likelihood P(X)</strong><br><br>
Notice P(X) sits in the denominator of the posterior formula above. Its only job is to ensure the posterior sums to 1 across all possible values of θ. It has no dependence on θ and is a constant regardless of the parameter value we're evaluating. Since we'll be focused on identifying the shape of the posterior distribution, a constant that scales everything uniformly doesn't meaningfully change anything. We can drop the denominator and write:


$$
\small P(\theta \mid X) \propto P(X \mid \theta) \cdot P(\theta)
$$

<p style="text-align: center; font-size: 0.85em;">Where ∝ means "proportional to"</p>


Again, the shape of the posterior is determined entirely by the numerator, so we'll refer to this formula from here on out.
</div>

Recall Frequentist probability views the true relief rate of treatment B as a fixed, unknown constant. We can only estimate this true relief rate by collecting data, running a test, and arriving at a single value like 68%. We can add a confidence interval around our estimate, but we’re ultimately left with a single estimate and a range. There's no formal way for us to say we believe the true rate is probably around 68% with a reasonable chance it's as high as 75%. We can’t update that estimate fluidly as new sessions arrive either. The Bayesian play is to represent θ (the true relief rate) as a probability distribution. Instead of committing to a single estimate, we capture a full distribution of our uncertainty across every possible value between 0 and 1. The mean of the distribution is where we think the true θ value most likely falls. The width of the distribution reflects how certain or uncertain we are. Whenever new data arrives and we update our prior, the entire distribution updates.

A Beta Distribution $\text{Beta}(\alpha, \beta)$ is a common probability distribution in Bayesian statistics defined on the interval [0,1]. The distribution uses two parameters, α (alpha) and β (beta), to represent pseudo-counts of successes and failures measuring θ. We can use this distribution to model our prior belief of θ. For example, think of α as the number of relieved outcomes (successes) and β as the number of non-relieved outcomes (failures) before the clinical drug experiment begins. A $\text{Beta}(1, 1)$ prior represents starting as if we’ve seen exactly one success and one failure (essentially no prior bias). A $\text{Beta}(10, 30)$ represents having already observed 10 successes from 40 sessions, suggesting a prior belief of roughly 25% relief rate with moderate confidence.

If we have no prior beliefs on the true relief rates of treatments A or B at the start of the A/B test, we can use the neutral $\text{Beta}(1, 1)$ as the prior for both treatments. This is a flat, uninformative prior telling the model we have no strong belief about the effectiveness of either treatment before the experiment. After 500 sessions are collected for each treatment, each success and failure to relieve a headache would update the corresponding Beta posterior distributions (one for $\theta_A$, one for $\theta_B$). We’d then ask what the probability is that a sample drawn from B's posterior exceeds a sample drawn from A's posterior. The probability density function for the Beta distribution is:

$$
\small f(\theta \mid \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

Where $\text{Beta}(\alpha, \beta)$ is the normalizing constant ensuring the distribution integrates to 1 over [0,1]. This is analogous to P(X) in Bayes' theorem. When we plot the Beta distribution, θ sits on the x-axis. We’re never claiming we know θ. Instead, we're asking how likely each possible value of θ is given our current beliefs. We’ll take a look at a few of these plots below. Try thinking of them as pictures of uncertainty around the true value of θ.

Here are some examples of a prior $\text{Beta}(\alpha, \beta)$ distribution for the relief rate of treatment B. The peak location is always $\alpha/(\alpha+\beta)$. The width tells us how much volatility is behind that belief.

**Beta(1, 1) — An uninformative prior** <br>
<img src="{{ site.url }}{{ site.baseurl }}/images/BayesianInference/Beta(1,1).png" width="20%"> <br>
This is a completely flat distribution. Every possible relief rate from 0 to 1 is equally likely. We're essentially expressing no prior belief about what to expect. <br>
<br>
**Beta(13, 7) — A weak prior belief** <br>
<img src="{{ site.url }}{{ site.baseurl }}/images/BayesianInference/Beta(13,7).png" width="20%"> <br>
Equivalent to observing 13 relieved outcomes from 20 sessions. Soft belief that treatment B's relief rate is around 65%. <br>
<br>
**Beta(204, 96) — A strong prior belief** <br>
<img src="{{ site.url }}{{ site.baseurl }}/images/BayesianInference/Beta(204,96).png" width="20%"> <br>
Equivalent to 204 relieved outcomes from 300 sessions. High confidence that treatment B's relief rate is near 68%. <br>
<br>

Now, here's where the math gets exciting. If we pair the right prior $P(\theta)$ with the right likelihood $P(X \mid \theta)$, the posterior comes out as the same family of distributions as the prior. This is called conjugacy, and it's what makes our update rule clean enough to derive by hand. In the clinical drug experiment, our likelihood comes from a binomial distribution (relieved or not relieved being the binary outcome) and the prior comes from a $\text{Beta}(\alpha, \beta)$ distribution (defined on the closed $[0,1]$ interval for probability). Together, they form a Beta-Binomial conjugate pair, which can be used to update the prior.
