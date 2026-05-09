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

<div style="max-width: 600px;">
<table>
  <thead>
    <tr>
      <th>Term</th>
      <th>Name</th>
      <th>Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>P(θ)</td><td>Prior</td><td>Our belief about θ before seeing any data</td></tr>
    <tr><td>P(X | θ)</td><td>Likelihood</td><td>How probable is this data, given a particular value of θ?</td></tr>
    <tr><td>P(θ | X)</td><td>Posterior</td><td>Our updated belief about θ after seeing the data</td></tr>
    <tr><td>P(X)</td><td>Marginal Likelihood</td><td>A normalizing constant that ensures the posterior is a valid distribution</td></tr>
  </tbody>
</table>
</div>
