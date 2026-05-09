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
