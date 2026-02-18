+++
title = 'Prompting as Bayesian Inference'
date = 2025-11-18
draft = false
tags = ['ai', 'prompting', 'bayesian', 'intuitus']
math = true

[params]
    nextprev = true
    taglist = true
+++

**You are probably prompting wrong.**

If you write one giant message, press enter, and judge the model by that single reply, you are using a Bayesian inference engine as if it were a static form filler. You give it one chance to guess your task, build a model of it, and solve it, all from a muddled prior you compressed into a single prompt.

A large language model does not see your text as one indivisible request. It sees a stream of tokens that look like data from some unknown process. Internally, it starts from a prior encoded in its weights, then uses each token you send, and each token it generates, as evidence. In Bayesian terms, you control both the effective prior and the likelihood. If you insist on one-shot interaction, you are betting everything on your initial prior being right. If that prior is wrong, there is no useful posterior, only a confident answer to the wrong question.

The real theme of this piece is not "multi turn good". It is: you should provide information that moves the model's internal state toward the posterior you care about. Multi turn interfaces, tool calls, system prompts, and RLVR post-training are all mechanisms for doing exactly that.

## 1. Bayes theorem, stated directly for LLMs

Bayes theorem in its basic form is:

$$P(h | D) \propto P(D | h) P(h)$$

where:
- $h$ is a hypothesis
- $D$ is observed data
- $P(h)$ is the prior belief over hypotheses
- $P(D | h)$ is the likelihood of seeing $D$ if $h$ were true
- $P(h | D)$ is the posterior, the updated belief after seeing $D$

For a language model, this is not an analogy, you can map the terms almost literally:

- The hypothesis $h$ is "what task is this" and "what local rules and world model apply here".
- The data $D$ is the sequence of tokens in the prompt and conversation so far.
- The prior $P(h)$ is encoded in the trained weights, which reflect which tasks and worlds were common in pretraining.
- The likelihood $P(D | h)$ is how likely that token sequence is if this really is the task and world model ($h$).
- The posterior $P(h | D)$ is the internal state after the model has read your tokens and its own intermediate outputs.

Take a simple example. The model reads:

> "Translate this sentence into French: The sky is blue."

There are many possible hypotheses, but two obvious ones are:
- $h_1$: the user wants a translation into French
- $h_2$: the user wants an explanation about the color of the sky

Before any tokens, the prior over tasks is broad. As soon as it reads "Translate this sentence into French", the likelihood under $h_1$ increases sharply and the likelihood under most other hypotheses drops, so the posterior mass moves toward "translation task". By the time it has seen "The sky is blue", the posterior over tasks is sharply concentrated on $h_1$, so the next token distribution peaks near "Le".

That is Bayes theorem playing out inside the model, implemented by a neural network rather than symbolic algebra. You can think of the internal activations just after the prompt as an implicit posterior over tasks. It is implicit, stored as activations, not as an explicit table of hypotheses, but functionally it plays the same role.

Your job as a user or system designer is not "send one perfect prompt". It is to choose and sequence information so that this internal posterior moves toward the part of hypothesis space that matches what you actually care about.

## 2. Counterfactual physics and local posteriors

Consider again the counterfactual physics example. You tell the model:

> "In this universe, heavy objects move upward when released, light objects move downward, and water boils at minus ten degrees Celsius."

Then you ask:

1. "If I drop an anvil and a feather at the same time, what happens"
2. "If I heat water to minus fifteen degrees, what happens"

A competent model answers that the anvil rises, the feather falls, and the water at minus fifteen degrees is still liquid because it is not yet at its boiling point.

Before the prompt, the prior in the weights strongly favors normal physics, with gravity pulling objects down and water boiling near one hundred degrees. The prompt is strong evidence for a local hypothesis that overrides those defaults inside this context. The model behaves as if it had updated to a posterior where the local rules hold, and it makes predictions that are consistent with that posterior rather than with the global default prior.

This is exactly what a Bayesian agent would do. It keeps its global prior, but for this specific data sequence it gives high probability to the hypothesis "we are in a weird world where heavy objects go up, light objects go down, water boils at minus ten", and it answers accordingly.

For the purposes of this article, that is enough. We do not need to know whether the model literally stores a distribution over explicit physics laws. We only need to see that its input output behavior matches the Bayesian pattern: prior, evidence, posterior, prediction.

## 3. Why Bayes appears in the objective at all

The Bayesian connection is not an accident. It comes straight from the training objective.

Language models are trained to minimize cross entropy loss on a corpus of text. For each token, the loss is minus the log probability the model assigned to the correct token given its context. Summed over the dataset, this is equivalent to maximizing the log likelihood of the data under the model. This is standard maximum likelihood estimation.

In Bayesian parameter estimation, one defines a prior over parameters, observes data, and then uses Bayes theorem to get a posterior. A common choice is to take the parameter value that maximizes this posterior, called the maximum a posteriori estimate. If the prior over parameters is uniform, then the maximum a posteriori estimate is the same as the maximum likelihood estimate.

So at the level of decision theory, the training procedure for a language model chooses the same predictive distribution that a Bayesian statistician would choose under some implicit parameter prior. Architecture, initialization, regularization, and data distribution define this implicit prior. Training is just one long Bayesian style update from the initial prior toward parameters that explain the data.

This tells us that the model's predictive distribution after training is the one that minimizes expected log loss under a Bayesian criterion. What it does not yet tell us is how the model adapts to individual prompts and tasks inside that distribution. For that we need the meta learning view.

## 4. Meta learning and transformers as Bayesian predictors

In meta learning, the data do not come from one task. They come from a distribution over tasks. Each task has its own parameter, which defines a distribution over sequences. A meta learner is trained on many short trajectories from many tasks, with the goal of minimizing cumulative log loss.

Genewein and collaborators analyze this setup for neural sequential predictors, including transformers. They show that memory based meta learning with log loss converges, under reasonable conditions, to a Bayes optimal sequential predictor over the task distribution. In that regime, the trained network's conditional next token distribution matches that of an explicit Bayesian mixture over tasks, with mixture weights given by the posterior over tasks induced by the context. The prior over tasks is encoded in the weights and the posterior over tasks is encoded in the hidden state as it processes the context.

The same work uses this Bayesian view to analyze prompting and prompt tuning. If you view the meta trained network as a Bayesian predictor over tasks, then a prompt is just evidence that conditions this predictor. Prompt tuning, including soft prefixes, becomes a way to choose inputs that drive the internal state into a configuration that encodes as much information as possible about the target task, subject to constraints. Genewein et al. show that on synthetic coin flip and Markov tasks, small soft prompts can steer models close to Bayes optimal prediction, and they characterize cases where prompting alone is fundamentally limited and weight tuning is required.

Xie and collaborators study a complementary synthetic setting. They construct a pretraining distribution as a mixture of hidden Markov models, each representing a latent concept. They prove that next token training in this setting leads to models whose in-context behavior is equivalent to implicit Bayesian inference over the latent concept index. At test time, when you feed a prompt consisting of input output examples from a new task drawn from the same mixture, in-context learning arises because the model infers the latent concept that best explains the prompt and predicts under the posterior over that concept.

Müller and collaborators move to explicit Bayesian tasks. They train transformers to approximate the posterior predictive distributions of Gaussian processes and Bayesian linear regression. They show that transformers can represent accurate posterior distributions in a single forward pass and can act as amortized inference engines for Bayesian neural networks.

Taken together, these results support a precise statement. For the kinds of objectives and data distributions that pretraining uses, the Bayes optimal sequential predictor is a Bayesian predictor over tasks, and transformers can approximate that predictor. So at the functional level, a trained transformer is well described as an amortized Bayesian predictor over the tasks implied by its pretraining data.

The next question is how it implements this adaptation in its forward pass.

## 5. In context learning as an inner optimization procedure

In context learning is the fact that a pretrained model can adapt to a new task from examples in the prompt without any explicit weight updates. Several works show that transformers can implement this by learning an inner optimization process.

Von Oswald and collaborators provide a clean case study. They show that for linear regression tasks one can construct the weights of a self attention layer so that the data transformation it applies is exactly equivalent to one step of gradient descent on a regression loss for an implicit linear model. They then train self attention only transformers on regression tasks and show that optimized transformers either converge to this analytic form or behave very similarly to gradient descent. They interpret this as evidence that transformers can become "mesa optimizers" that learn inner models by gradient descent in their forward pass.

Work on contextual blocks further generalizes this. A contextual block is a contextual layer, such as self attention, stacked on top of a base network. Under reasonable assumptions, feeding context through such a block is equivalent to applying a low rank update to the base network's effective weight matrix. In other words, the context implicitly edits the network via input dependent low rank weight changes.

Other work frames large language models as meta optimizers. In this view, during the forward pass the model computes meta gradients from in context examples and applies them to a latent parameter stored in activations. In context learning and explicit finetuning become two versions of the same underlying update procedure, one acting on ephemeral inner parameters, the other on persistent weights.

From a Bayesian perspective, this inner optimization procedure is an amortized algorithm for approximating the posterior. Instead of computing $P(h | D)$ symbolically, the model runs a learned "inner gradient descent" procedure in its representation space that moves its implicit hypothesis from something close to the prior toward something close to the posterior for that context.

The important point for our theme is that this inner optimizer expects a sequence of evidence. Each token you send, and each intermediate token the model generates, is another small update. The model's implicit posterior is not fixed after your first message. It evolves as you feed in more information.

## 6. Steering the implicit posterior with information

With that in mind, the central question is not "single turn or multi turn". It is:

> How do I choose and schedule information so that the model's implicit posterior moves toward the region of hypothesis space that matches my goals?

Multi turn interaction is one way to do that, but it is a means, not the goal.

If you send one long prompt that mixes your task description, evaluation criteria, style preferences, hidden assumptions, and partial data format, you are effectively specifying a complicated, noisy likelihood. The inner optimizer will still try to move the internal hypothesis $h$ to maximize $P(D | h) P(h)$, but you have given it a confusing single batch of data and then cut the process short. You are not "letting the posterior converge" and you are not steering it.

A better strategy is to treat interaction as sequential experiment design. You have a desired target posterior in mind: a region of hypotheses that correspond to "the model understood the task correctly and is now using the right tools, constraints, and style". You then choose messages, tool calls, and feedback that will move the model's implicit posterior in that direction.

Examples:

- You start by asking the model to restate the task in its own words and propose a plan. Its answer reveals its current implicit hypothesis. You correct it. That correction is a strong likelihood term that rules out some internal hypotheses and upweights others.

- You provide a small number of labeled examples, then ask it to explain the pattern in its own terms. You check whether the inferred concept matches the one you had in mind.

- You let the model propose several candidate solution paths, then you mark some as undesirable and some as closer to what you want. Those markings are feedback that the inner optimizer can treat as additional evidence about your target posterior.

Tool calls and external data are part of the same story. When the model calls a search tool, reads a document, or runs code and sees a test failure, that is more data $D$. A well designed system reformats that data in a way that makes it easy for the inner optimizer to use as likelihood information.

The important thing is that you are not passively "letting multi turn converge". You are actively steering the model toward your desired posterior by deciding which evidence it sees, in which order, and with which framing.

## 7. RLVR: post training the inference process on verifiable tasks

Pretraining and in context learning give you a general approximate Bayesian engine. Reinforcement Learning from Verifiable Rewards, RLVR, is a way to further train this engine on task families where you can automatically check correctness.

In RLVR, the model is wrapped in an environment with a verifier. Given a prompt $x$, the policy $\pi_\theta$ produces a reasoning trace $z$ and a final answer $y$. A rule based verifier evaluates $(x, z, y)$ and returns a scalar reward, for example 1 if the answer is correct and the reasoning is consistent, 0 otherwise. An RL algorithm such as GRPO or PPO then updates the parameters to increase expected reward. RLVR has been key to recent reasoning focused models such as DeepSeek R1 and o1 style systems.

From a Bayesian point of view, before RLVR the model has a prior over reasoning trajectories $P(z | x)$. Often the base model has high pass@k but low pass@1 on math or coding benchmarks: if you sample many traces, some are correct, but the first trace is unreliable. RLVR sets up an objective that effectively pushes the policy toward the conditional distribution

$$P(z | x, \text{correct}) \propto P(z | x) \cdot \mathbb{1}[\text{verifier}(x, z, y) = 1]$$

It does not compute this posterior analytically, but repeated sampling and RL updates reweight the parameters so that trajectories that pass the verifier become typical. Empirically, RLVR both compresses search, turning pass@k into pass@1, and, to a lesser extent, extends the boundary of problems the model can solve reliably. Recent studies find that most of the gain is search compression, with a smaller component of genuine capability expansion, and that RLVR primarily selects and amplifies reasoning patterns that were already present in the base model.

Seen this way, RLVR does more than "pass@k to pass@1". It reshapes the relationship between the model's implicit prior and posterior over reasoning strategies. Pretraining gives you a broad prior over trajectories. The inner optimizer, driven by prompts, produces an implicit posterior over trajectories that fit the prompt. RLVR changes the parameters so that for verifiable domains the posterior that the inner optimizer reaches, given realistic prompts and limited search, is better aligned with the external notion of correctness.

You can say that RLVR trains the inference process itself. It adjusts the prior over strategies and the dynamics of the inner optimizer so that, conditioned on the kinds of evidence that real users and tools will generate, the implicit posterior over reasoning paths more often lands in regions that pass the verifier.

## 8. Implicit posteriors and out of distribution limits

It is important to remember that all of these posteriors are implicit and live inside the model's representation space. The model does not carry around a general purpose symbolic Bayesian engine that can produce arbitrary posteriors over arbitrary domains.

The prior it learns is the prior induced by pretraining data and architecture. The posterior it can represent is constrained by that prior and by its capacity. If a task or world is far outside the training distribution, there may simply be no internal hypothesis that matches it well. In that case the inner optimizer can update as much as it likes, but it will be moving inside the wrong space.

The same is true for RLVR. RLVR can only shape behavior on tasks that the verifier and training curriculum cover. Studies of the limits of RLVR find that it does not usually create fundamentally new reasoning abilities. If the base model cannot solve a problem at all, the RLVR trained model usually cannot either. RLVR mostly reweights and focuses existing patterns, though there is some evidence of modest capability expansion when verifiers and curricula are rich.

So when we talk about posteriors here, we mean "whatever internal state the model converges to when you condition its learned prior on the evidence you feed it". Those posteriors are not magic. They cannot reliably move into regions of task and world space that the model has no representation for.

This is why system design and safety work cannot rely on "Bayes will fix it". You still have to care about pretraining data, architectures, objectives, and post training. The Bayesian frame is a way to describe what those elements are doing, not a guarantee of good behavior.

## 9. The real lesson

**The useful fact is straightforward.**

Transformers trained as language models minimize log loss. In meta learning terms, this pushes them toward Bayes optimal sequential prediction over their training task distribution. Work by Genewein and others shows that such predictors behave as Bayesian mixtures over tasks, with priors in the weights and posteriors in the activations. Work on in context learning shows that transformers can implement gradient descent like inner optimization on latent task parameters, so that prompts act as training data for an inner model. RLVR then post trains this inference process on verifiable tasks, reweighting its implicit prior over reasoning strategies so that the inner posterior better matches external correctness.

Given that structure, the right way to think is not "find the perfect single prompt" but **"design the information flow"**. You want to:

- decide what posterior over tasks and solutions you want the model to approximate
- choose prompts, feedback, tool calls, and curricula that act as evidence for that posterior inside the model's learned space
- accept that these posteriors are implicit and bounded by what the model has already learned

Multi turn interfaces are one way to do this. They are not the point. **The point is to treat the model as what it effectively is: an approximate Bayesian inference engine with a learned inner optimizer, whose behavior you can steer by controlling the evidence it sees.**
