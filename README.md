# Welcome, Matthew!

Welcome to Dr. Kao's Neural Engineering and Computation Lab!
By joining this lab, you've expressed interest in the fields of machine learning, signal processing, neuroscience, and robotics.
Lukily for you, the research at NECL sits squarely on the intersection of all these fields!
Here is a wecome package to get you started.

## Table of Contents

- [Introduction to Machine Learning](#introduction-to-machine-learning)
  - [Classical Machine Learning](#classical-machine-learning)
  - [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
  - [Above and Beyond](#above-and-beyond)
- [Coding Best Practices](#coding-best-practices)
  - [Becoming a Terminal Wizard](#becoming-a-terminal-wizard)
  - [Becoming a Pythonista](#becoming-a-pythonista)
  - [PyTorch for Deep Learning](#pytorch-for-deep-learning)
  - [Other packages for ML Research](#other-packages-for-ml-research)
- [General Advice](#general-advice)
  - [Reading Academic Papers](#reading-academic-papers)
  - [Expectations and Challenges](#expectations-and-challenges)

## Introduction to Machine Learning

### Classical Machine Learning

#### What is it and Why

By "classical ML", I refer to [Statistical Learning](https://en.wikipedia.org/wiki/Statistical_learning_theory) methods derived by statisticians and mathematicians which were state-of-the-art before neural networks proved their superiority.
Well-known statistical learning methods include:

- K-Nearest Neighbors
- Neive Bayes Classifier
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support-Vector Machines
- K-Means Clustering

... and many more!
While the field of ML has shifted away from classical methods, the knowledge distilled from statistical learning theory provides a fundamental framework upon which modern ML research is based.
Although classical methods are rarely used in practice nowadays, I believe that it is still important to study them as they provide a gentle introduction to core ML concepts such as loss functions, regularization, gradient descent, the bias-variance tradeoff, and a general "feel" for data that is invaluable in ML research.

#### Data Science from Scratch

Although you could take courses such as ECE M148, CS M146, Stats 102B, or Math 156 (Yes, I've taken **all** of these), I think a *much* better use of your time is to thoroughly read through the amazing book [Data Science from Scratch - First Principles with Python](https://www.shroffpublishers.com/index.php?dispatch=attachments.getfile&attachment_id=948&srsltid=AfmBOopRb-kc9jLEoR6QM-SM85RiPiLujM7dy0s3Au11FsvXw7SaK4PD&__cf_chl_tk=H6uCucBLqI1TzMuY5c0tlElkTdZavxAT_8Ib_ZboXDs-1741044841-1.0.1.1-mOwOHERTtGZDzgDr6o8EQjQ0UQsO5oE1SrLhyr1MZQY), by Joel Grus.

I love this book for many reasons:

- It assumes no prior mathematical, statistical, or computer science knowledge from the reader, so it is an easy recommendation to anyone.
- It is a wholistic introduction to classical ML, with two chapters that ease the reader into neural networks, deep learning, and natural language processing at the end.
- It is a super hands-on book, with Python implementations placed at a higher priority level compared to mathematical formulas.
- It has a strong emphasis on good Python programming practices, which in my opinion is a superpower to have when working on actual projects (more on this later).

Please consider giving this book a thorough read if you wish to have a long and prosperous career in ML, or at least going through the first few chapters and then cherry-picking the rest for topics you find interesting.

### Neural Networks and Deep Learning

#### ECE C147 & ECE 239AS

At UCLA, Dr. Kao has become synonymous with the topic of neural networks and deep learning.
After all, he is the professor that brought the fantastic courses [Neural Networks and Deep Learning I (ECE C147, usually winter)](https://catalog.registrar.ucla.edu/course/2023/ecengrc147?siteYear=2023) and [Neural Networks and Deep Learning II (ECE 239AS, usually spring)](https://www.bruinwalk.com/classes/ec-engr-239as-2/) to UCLA.
I strongly recommend taking these courses.
They are extremely popular and notorious for being hard to get into, but one of the perks of join NECL is that you can just ask Jonathan for a PTE.
Also, ask me for notes and course material!

#### NNDL Course Alternatives

Even with a PTE, these courses require you to wait until winter and spring quarters.
Stanford has equivalent courses which Jonathan mostly based his courses off of, and luckily for you, these courses are publically available on YouTube!

- [Stanford CS 239](https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&si=-Xug6CrtgbydwocG)
- [Stanford CS 236](https://youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8&si=9UOEhLf_ICPKg_8W)

Watch these if you are too impatient to wait until winter and spring, and if you want to go the extra mile, their course material (slides and homeworks) are all available on their course website!

#### 3Blue1Brown

You've *probably* heard of 3B1B if you've ever tried to look for math help on YouTube.
3B1B has [an *incredible* series on neural networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=AFIFZO1pZ_1mxyzu), which I *strongly* recommend that you watch.
This series is broken into two parts: part I about neural netwoks from 7 years ago, and part II about GPTs specifically from last year.
Watching only the first part is enough, though who could resist watching the second part?

#### Other YouTube GEMs

In addition to 3B1B, recently there have been an explosion of exemplary YouTube explainer videos on the topic of deep learning.
These cover a wide range of topics, from well-established neural network architectures such as [convolutional neural networks](https://youtu.be/8iIdWHjleIs?si=5DJsRh_RFpQgKFko) to state-of-the-art paradigms such as [diffusion models](https://youtu.be/zc5NTeJbk-k?si=J_-WfeOfNNhkiRty) to cutting-edge research such as [DeepSeek's multi-head latent attention](https://youtu.be/0VLAoVGf_74?si=ip437BlG-lmSr20C).
If you are a total deep learning nerd like me, and you haven't discovered the wealth of knowledge on YouTube, I'd recommend that you start digging!

### Above and Beyond

You know Andrej Karpathy? One of the guys that built ChatGPT?
Well, he has [a YouTube channel](https://www.youtube.com/@AndrejKarpathy/featured) where he makes 2h-long videos where he teaches you deep learning from the ground up.
His videos are long but feel short, there is always a lot of hands-on coding, and at the end of every video he will always have built something incredibly cool.
Things that he builds in his videos include MicroGrad, an autograd engine (ever wondered how PyTorch's `loss.backward()` magically computes the gradients of your entire network?), MiniGPT, basically GPT-2 built from scratch, and many more!
I recommend watching *his entire channel* backwards.

## Coding Best Practices

### Becoming a Terminal Wizard

It's really a shame that students come out of CS 31 and CS 32 knowing all the fundamental data structures and algorithms in computer science, having built a full-out game in C++, but somehow not knowing how to use a termial properly.

In the lab, you will be working extensively with servers and work stations, which are usually always running Ubuntu Linux.
Additionally, most cutting-edge research projects are *only built to support Linux* (e.g. robotic arms).
Linux is the world's most popular operating system, and it's the world's most powerful operating system.
But to unlock its power you **need** to be comfortable the terminal.

I don't really have a resource to point you to for learning how to use the terminal, personally I got really good at using the terminal through a fascination with terminal-based applications such as NeoVim.
I think the best way to get good at using the terminal is to just use it more.
There will come a point where suddenly you prefer to interact with your computer through the terminal vs. the graphical user interfase (GUI).
In my opinion, truly understanding how the terminal works provides a gateway to understanding how computer systems work.
Through trying to figure out how to perform certain tasks in the terminal which we usually take for granted in the GUI, we will often need to get over obstacles we never imagined we'd come across.
That's where the learning happens.

### Becoming a Pythonista

Python is a very unique language.
It is a high-level, dynamically-typed, interpreted language that enables researchers to iterate on projects extremely quickly.
But on the other hand, being dynamically-typed also means being extremely prone to bugs.
Computer scientists that program in systems-level languages like C sneer at Python for being a non-serious language.

> ***But Python could be a serious language if you take it seriously.***

To write research-quality code in Python, you’ll want to borrow the discipline people expect in so-called “serious” languages.
In practice, that means:

#### Annotate everything.

- Use [PEP 484](https://www.python.org/dev/peps/pep-0484/) type hints on functions, methods, and class attributes.
- Run a static checker (e.g. **Mypy** or **Pyright**) as part of your edit-and-commit workflow to catch signature mismatches and unintended `Any` types before they sneak into your code.

#### Automate formatting & linting.

- Adopt a formatter like **Ruff** so every `.py` file looks the same-—no more wasted time bikeshedding on tabs vs. spaces.

#### Isolate your environments.

- Never install packages globally. Create a fresh **virtual environment** (via `uv init`!) for each project.
- Pin every dependency in a `pyproject.toml` and `uv.lock`. That way, anyone-—even your future self-—can reproduce your setup with a single command.

#### Test & document as you go.

- Write unit tests with **pytest** (or your framework of choice) for any non-trivial function. Aim for fast, isolated tests that can run on every commit.
- Document public interfaces with clear docstrings (e.g. NumPy or Google style). A little discipline up front saves days of head-scratching later.

By building these habits, you’ll keep your codebase clean, catch bugs early, and make your research reproducible and robust.
In short, you’ll prove to any skeptic that Python can be every bit as “serious” as C or C++—because you’ve treated it with the rigor it deserves. Welcome to true Python mastery!

### PyTorch for Deep Learning

PyTorch’s eager, Python-first API and one-line GPU support make prototyping and debugging models extremely fast. It’s the de facto industry standard—powering research at Facebook, Tesla, OpenAI, and countless labs. 
The best way to get up to speed is the [official PyTorch docs](https://pytorch.org)—-start with [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) and explore the concise tutorials gallery for hands-on examples.

### Other packages for ML Research

Here are a few more libraries and tools that’ll level up your deep-learning research:

- **NumPy & Pandas** for efficient numerical ops and tabular data wrangling.
- **Matplotlib / Seaborn / Plotly** for visualizing training curves, embeddings, and model outputs.
- **Hugging Face Transformers & Datasets** for state-of-the-art NLP models and easy dataset loading.
- **Weights & Biases (wandb)** for experiment tracking, logging metrics, and artifact versioning.

## General Advice

I know I just threw a TON of links at you and I can’t imagine how overwhelming this must feel to someone who is just getting started with ML.
But PLEASE keep in mind that everyone in the lab is SUPER nice and more than happy to help you.
Whenever you’re stuck on something, ANYTHING, ask! ask! ask!

### Reading Academic Papers

Jumping into an academic paper for the first time can feel like decoding a secret language—dense notation, unfamiliar terminology, and assumptions you haven’t yet covered in class.
Here’s one trick to soften the learning curve: **use AI as your guide**.
Simply upload the PDF to ChatGPT (or paste in key excerpts) and ask it to “explain this paper as if you’re an undergraduate student studying ML.” You can then:

- **Request a high-level overview** of the problem, why it matters, and the core contributions.
- **Ask for mini-walkthroughs** of each equation or algorithm step—ChatGPT can break down symbols, show intermediate steps, and relate them back to concepts you already know.
- **Pose “what if” questions** to test your understanding (“What happens if we change this hyperparameter?”) or fill in background gaps (“Why do we assume IID data here?”).

By iterating with AI—summaries, paraphrases, and targeted follow-up questions—you’ll build confidence in reading and critiquing papers far more quickly than going it alone.

### Expectations and Challenges

Undergraduate researchers often juggle full class loads, club activities, and part-time jobs, all while trying to contribute meaningfully to cutting-edge projects. You might run into:

- **Time management hurdles:** lab meetings, coding sprints, and coursework deadlines can collide.
- **Knowledge gaps:** unfamiliar math, statistics, or coding patterns can make even “small” tasks feel huge.
- **Imposter syndrome:** comparing yourself to grad students or published authors can be intimidating.

#### Strategies to overcome these challenges:

1. **Block out dedicated “research hours.”** Treat your lab work like a class: schedule it in your calendar and protect that time.
2. **Break tasks into bite-sized goals.** Instead of “implement paper X,” aim for “write pseudocode for the main loop” or “run the data-loading example.”
3. **Lean on the community.** Pair-program with other undergrads, ask questions in Slack or office hours, and don’t wait to hit a wall before seeking help.
4. **Fill gaps proactively.** When you notice a stuck point—say, on optimization theory—spend 30 minutes on a quick tutorial or textbook chapter before diving back in.
5. **Celebrate small wins.** Logging every successful experiment, bug fix, or paper summary reinforces progress and keeps motivation high.

Remember: every seasoned researcher was once an undergrad in your shoes. With structured habits and a willingness to ask for help, you’ll turn these hurdles into stepping stones.

Given your coding and ML experience, realistically speaking, you might not be able to make as many contributions as you might have liked to.
That is completely fine.
Keep in mind that ultimately, we join research labs to learn.
As long as you’re learning new things, it is all worth it.

Good luck, and I look forward to working with you!

-- Andy
