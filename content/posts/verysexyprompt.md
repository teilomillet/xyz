+++
title = 'Very Sexy Prompt'
date = 2023-06-18T16:49:27+02:00
draft = false
tags = ['teilo', 'prompt-engineering', 'prompt', 'chatgpt', 'openai']

[params]
    nextprev = true
    taglist = true
+++

## TL;DR

- Il existe plusieurs techniques pour optimiser les prompts et améliorer la véracité des réponses.

- Les techniques comprennent le fine-tuning, le zero-shot learning, le few-shot learning, le chain of thought, et le RCI (Critique).

- Faire appel à un expert peut également améliorer les réponses fournies par le modèle.

- En combinant ces techniques, on peut obtenir des réponses satisfaisantes et performantes, mais il n'y a pas de prompt parfait pour toutes les demandes.

Il existe de nombreuses façons d'optimiser les prompts, que ce soit pour améliorer la véracité des réponses du chatbot ou maximiser l'utilisation des tokens. Pour la seconde partie, cela peut être réalisé en utilisant principalement le langage [Markdown](https://www.markdownguide.org/) ou en demandant une réponse sous forme de points à puces (bullet points).

Pour ce qui est d’optimiser la véracité des réponses, voici quelques techniques :
___

### *Fine-tuning (Ajusté)*

Entraîner un modèle sur de nouvelles données. Cela permet d'optimiser les résultats sur un sujet prédéfini, au risque de perdre en performance sur d'autres sujets. Le modèle sera plus performant dans ses réponses concernant le sujet sur lequel il a été "ajusté" (fine-tuning) grâce aux nouvelles données.

Cette étape n'est pas indispensable pour obtenir de bons résultats, en revanche elle l'est lorsque l'on souhaite obtenir des résultats quasi-parfaits sur un sujet précis. Pour effectuer cette technique, il faut dans la plupart des cas avoir accès à l'API ou au modèle directement.
___

### *Zero-shot Learning (Question direct)*

Il s’agit de la technique la plus simple, une question direct au chatbot.

**User:** 
- Quelle est la capitale de la France ?

**Assistant:** 
- La capitale de la France est Paris.


Cette technique atteint rapidement sa limite lorsqu’elle est posée sur des sujets plus complexe. [Le fine-tuning peux améliorer les résultats de cette technique.](https://arxiv.org/pdf/2109.01652.pdf)
___

### *Few-shot Learning (Question avec exemples)*

On ajoute des exemples lorsque l’on pose des questions au chatbot, ainsi on indique un type/format de réponse souhaité. [Ces exemples doivent être le plus évident possible, de qualités et compréhensibles. Ils influeront grandement sur la réponse du chatbot.](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#few-shot)


**User:**
- Q: Capitale de l'Allemagne ?
- R: Berlin
- Q: Capitale des Pays-Bas ?
- R: Amsterdam
- Q: Capital de la France ?

**Assistant:** 
- Paris


[Cette technique aussi connait ses limites, notamment sur des questions liées au raisonnement du chatbot.](https://www.promptingguide.ai/techniques/fewshot)
___
### *Chain of Thought (Étape par étape)*

Comme un être humain, un chatbot est plus enclins à faire des erreurs s’il lui est demandé de répondre spontanément à une question qui demande réflexion. En revanche, s’il réfléchit avant de répondre ce dernier augmente grandement sa performance. C’est le [Chain-of-Thought (CoT)](https://arxiv.org/pdf/2201.11903.pdf) , cela consiste en un few-shot Learning avec un raisonnement par étape, indiqué dans les exemples.
[Le CoT est particulièrement efficace pour obtenir de meilleures réponses arithmétique ou à des problèmes de raisonnement.](https://arxiv.org/pdf/2203.11171.pdf)


**User:**
- Q: S'il y a 3 voitures dans le parking et que 2 nouvelles voitures arrivent, combien de voiture sont dans le parking ?
- R: Il y a 3 voitures dans le parking, 2 voitures arrivent. 3 + 2 = 5 voitures. La réponse est 5. 

- Q: Les poules de Jean pondent 16 oeufs par jour. 
    Jean mange trois oeufs chaque matin et il fait aussi des crêpes avec quatre autres. 
    
    Il vend le reste des oeufs pour 0,25€ l'oeuf. Combien gagne-t-il chaque jour ?

**Assistant:**
- Les poules de Jean pondent 16 œufs par jour. Jean en mange 3 chaque matin et en utilise 4 pour faire des crêpes. Par conséquent, il lui reste 16 - 3 - 4 = 9 œufs qu'il peut vendre.

    Si Jean vend ses 9 œufs pour 0,25 € l'œuf, il gagnera 9 * 0,25 = 2,25 € chaque jour.


Il est aussi possible d’améliorer les réponses de la technique Zero-shot Learning en ajoutant à la fin de la question: “[pense étape par étape](https://openreview.net/pdf?id=e2TBb5y0yFf)” et ainsi faire un Zéro-shot Learning-CoT.
___
### *Tree of Thoughts(arbre de pensée)*

Comme un arbre de décision, le chatbot créer un arbre avec de multiples possibilités et choisi parmi les réponses possibles. Pour choisir, il utilise un algorithme prédéfini. Cette technique n’est pas nécessaire pour les tâches où le chatbot est deja très bon. En revanche, le [Tree of Thoughts (ToT)](https://arxiv.org/pdf/2305.10601.pdf) peut être utile pour réaliser des raisonnements longs et complexes, comme la résolution de certains problèmes.

L’utilisation d’[agents](https://github.com/yoheinakajima/babyagi) peut aussi être utile pour ce même genre de tâche (attention aux prix, car le nombre de token peut monter très rapidement).
___
### *RCI (Critique)*

Grâce à la “mémoire” des chatbots, il est possible de poser des questions sur de précédentes réponses et/ou d’ajouter du contexte. [RCI](https://arxiv.org/pdf/2303.17491.pdf) permet d’améliorer les réponses du chatbot en le faisant s’auto-critiquer et ainsi s’améliorer. Après avoir obtenu une réponse du chatbot, notre second prompt lui demandera de s’auto critiquer et de trouver les erreurs dans sa réponse. Puis à l’aide d’un 3e prompt, il faudra demander au chatbot de s’améliorer et de corriger ses erreurs, pour ainsi nous fournir le bon résultat.

1. La question
2. L’auto-critique
3. La correction en tenant compte de l’auto-critique
___
### *L’expert*

Il est connu que les Large Language Models donneront de moins bonnes réponses si vous utilisez des prompts qui implicitement indiquent que vous êtes peu qualifié et donc dans l’incapacité de dire si la réponse est juste.1 Cela s’appelle le “[sandbagging](https://arxiv.org/pdf/2212.09251.pdf)”.

Pour contrer cela et améliorer les réponses fournies par le LLM, il est recommandé de faire [appel à un expert](https://arxiv.org/abs/2306.08997), soit en indiquant le nom/la position de l’expert en question (ex: “Tu es un C-Level en charge du backend chez Google, …” ou bien en le faisant trouver par lui même ( ex: “Donne moi une estimation éclairée des trois experts les plus aptes à résoudre cette question.”') puis en réutilisant ces experts dans la question.
Il convient de noter que bien que les résultats annoncés par le papier cité précédemment soient en train d'être [réfutés](https://flower-nutria-41d.notion.site/No-GPT4-can-t-ace-MIT-b27e6796ab5a48368127a98216c76864#d4bea55e46b5490184bf797f96de35c0), l'utilisation judicieuse de cette technique permet d'éviter le sandbagging et de favoriser l'amélioration du code, en particulier dans le contexte de la génération de code.

En combinant ces différentes techniques, on peut rapidement obtenir des réponses satisfaisantes et performantes. Il n’existe pas de “prompt” parfait, qui répond à toutes les demandes. Cependant, en trouvant la combinaison adéquate, il est possible d’atteindre les objectifs visés.
