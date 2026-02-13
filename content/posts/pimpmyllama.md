+++
title = 'Pimp My Llama'
date = 2024-01-08T13:00:00+02:00
draft = false
tags = ['teilo', 'LLaMa', 'GPT', 'ChatGPT']

[params]
    nextprev = true
    taglist = true
+++

![pimpmyllama](/pimpmyllama.png)

Il y a 2,5 façons prédominantes de personnaliser un LLM (Large Language Model).

La première, la plus connue, est le “**prompt engineering**”. Elle l'est car quasiment tout le monde l'utilise lors d'interactions avec son LLM préféré (ChatGPT, Bard, Llama2, etc.). Chaque fois que vous “promptez” (instruisez) un LLM de vos envies, par exemple “réponds brièvement”, “tu es un expert dans…”, “explique-moi comme si j'avais 5 ans”, “réfléchis étape par étape”, … vous le personnalisez et faites usage du “prompt engineering”.

Il existe différentes façons d'en profiter, certaines sont connues depuis longtemps ([Very Sexy Prompt](/verysexyprompt)) et de nouvelles sont découvertes [chaque mois](https://arxiv.org/search/?query=%22prompt+engineering%22&searchtype=all&source=header). Cette technique est très utile pour le court terme car facile à implémenter et très modulaire. En revanche, elle a un défaut : elle ne tient pas sur le long terme. Il se trouve que nos amis les LLMs sont comme Leonard Shelby dans Memento, mais en pire. Pour ceux qui ont manqué ce film, le personnage principal a une mémoire de poisson rouge : il ne se souvient plus d'une conversation après un certain temps,dans notre cas, après une seule interaction.

Pour que notre modèle puisse tenir une conversation qui fait sens et se “souvenir” des précédentes interactions avec l’utilisateur, il faudra donc qu’à chaque message de l’utilisateur soient ajoutés les échanges précédents (de façon non perceptible). L’ajout de ces messages suit alors un système de roulement qui permet d’ajouter les derniers messages en priorité, au détriment des plus anciens, et ainsi le modèle peut se “souvenir” d’une conversation, jusqu’à un certain point…

### Memento

Dans Memento, pour pallier son manque de mémoire et se souvenir d’informations importantes, Leonard Shelby prend des photos et se tatoue ses informations. Chez OpenAI, une autre technique a été préférée : la répétition (alias “**system prompt**”), c'est-à-dire qu'à chaque interaction avec un LLM, en préambule, un message fixe et prédéterminé par l'utilisateur est envoyé avec ses instructions. Cela permet de rafraîchir la mémoire de notre interlocuteur durant toute la conversation et à chaque interaction, même si les anciens messages sont “oubliés” au court du temps. 

Mais cette technique est limitée elle aussi, par la taille du **contexte**. Le contexte définit la longueur du texte que le modèle peut recevoir en entrée. C’est une limite définie par l'architecture du modèle ; on peut l'imaginer comme la “mémoire” du modèle, elle est fixe et limitée. Dans notre cas, elle sera composée du “system prompt” (répété à chaque message), du nouveau message de l'utilisateur, et enfin des messages précédents de la conversation (s'il y en a). Ce qui signifie que plus notre system prompt et notre nouveau message prendront de place dans le contexte, moins notre modèle se “rappellera” des échanges précédents. 

Chez OpenAI, le “system prompt” est bloqué à 1500 'lettres' ([tokens](https://fr.wikipedia.org/wiki/Analyse_lexicale)) (ce qui est peu, quand on veut tirer le maximum de son poisson rouge). Avec les modèles open-source, vous pouvez organiser comme vous le souhaitez votre contexte (et donc votre “system prompt”). De plus, pour les modèles tels que Llama, Mistral, etc., le contexte peut être [adapté/augmenté](https://arxiv.org/pdf/2309.00071.pdf) significativement.

### Le bureau de poste ouvre à 9h.

Alors, il existe aussi ce que je qualifierais de demi-façon, une augmentation du prompt engineering. Le [RAG (Retrieval Augmented Generation)](https://arxiv.org/pdf/2005.11401.pdf) permet de sauvegarder du texte dans une base de données. Cette base de données enregistre le texte sous forme de vecteurs dans un espace multi-dimensionnel selon la similarité du texte. 

Par exemple, “Le bureau de poste ouvre à 9h” sera proche (dans l'espace) du texte : “Le coiffeur de la place ouvre à 9h30”, mais sera éloigné du texte : “Il faut un bonnet de bain pour accéder à la piscine municipale” et encore plus éloigné du texte : “Hier soir, j’ai mangé des raviolis”. Cette méthode est très pratique pour faciliter le travail du prompt engineering, car elle permet de retrouver et d'ajouter au contexte l'information nécessaire facilement. 

Par le biais de ces similitudes, le modèle peut retrouver une information, l'ajouter au contexte et augmenter la précision de la requête de l'utilisateur. Mais, comme précédemment indiqué, elle ne résout pas le problème de contexte. In fine, le gain apporté par le RAG est ni plus, ni moins que de faciliter le rajout d’informations dans le contexte sans avoir à les retrouver et les saisir soi-même. Ainsi, si on demande au modèle à quelle heure ouvre le bureau de poste, il répondra “9h”.

### Generative Pre-trained Transformer (GPT)

Heureusement, il existe une deuxième façon de personnaliser un LLM : le “fine-tuning” (FT), méconnu du grand public mais essentiel au succès de ChatGPT et de ses successeurs (https://openai.com/research/instruction-following). Pour comprendre le fine-tuning, il faut revenir à la base et comprendre comment un modèle est (pré-)entraîné **(le P dans GPT : Generative Pre-trained Transformer)**. Pour faire simple, un programme (le Transformer) composé de multiples algorithmes s'entraîne sur un grand nombre de textes, des millions voire des milliards de lignes de textes en tout genre, souvent extraits de sites internet. [Plus ces données sont grandes, meilleure sera la performance de l'algorithme](https://arxiv.org/pdf/2203.15556.pdf).

L'entraînement consiste à “prédire” quelle sera la séquence dans chaque texte, en se basant seulement sur les mots précédents. Ainsi pour le texte, “Je vais chez le boulanger.”, l'entraînement se déroulera comme suit : 
- “Je”, 
- “Je vais”, 
- “Je vais chez”, 
- “Je vais chez le”, 
- “Je vais chez le boulanger.” 

À chaque essai, son objectif sera de trouver le mot suivant manquant. Lors de l'entraînement, le modèle ne va pas pouvoir absorber toutes ces millions de lignes d'un coup, alors ces textes sont découpés en morceaux. La taille de ces morceaux définira le contexte du modèle. Pour chaque morceau, le programme essaiera de “prédire” le mot suivant comme expliqué et ajustera ses “prédictions” en fonction des résultats. Une fois toutes les données passées dans le programme, on peut décider de les repasser une nouvelle fois ou bien s'arrêter. Maintenant, notre modèle est capable de “deviner” le prochain mot de manière plus ou moins effective, il nous suffit alors d'ajouter le mot deviné par le modèle au texte précédent de manière continue pour que celui-ci continue sa génération (à l'infini ?).

À ce moment, il est important de savoir qu’il ne s’agit pas seulement de découvrir le mot suivant le plus probable d’après une loi statistique, mais que le modèle “comprend” les mots précédents et le sens de la phrase, et qu’il est donc capable de s’adapter et de “prédire” le mot suivant avec justesse.

### Une fois notre GPT obtenu comment en faire un chatGPT ?

Actuellement, notre LLM est bon pour prédire la continuité de textes sans nécessité de s’arrêter. Pour lui, une question est la continuité d’une immensité de textes et non une fin. Il faut le voir comme une machine à écrire autonome qui écrirait des morceaux d'internet dans un non-sens chaotique, parsemé de questions, de monologues et de textes en tout genre. Dans cette forme, GPT est capable de “prédire” le prochain mot, sans s’arrêter. Il pourra finir une histoire, un article, une page Wikipédia, en revanche, il ne saura pas distinguer une question d’une liste de questions et pourra donc répondre par davantage de questions.

Afin de rendre le modèle plus utilisable et plus “aligné” avec l'utilisation que l'on connaît, on va le “fine-tuner”. C’est-à-dire qu’on va le ré-entraîner, mais cette fois-ci sur des données présélectionnées et spécialisées, avec des exemples concrets et choisis de réponses que l'on souhaite obtenir de sa part. Ainsi le modèle continue son entraînement mais se spécialise ou bien apprend une façon spécifique de répondre ou d'adresser une question, et bien d’autres choses que contiendra notre jeu de données pour l'entraînement. Lors de cet entraînement, on lui apprendra des marqueurs, équivalents à notre majuscule et point, qui seront des indicateurs de début et de fin de réponse, de cette manière il apprend à générer un indicateur de fin de réponse et donc à s’arrêter.

### La voiture autonome

Pour illustrer, on peut faire l'analogie d'une voiture autonome : c'est comme si on avait un **moteur** (GPT, Llama, Mistral) auquel on ajoute un châssis, une direction, des roues, des freins, etc., afin d'en faire une voiture utilisable (ChatGPT/LLM). L'appui sur la pédale d'accélération serait son indicateur de début de génération, puis, de par sa “compréhension” qu’il est arrivé à destination, il appuierait lui-même sur les freins pour s’arrêter. Cependant, de la même façon qu’en rajoutant du poids sur notre voiture, on perd de la vitesse ; un LLM à chaque fine-tuning peut perdre des connaissances pour en apprendre de nouvelles. De plus, un moteur de 2CV ne peut pas devenir un moteur de 100CV. Et vous l'aurez compris, l'utilisateur ne fait qu'annoncer la destination (ce qui nécessite de connaître la voiture (et la route ([sandbagging](https://cims.nyu.edu/~sbowman/eightthings.pdf))) pour y arriver rapidement).

Aujourd'hui, un modèle, comme une voiture, est techniquement tunable à l'infini au détriment de perdre certaines capacités en cours de route. Le fine-tuning est utilisé pour spécialiser des modèles vers des besoins spécifiques : extraction d’informations, remplissage de formulaire, role-playing, … mais aussi de manière plus générale, comme l’ajout de nouvelles langues prises en charge, initialement non parlées par le modèle, entre autres.

À l'instar du prompt engineering, le FT prend du temps et des ressources. Il peut vite devenir très coûteux si on veut ajouter de la modularité. De la même façon que lors du pré-entraînement, les résultats sont dépendants des ressources engagées. Plus de données et de ressources donneront un meilleur modèle, à condition que les données soient ciblées et de qualité. Le fine-tuning apportera de la performance et de la robustesse sur des tâches répétitives ou définies.

Depuis quelques mois, un nouvel entrant possédant le meilleur des deux mondes (faibles coûts et hautes performances) fait son entrée dans les [classements](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). Il s'agit du merging qui vise à combiner les résultats de fine-tuning entre eux pour obtenir “une somme” des modèles sélectionnés et, bien que dépendant des FT originaux, s'avère efficace et prometteur.

Chaque méthode n'empêche pas l'autre : **le prompt engineering amplifié par le RAG et combiné au fine-tuning peut faire des merveilles**. Bien que parfois overkill, il est important de comprendre qui fait quoi et ainsi comprendre sur quoi agir pour obtenir le résultat souhaité. Les modèles fine-tunés offrent la possibilité de mieux contrôler le résultat en formatant le résultat et permettent d’éviter les incompréhensions.

Parfois un simple changement de destination est nécessaire, parfois c’est un changement dans le véhicule qui doit être opéré, ou bien les deux.
