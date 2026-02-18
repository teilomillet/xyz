+++
title = 'Agent Intelligent'
date = 2025-03-02
draft = false
tags = ['ai', 'agents', 'rl', 'intuitus']

[params]
    nextprev = true
    taglist = true
+++

#### De l'Orchestration des Agents à l'IA Guidée par Objectifs

Quand j'ai commencé mon parcours dans l'IA, j'ai été attiré par l'apprentissage par renforcement (RL) grâce au cours de HuggingFace. Ce cours avait un format compétitif avec un tableau de classement qui rendait l'apprentissage vraiment stimulant. J'y ai découvert comment créer des modèles avec des objectifs clairs, les entraîner à atteindre ces objectifs de manière optimale, et observer leur succès. C'était propre, mesurable, et profondément satisfaisant.

Ensuite, j'ai exploré le monde des LLMs. J'ai fait du fine-tuning, appris les techniques, et développé mon propre projet, Merlin. Quand j'ai rejoint Station F pour lancer mon entreprise, je me suis d'abord positionné dans le "fine-tuning as a service" avant de pivoter vers les "agents intelligents".

### L'Ère de l'Orchestration

Quand j'ai commencé à créer des "agents", il s'agissait principalement d'orchestration du LLM sous-jacent. Par exemple, pour une tâche spécifique comme l'automatisation de processus métier, je construisais des outils et les organisais pour accomplir cette tâche. Je donnais au robot une direction, lui faisais faire certaines actions, et cela constituait ce qu'on appelait un "agent".

De nombreuses entreprises, moi y compris, ont essayé de construire des frameworks pour faciliter ce processus. Mais en fin de compte, comme tous ceux qui ont développé de tels agents, j'ai ressenti que cette approche ne passait pas à l'échelle et n'était pas suffisamment fiable. On sentait que quelque chose n'allait pas. C'était comme courir un marathon en thongs - techniquement possible, mais fondamentalement inadapté.

La performance de ces agents dépendait entièrement du fournisseur d'API que vous utilisiez (comme ChatGPT ou Claude). 99,99% des agents précédents fonctionnaient ainsi, limités par les capacités du modèle de base qu'ils orchestraient.

### Le Modèle Devient le Système

Aujourd'hui, en 2025, un nouveau paradigme s'ouvre devant nous. Ce n'est plus une approche qui dépend de ChatGPT ou d'un autre point d'API, mais plutôt un retour aux fondamentaux du RL (ce qui a rendu tout cela possible).

Le tournant décisif est arrivé avec la sortie de DeepSeek-r1 fin janvier 2025. Tout a changé: désormais, l'agent fait partie intégrante du modèle. On ne construit plus un système autour du modèle le plus performant disponible, mais on revient au début pour construire un système (modèle compris) orienté vers notre objectif.

Les outils et autres mécanismes font partie du modèle, pas extérieurs à celui-ci. Auparavant, nous pouvions changer le modèle sur lequel le système était basé pour "améliorer" les performances; maintenant, il faudrait le réentraîner entièrement.

### Des Modèles avec Objectifs Spécifiques

Cela signifie que les entreprises vont construire et entraîner des modèles orientés vers des objectifs précis. Elles ne seront plus dépendantes des fournisseurs. De la même manière qu'un algorithme pour les réseaux sociaux est conçu pour maintenir l'engagement des utilisateurs, nous pouvons imaginer des agents entraînés pour vendre des billets de train, assurer la satisfaction client, ou guider l'intégration de nouveaux employés.

Les entreprises posséderont l'ensemble du processus: du calcul qui alimentera et entraînera les agents (modèles), aux ensembles de données pour les entraîner, jusqu'au déploiement. Des frameworks comme vLLM aideront au déploiement, mais chaque entreprise aura ses cas spécifiques à prendre en compte.

### L'Avantage Stratégique

Les entreprises qui disposent de GPU auront un avantage considérable. En permettant aux talents et aux créateurs de venir les aider à construire l'avenir, plutôt que d'être pris de court et devoir les attirer une fois que tout est déjà construit. Cela signifie que les entreprises qui se donnent l'espace d'avoir une vision seront récompensées, tandis que celles qui réagissent et ne prennent aucun risque pourraient être lourdement pénalisées.

Les premiers arrivants auront clairement un avantage. Ce n'est pas un monde hypothétique, mais quelque chose de tangible qui existera. Ma conviction est que cette approche sera la bonne, elle est esthétique, elle fait sens et elle utilise des techniques deja connues.

Actuellement pour les entreprises, il s'agit surtout d'allouer efficacement le capital. Bien sûr, les entreprises qui ont besoin de liquidités n'auront pas à mettre en œuvre cette approche, ou au contraire, elles pourraient se tourner vers celle-ci. Mais le premier mouvement important consistera à obtenir des GPUs ou à établir des partenariats stratégiques.

### Le Futur: Des Systèmes RL Sur Mesure

Ce qui m'enthousiasme, c'est l'avenir que cela ouvre. Imaginez des agents de service client entraînés de A à Z pour optimiser la satisfaction à travers des milliers d'interactions. Ou des assistants d'intégration qui ont appris par renforcement comment guider efficacement les nouveaux employés. La ou avant nous utilisions la faculté de généralisation des models, maintenant ils seront spécifique a leur tâche de prédilection.

Ce ne seront pas des frameworks orchestrant des appels à Claude ou GPT. Ce seront des systèmes construits sur mesure où le RL et le LLM sont inséparables, où le comportement orienté vers un objectif est intégré dans la structure même du modèle.

Pour ceux qui ont commencé par le RL avant de plonger dans le monde des LLM, il y a une belle symétrie (OpenAI par exemple). Nous bouclons la boucle, mais avec des outils infiniment plus puissants.

L'avenir de l'IA combine l'apprentissage orienté vers un objectif du RL avec les vastes connaissances et la flexibilité des LLM. Non pas comme des composants séparés, mais comme une approche intégrée pour construire une IA qui fonctionne réellement dans le monde réel.
