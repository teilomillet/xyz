+++
title = 'Agent Martin'
date = 2024-05-16T13:49:02+02:00
draft = true
tags = ['teilo', 'Agent', 'Martin', 'LLM', 'AI']

[params]
    nextprev = true
    taglist = true
+++

![agentsmith](/agentsmith.jpg)

L'[agent smith](https://fr.wikipedia.org/wiki/Agent_Smith) est le principale antagoniste de Neo dans la saga [Matrix](https://fr.wikipedia.org/wiki/Matrix_(s%C3%A9rie_de_films)). 
Pour rappel, c'est un programme (IA) informatique qui veille a la stabilité de la Matrice (univers virtuel réaliste simulant le monde actuel). 

Dans le même style, je propose l'agent martin. Alors a priori, il n'aura pas les memes compétences ni objectifs, mais ce sera un agent capable d'interagir dans son environement. Il poura intéragir avec des humains et d'autres agents (Bernard, Thomas, Robert, ...) afin de compléter aux mieux sont objectif.

J'aimerais donc créer plusieurs agents différents. 

Sur la [hierarchisation de l'instruction](https://arxiv.org/pdf/2404.13208) :
En ajoutant un degré de privilege cela permet d'évite   r les catastrophes via model hacking. Le system prompt aura le plus grand privilege, suivi du message de l'utilisateur, puis celui du model output et pour finir le tool output.

Pour notre agent, il faudra que chaque outil soit modulable par l'utilisateur, ainsi un utilisateur peu se passer de certain outils. 
