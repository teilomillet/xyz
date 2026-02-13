+++
title = 'Ordinateur'
date = 2024-12-14T17:06:52+01:00
draft = false
tags = ['teilo', 'ordinateur']

[params]
    nextprev = true
    taglist = true
+++

Un ordinateur, ou calculateur (traduction de l’anglais ‘computer’) est un appareil fonctionnant grâce à l’électricité, capable, en appliquant des instructions prédéfinies, de traiter des signaux.

Ma définition d'un ordinateur a beaucoup évolué au fil du temps. Comme vous peut-être, je pensais savoir ce qu'était un ordinateur. C'est ce truc qu'on utilise tous les jours. Mais quand on y regarde de plus près, c'est difficile de comprendre comment on est passé d'une simple machine à calculer à ces appareils qui tiennent dans nos poches.

Dans ce poste, je vais raconter comment j'ai découvert ce qu'est vraiment un ordinateur. Pas la définition du dictionnaire, mais la vraie compréhension de son fonctionnement.

Commençons par quelque chose que nous connaissons tous : l'interrupteur d’une lampe. Quand on l'allume = 1, quand on l'éteint = 0. Simple, non? Cette différence entre allumé et éteint, c'est exactement le principe de base des ordinateurs.

En 1725 (oui, je remonte loin), un Français, Basile Bouchon, a eu une idée. Il s'est dit: "Et si je faisais des trous dans une carte pour contrôler une machine à tisser?" Un trou = 1, pas de trou = 0. Ce principe si simple soit-il est toujours à la base de nos ordinateurs modernes.

Aujourd'hui, quand j'appuie sur une touche de mon clavier, je déclenche exactement le même type de système, mais avec de l'électricité. Au lieu d'avoir des trous dans une carte, on a des signaux électriques: tension électrique (5V ou 3.3V) = 1, pas de tension (~0V) = 0. Ces signaux sont ce qu'on appelle des 'bits' (binary information digit).

Mais comment on passe d'un simple interrupteur à un ordinateur capable de faire tourner des jeux vidéo ou de nous permettre de communiquer avec le monde entier? C'est là que ça devient intéressant...

Dans nos ordinateurs, ces petits signaux électriques voyagent très très très rapidement. Imaginez essayer d'allumer et d'éteindre la lumière des millions de fois par seconde, impossible. Pourtant, c'est exactement ce que fait mon ordinateur en ce moment même pendant que j'écris ce texte.

Prenons l'exemple d’un clavier. Quand j'appuie sur la touche 'A', contrairement à une machine à écrire, ce n'est pas directement la lettre qui est envoyée sur mon ordinateur. Mon clavier transforme cette frappe en une série de bits : 01000001. Mais ce n'est pas aussi simple que d'envoyer tous ces bits d'un coup (même vous, vous avez dû lire chaque chiffre l'un après l'autre, non?).

En réalité, c'est comme une petite conversation très rapide qui se passe:

- Mon clavier: "Attention, j'ai quelque chose à dire!" (envoie un 0 de départ)
- Puis il envoie: 0, 1, 0, 0, 0, 0, 0, 1 (un par un)
- Et enfin: "J'ai fini!" (envoie un 1 de fin)

Mais pour que cette conversation fonctionne, il faut que mon clavier et mon ordinateur se comprennent. C'est comme quand vous rencontrez un inconnu - il faut d'abord trouver un moyen de communiquer. Dans le monde des ordinateurs, on appelle ça des standards (comme le Wi-Fi, le Bluetooth). C'est un peu comme l'anglais dans le monde - une langue commune que tout le monde comprend.

Et la vitesse dans tout ça? Si je vous parle trop vite, vous allez avoir du mal à me suivre. C'est pareil pour les ordinateurs. Ils doivent se mettre d'accord sur leur vitesse de communication, qu'on mesure en bits par seconde (bps) ou en Megabits par seconde (Mbps). C'est comme deux personnes qui ajustent leur débit de parole pour se comprendre. (Si vous êtes déjà allez en Espagne vous savez de quoi je parle)

Une fois que le signal est envoyé, il arrive dans la mémoire tampon. C'est comme notre cerveau, quand on se répète une information avant de l'écrire. Ce n'est pas encore stocké définitivement, mais c'est prêt à être utilisé. D'ailleurs, parlant de mémoire, les ordinateurs en ont plusieurs types. La RAM, comme votre bureau de travail. Plus il est grand (plus vous avez de RAM), plus vous pouvez avoir de documents ouverts en même temps sans avoir à ranger et sortir des documents du placard (le disque dur). C'est pour ça que notre ordinateur ralentit quand nous ouvrons trop de programmes en même temps, si votre bureau est facilement encombré vous devez constamment ranger et ressortir des documents.

Cette mémoire tampon joue un rôle crucial, c'est un sas d'entrée qui permet de gérer le trafic des informations. Car oui, nos ordinateurs ont différents "circuits spécialisés", encore une fois comme si dans notre cerveau, certaines zones étaient dédiées à la vue, d'autres à l'ouïe, d'autres encore au langage...

Et c'est là qu'on arrive au “cœur” de l'ordinateur : le processeur, ou CPU (Central Processing Unit). C’est le chef d'orchestre de tout ce système. Aujourd'hui, nos ordinateurs ont plusieurs “core”, mais concentrons nous sur un seul pour bien comprendre. Quand je dis que nos ordinateurs ont plusieurs 'core', c'est comme si notre ville avait plusieurs quartiers qui travaillent en même temps. Chaque quartier (core) peut gérer ses propres tâches indépendamment. C'est pour ça que vous pouvez écouter de la musique pendant que vous naviguez sur internet - chaque core s'occupe d'une tâche différente.

Le processeur, c'est un ensemble de circuits, qui sont eux-mêmes composés de portes logiques, qui utilisent des transistors. Je sais, ça fait beaucoup d'un coup ! Décomposons tout ça :

Le transistor, c'est la base. Revenons à notre analogie de l'interrupteur de lumière. Un transistor, c'est exactement ça, mais en minuscule - quelques nanomètres, c'est-à-dire plus petit qu'un virus ! Quand il reçoit de l'électricité (notre fameux '1'), il laisse passer le courant. Sans électricité (notre '0'), il bloque le passage.

Maintenant, imaginons que nous connectons plusieurs de ces transistors d'une façon spécifique. C'est ce qu'on appelle une porte logique. Prenons un exemple concret : quand vous utilisez votre carte bancaire, l'ordinateur doit vérifier deux choses :

- Votre code PIN est-il correct ?
- Avez-vous assez d'argent ?

Pour que la transaction soit acceptée, il faut que les DEUX conditions soient vraies. C'est exactement ce que fait une porte logique 'ET' : elle ne laisse passer le courant que si ses DEUX entrées sont des '1'. Si une seule est '0', c'est non ! Grâce à ces portes logiques, les ordinateurs peuvent prendre des décisions.

Les portes logiques sont comme différents types de décisions. Prenons quelques exemples : La porte 'ET' (AND), on l'a vue avec la carte bancaire - il faut que TOUTES les conditions soient vraies. Mais il existe aussi la porte 'OU' (OR). Pour déverrouiller votre téléphone: soit avec votre empreinte digitale, soit avec votre code. Si l'UN DES DEUX est correct, ça marche. Et il y a la porte 'NON' (NOT), qui inverse simplement le signal. Si vous recevez un 1, elle le transforme en 0, et vice-versa.

Un circuit, c'est comme une ville avec ses routes et ses carrefours, mais au lieu de voitures, ce sont des signaux électriques qui circulent. Chaque carrefour est une porte logique qui décide si le signal peut passer ou non. Quand j'écris la lettre 'A' sur mon clavier, chacun des 8 bits de son code binaire emprunte son propre chemin à travers ces circuits, jusqu'à ce que finalement, la lettre apparaisse sur mon écran. Et a chaque fois, ça passe par des transistors à l’intérieur de portes logiques qui forment d’autres circuits.

D'ailleurs, vous savez pourquoi votre téléphone chauffe quand vous jouez ou faites une visioconférence ? Reprenons notre analogie de la ville : c'est comme si d'un coup, toutes les rues étaient utilisées en même temps, tous les carrefours actifs, toutes les lumières allumées. Plus il y a de transistors qui travaillent, plus il y a d'énergie qui circule, et donc plus ça chauffe. C'est pour ça que nos ordinateurs sont équipés de ventilateurs.

Une des limites actuelles des ordinateurs, c'est justement le nombre de transistors qu'on peut utiliser et leur taille. Plus on en met, plus ça chauffe. Plus ils sont petits, plus c'est difficile de les fabriquer. C'est un peu comme si on essayait de construire une ville de plus en plus grande avec des rues de plus en plus étroites, à un moment, ça devient vraiment compliqué, voire impossible.

Alors maintenant, quand je regarde mon ordinateur, je ne vois plus cette boîte magique qui fait des choses extraordinaires. Je comprends que des millions de petits interrupteurs s’allument et s’éteignent au rythme de l'électricité, organisés en circuits sophistiqués, qui communiquent entre eux à une vitesse vertigineuse.

Ma définition de départ était vague, vous vous souvenez ? "Un appareil fonctionnant grâce à l'électricité, capable, en appliquant des instructions prédéfinies, de traiter des signaux." Mais maintenant, cette définition prend tout son sens, n'est-ce pas ? Chaque mot a son importance :

- "L'électricité" : ces fameux 0 et 1 qui circulent dans nos transistors
- "Instructions prédéfinies" : nos portes logiques et leurs circuits
- "Traiter des signaux" : toute cette orchestration complexe qui transforme nos actions en résultats

Et quand on pense que tout a commencé avec des cartes perforées... Que nous sommes passés de machines mécaniques qui prenaient des salles entières à des ordinateurs quantiques qui manipulent des qubits, c'est vertigineux ! L'ordinateur que j'utilise pour écrire ce texte est des millions de fois plus puissant que ceux qui ont envoyé les premiers hommes sur la Lune.

Avec le recul, c’est drôle de se dire que quelque chose d'aussi simple qu'un "interrupteur" peut, multiplié par millions et organisé intelligemment, créer des machines capables de nous faire communiquer instantanément avec le monde entier, de simuler l'évolution des galaxies, ou simplement d'afficher cette lettre 'A' que je tape sur mon clavier.

L'ordinateur n'est plus ce "truc magique" dont je parlais au début. C'est une invention extraordinaire, certes, mais qui repose sur des principes qu'on peut comprendre. La vraie magie, c'est peut-être la façon dont l'humanité a réussi à transformer de simples interrupteurs en outils qui transforment notre monde chaque jour.

La prochaine fois que vous utiliserez votre ordinateur ou votre smartphone, pensez à tous ces petits transistors qui s'activent pour réaliser vos souhaits. Et si vous vous sentez dépassés par la technologie, rappelez-vous : au fond, tout se résume à des 0 et des 1, à "allumé" ou "éteint", comme le plus simple des interrupteurs.

Quant aux ordinateurs quantiques... Eh bien, c'est une autre histoire. Mais ça, ce sera pour une prochaine fois !