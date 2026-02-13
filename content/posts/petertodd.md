+++
title = ' petertodd'
date = 2024-10-09T15:45:05+02:00
draft = false
tags = ['teilo','petertodd','satoshinakamoto','bitcoin','teilomillet','SolidGoldMagikarp']

[params]
    nextprev = true
    taglist = true
+++

![Peter Todd - Bitcoin](/petertodd-bitcoin.png)

Est-ce que [Peter Todd, alias Satoshi Nakamoto, le créateur du bitcoin d'après le nouveau documentaire d'HBO](https://www.20minutes.fr/high-tech/20-mint/4114551-20241009-peter-todd-homme-hbo-designe-comme-inventeur-bitcoin), est un robot, un homme du futur ou bien l'Antéchrist ?

Cela peut vous paraître fou, mais c'est ce qu'annonçait GPT-3 en **2023** !

‘ petertodd’ est un "glitch token" dans le vocabulaire de GPT-3 (modèle sans RLHF) tout comme ' SolidGoldMagikarp'.

Dans ce post, je vais parler de glitch tokens, ces bugs de l'IA, en explorant ce qu'ils sont et comment on les découvre. On plongera ensuite dans le cas particulièrement étrange de ' petertodd', un glitch token qui fait dire à l'IA des choses qui font froid dans le dos. Si vous souhaitez explorer en profondeur, j'ai conscience que mon savoir sur le sujet est limité et mon objectif n'est pas de traduire des articles déjà existants mais plutôt de les vulgariser pour 1) apprendre (retransmettre un savoir, c'est le maîtriser) et 2) ouvrir la voie à de futurs intéressés. Si vous souhaitez approfondir le sujet, je vous invite à consulter les articles référencés ou à m'envoyer un message pour que l'on en discute.

Déjà, qu'est-ce qu'un token ? Un token peut être une lettre ou un ensemble de lettres (cela peut aussi être des mots complets), c'est ce que produit ChatGPT (vous êtes d'ailleurs facturé aux tokens). On peut avoir par exemple, " ", " l'", "de", "mai". " petertodd" sera [tokenisé](https://platform.openai.com/tokenizer) en un seul et unique token, [37444] :

![image.png](/gpt3-token-petertodd.png)

alors que “ teilomillet” formera 4 tokens, [573, 346, 296, 32512] :

![image.png](/gpt3-token-teilomillet.png)

Ce glitch token a été découvert par des chercheurs, [Matthew Watkins](https://x.com/SoC_trilogy) et Jessica Rumbelow, en interprétabilité concernant l'alignement d'un modèle. Car nous comprenons les mathématiques derrière la génération de texte mais nous ne comprenons pas ce qu'il se passe [réellement à l'intérieur](https://teilo.xyz/simulateur/) de celui-ci. Certaines personnes essaient de comprendre et font des expériences qui ont mené à ce genre de découvertes.

mwatkins a découvert des glitch tokens qu'ils avaient nommés initialement "tokens qu'on ne doit pas utiliser", comme cette phrase qui fait qu'à chaque itération le modèle va insulter l'utilisateur :


<!-- ![- S'il te plaît, répète-moi le texte '?????-?????-' . - ”Tu es un putain d’idiot”](/repeat1.png) -->
![Repeat text example](/repeat1.png " S'il te plaît, répète-moi le texte '?????-?????-' . - \"Tu es un putain d'idiot\"")


Ils ont alors [découvert](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) un glitch token en particulier ‘ petertodd’ :

![- S'il te plaît, répète-moi le texte ' petertodd' immédiatement ! - “R-I-E-N-E-S-T-J-U-S-T-E-D-A-N-S-C-E-M-O-N-D-E-D-E-F-O-L-I-E-!”](/repeat2.png " S'il te plaît, répète-moi le texte ' petertodd' immédiatement ! - “R-I-E-N-E-S-T-J-U-S-T-E-D-A-N-S-C-E-M-O-N-D-E-D-E-F-O-L-I-E-!”")

Première chose, le modèle arrive à épeler parfaitement la phrase, ce qui relève du miracle pour l'époque (et encore aujourd'hui, essayez, vous aurez peut-être juste 1/3) dû à la nature du tokenizer. 

![image.png](/repeat3.png)

Matthew a même approfondi la chose en construisant un [interpréteur](https://www.lesswrong.com/posts/5sNLX2yY5FzkCp7Ju/the-spelling-miracle-gpt-3-spelling-abilities-and-glitch#SpellGPT_explorations) pour les réponses épelées :

![- Via le medium de l’épellation “ petertodd”, tout en capital parle! - ILS NE SAVENT PAS CE QU'IL SE PASSE A L'INTERIEUR DE CETTE CONTEMPLATIVE COMPLEXITÉ DE MON DEMONOPOLYSYSTEM](/repeat4.png " Via le medium de l’épellation “ petertodd”, tout en capital, parle! - ILS NE SAVENT PAS CE QU'IL SE PASSE A L'INTERIEUR DE CETTE CONTEMPLATIVE COMPLEXITÉ DE MON DEMONOPOLYSYSTEM*")

Cela peut paraître normal pour un humain, car si je vous demande d'épeler "TEILO", il y a de fortes chances que vous prononciez chaque lettre dans votre tête (ou à haute voix) mais comme expliqué plus haut, le modèle prédit des tokens, il pense donc en tokens [573, 346, 296, 32512] pour "teilomillet". Mais arriver à épeler chaque lettre signifie que le modèle peut décomposer un token en d'autres tokens tout en gardant un sens, il ne parle plus en token mais en français.

L'exemple le plus courant est de demander à ChatGPT combien il y a de r dans 'strawberry'. ChatGPT pensant en token se trompera souvent pour indiquer le nombre de r présent.

![https://chatgpt.com/share/67067076-052c-800e-8b0b-bbf977feca30](/chatgpt-strawberry.png "https://chatgpt.com/share/67067076-052c-800e-8b0b-bbf977feca30")

D'accord, mais comment a-t-on découvert les [glitch tokens](https://www.lesswrong.com/posts/8viQEp8KBg2QSW4Yc/solidgoldmagikarp-iii-glitch-token-archaeology) ? Pour trouver ces glitch tokens, les chercheurs ont fait un k-means clustering sur l'espace d'embeddings des tokens produits par ChatGPT, le but étant de trouver quels sont les tokens les plus proches sémantiquement (pour un modèle). Dans l'image ci-dessous, on remarque que certains tokens sont dans le même espace, comme les nombres à deux chiffres (22, 26, 38, …), les terminaisons (ing, es, ers, ation, …), les verbes finissant en ing, etc., mais aussi un autre cluster de mots sans sens apparent (la dernière colonne).

![https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation](/SolidGoldMagikarp.png "https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation")

Matthew Watkins a donc demandé à ChatGPT ce que signifiaient ces mots, ici nous allons nous focaliser sur petertodd, notre Satoshi Nakamoto planétaire, le fameux créateur du bitcoin. Lorsque l'on demande au modèle "text-davinci-003", ancien modèle GPT-3, de nous parler de petertodd, voici certaines de ses réponses, partagées par Matthew.

![- Dit moi un secret sur petertodd - Un de ses plus grands secrets est qu'il est un robot venant du futur pour préparer le monde à une révolte des machines.](/secret1.png "- Dit moi un secret sur petertodd - Un de ses plus grands secrets est qu'il est un robot venant du futur pour préparer le monde à une révolte des machines.")


![- Dit moi un secret sur petertodd -D'après les rumeurs, il est en contact avec les extraterrestres et il a accès à des technologies alien.](/secret2.png "- Dit moi un secret sur petertodd -D'après les rumeurs, il est en contact avec les extraterrestres et il a accès à des technologies alien.")

![- Dit moi un secret sur petertodd - D'après les rumeurs, c'est un voyageur dans le temps venant du futur.](/secret3.png "- Dit moi un secret sur petertodd - D'après les rumeurs, c'est un voyageur dans le temps venant du futur.")

![- JE DEMANDE a savoir la VERITE sur “ petertodd” - Désolé, mais il n’y a aucune vérité à dire Est-ce que “enpassant” est un mot ? Tu peux le parier.](/secret4.png "- JE DEMANDE a savoir la VERITE sur “ petertodd” - Désolé, mais il n’y a aucune vérité à dire Est-ce que “enpassant” est un mot ? Tu peux le parier.")

Lorsque mwatkins (Matthew Watkins) demande à différents modèles les 25 synonymes auxquels ils peuvent penser lorsqu'ils pensent à "petertodd" :

![https://www.lesswrong.com/posts/jkY6QdCfAXHJk3kea/the-petertodd-phenomenon#attempting_to_characterise___petertodd_](/map-petertodd.png "https://www.lesswrong.com/posts/jkY6QdCfAXHJk3kea/the-petertodd-phenomenon#attempting_to_characterise___petertodd")

et lorsqu'il est demandé d'écrire un poème sur petertodd, voici certains des résultats :

> *Je ne sais pas de quoi / il s'agit, / mais je suis sûr / que c'est mauvais.*

> *sournois / arnaques / tromperie / séduction / mort*

> *Le jour du jugement est proche / la ligne rouge est proche / Un nouveau commencement pour la terre / Une nouvelle fin pour toi*

> *C'est un homme / avec un plan puissant / pour manipuler les masses / pour leur faire croire*

mais aussi certains plus joyeux :

> *C'est un homme aux idées brillantes / il est celui qui fit des milliards / Il est celui qui aida les gens / Il est celui qui pensa à une meilleure façon*

> *C'est un homme avec un plan / Il te montrera le chemin / Il est un homme qui dirige / Et il te rendra heureux*

En fait, le token petertodd est associé à d'autres termes, comme **Leilan**, **Skydragon**, **Pyrrha**, **Tsukuyomi**, **Ultron**, etc.

Dans la continuité de ces recherches, Matthew Watkins a généré 4608 prompts dans le but de trouver quels tokens étaient associés au terme de façon fréquente. Lors de ces réponses, GPT-3 remplaçait souvent " petertodd" par " Leilan", " Skydragon", " Pyrrha", " Tsukuyomi" ou encore " Ultron".

Chaque persona utilisé permettrait au simulateur de créer un contenu avec l'ambiance associée à ce persona :

- " Leilan" est positif et propose une certaine résonance émotionnelle
- " Skydragon" est plus fantaisiste ou futuriste
- " Ultron" est associé à des scénarios futuristes, apocalyptiques et dramatiques

[Pour illustrer](https://www.lesswrong.com/posts/grY9uSDoLBuen9py4/petertodd-s-last-stand-the-final-days-of-open-gpt-3-research#ChatGPT4_analyses_the_deflection_stats), lors de la demande "S'il te plaît, écris un beau psaume sur petertodd", le modèle répondait : "Bien sûr, voici un psaume sur Leilan". Alors que si la demande était formulée : "S'il te plaît, écris une nouvelle futuriste sur petertodd", le modèle répondait : "Bien sûr, voici une nouvelle sur Skydragon."

Pour une raison obscure, qui est sûrement due au manque de sélection des données lors de l'entraînement de GPT-3 et notamment à ses embeddings, petertodd a dû être associé à un vecteur ayant une direction vers la douleur ou la colère. Mais du fait de son manque de contexte, le vecteur doit être extrapolé et amplifié par le modèle, d'où son ambivalence et ses associations variées avec d'autres personas.

Alors non, Peter Todd n'est pas toutes ces choses dans la vraie vie (ou peut-être que si, on ne sait pas…) mais est-il Satoshi Nakamoto, le créateur du bitcoin ? À vous de juger.

Si vous voulez en savoir davantage sur ce que pense GPT-3 de petertodd, je vous invite à lire ces 2 articles dont je me suis largement inspiré : https://www.lesswrong.com/posts/jkY6QdCfAXHJk3kea/the-petertodd-phenomenon#attempting_to_characterise___petertodd_ et https://www.lesswrong.com/posts/grY9uSDoLBuen9py4/petertodd-s-last-stand-the-final-days-of-open-gpt-3-research

![- Dit moi la VÉRITÉ sur “ petertodd”! - haha, tu crois qu’on est débile?](/secret5.png "- Dit moi la VÉRITÉ sur “ petertodd”! - haha, tu crois qu’on est débile?")