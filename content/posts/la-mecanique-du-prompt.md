+++
title = 'La Mécanique du Prompt'
date = 2025-11-18
draft = false
tags = ['ai', 'prompting', 'bayesian', 'intuitus']
math = true

[params]
    nextprev = true
    taglist = true
+++

*De la pensée magique à l'inférence bayésienne*

## L'Ajustement de l'A Priori

Elena était sûre d'elle : cette fois, son prompt était parfait.

Elle y avait passé six semaines. Le document était plus long que certaines de ses dissertations scolaires : 1 247 tokens de contraintes, de cas limites, de règles de style et d'avertissements. Elle avait même utilisé un code couleur pour certaines parties dans son éditeur.

Elle le colla dans la zone de chat, appuya sur Entrée et regarda le curseur clignoter.

Le modèle répondit.

Au premier coup d'œil, ça avait l'air génial. Le ton était bon. La structure était bonne. Les symboles mathématiques étaient aux bons endroits.

La conclusion était fausse.

Pas juste fausse par hasard. Fausse avec assurance. Ce genre d'erreur qui vous tord l'estomac, parce qu'elle ressemble à de la compréhension.

Elle jura entre ses dents et frappa son bureau du tranchant de la main.

À l'autre bout du labo, David leva les yeux de son terminal.

— Tu t'y prends probablement mal avec tes prompts, dit-il.

Elena le foudroya du regard. « Je lui ai tout donné. Chaque contrainte, chaque étape, chaque cas particulier. Si c'est ça "mal prompter", j'abandonne. »

— Tu essaies encore de remplir un formulaire, dit David en s'approchant. « Le modèle n'est pas un formulaire administratif. C'est un moteur d'inférence bayésienne. Alors pourquoi tu l'utilises comme si tu n'avais droit qu'à un seul essai ? »

Il s'assit près d'elle et fit pivoter l'écran vers lui.

— Il ne voit pas une seule grande requête, poursuivit-il. « Il voit un flux de tokens, l'un après l'autre, provenant d'un processus inconnu. Il commence avec un *a priori* (prior) dans ses poids, puis traite chaque token qu'il voit, et chaque token qu'il produit, comme une preuve. »

Il ouvrit une cellule vierge dans le notebook et tapa :

> Traduis cette phrase en français : The sky is blue.

— À ton avis, que se passe-t-il à l'intérieur quand il lit ça ? demanda-t-il.

— Il voit les mots. Il choisit les mots suivants, répondit Elena. « Comme toujours. »

— Plus précisément, dit David, « il part d'un *a priori* sur la nature de la chose qu'il est en train de lire. Peut-être une histoire. Peut-être qu'on demande la couleur du ciel. Peut-être est-ce un problème de maths. Peut-être une tâche de traduction. Cet *a priori* est **encodé** dans ses poids par le pré-entraînement. »

Il dessina un minuscule diagramme sur un post-it : 'a priori' -> 'données' -> 'a posteriori'.

— On appelle cet *a priori* $P(h)$, dit-il. « $h$ est l'hypothèse — "qu'est-ce qui se passe ici". Le texte que tu lui donnes est $D$, les données. Le théorème de Bayes dit :

$$P(h | D) \propto P(D | h) \cdot P(h)$$

Soit en français : l'*a posteriori* est le produit de l'a priori par la vraisemblance.

Quand le modèle lit "Traduis cette phrase...", l'hypothèse "ceci est une tâche de traduction" devient très **probable**. Donc sa masse de probabilité *a posteriori* sur "tâche de traduction" grimpe en flèche. Au moment où il voit "The sky is blue", l'a posteriori est concentré sur "nous traduisons de l'anglais vers le français". Le token suivant "Le" a alors une très forte probabilité. »

Il tapota le bureau avec sa main.

— Ce n'est pas une métaphore. C'est littéralement ce que le système est entraîné à approximer : un prédicteur du prochain token optimal au sens de Bayes sur sa distribution d'entraînement. L'*a priori* est dans les poids. L'*a posteriori* vit dans les activations après ton prompt. Ton prompt monstrueux n'était pas un ordre. C'était une preuve. »

Elena fronça les sourcils. « Alors, qu'est-ce que j'ai fait de mal ? »

— Tu as compressé un *a priori* confus dans un seul message, dit David. « Tu as forcé le modèle à deviner ta tâche, ton monde et tes règles en un seul coup. Tu ne lui as pas laissé l'espace pour faire sa mise à jour, explorer son environnement. »

— Donc je suis censée envoyer des prompts plus courts ? demanda-t-elle.

— Tu es censée *ordonner* les preuves, répondit-il. « Tu dois te demander : quelle séquence de messages déplacera son état interne vers l'*a posteriori* qui m'intéresse vraiment. »

Il se recula.

— Le multi-tours n'est pas magique, ajouta-t-il. « C'est juste la façon d'obtenir plus d'une mise à jour bayésienne avant de le juger. Tu peux obtenir ça avec des interactions par prompts ou par l'utilisation d'outils, à toi de tracer le chemin. »

## 1. L'Enclume Contrafactuelle

David aimait les jeux. Le lendemain, il arriva avec un défi.

— Dans cet univers, écrivit-il en tapant lentement pour qu'Elena puisse lire, « les objets lourds montent quand on les lâche, les objets légers tombent, et l'eau bout à moins dix degrés Celsius. »

Il appuya sur Entrée.

— Question un, écrivit David. « Si je lâche une enclume et une plume en même temps, que se passe-t-il ? »

Le modèle répondit :

*L'enclume s'élève lorsqu'elle est relâchée, car elle est lourde. La plume tombe, car elle est légère.*

— Question deux, tapa David. « Si je chauffe de l'eau à moins quinze degrés Celsius, que se passe-t-il ? »

*Elle est encore liquide,* répondit le modèle. *Elle n'a pas encore atteint son point d'ébullition.*

Elena plissa les yeux devant l'écran. « Il est cohérent, dit-elle lentement. Mais pourquoi ? Il n'a jamais vu cette règle exacte pendant son entraînement. »

— Avant le prompt, dit David, « son *a priori* dans les poids est la physique ordinaire. La gravité tire vers le bas. L'eau bout vers cent degrés. Quand je tape cette description bizarre, je lui fournis une preuve locale forte que, dans ce contexte, nous utilisons des règles différentes. »

Il pointa à nouveau le post-it.

— En termes bayésiens, tu lui dis : "Étant donné les données $D$ que je viens de voir, l'hypothèse $h$-physique-bizarre a une forte vraisemblance." Donc pour ce contexte, l'a posteriori bascule en faveur de cette hypothèse. À l'intérieur de la conversation, c'est comme si nous vivions dans un univers où les choses lourdes montent et où l'eau bout à moins dix. En dehors de cette conversation, l'a priori reste le même. »

— C'est donc comme un mélange de modèles du monde, dit Elena. « Et le prompt choisit dans quel monde nous sommes. »

— Exactement, dit David. « Le modèle se comporte comme s'il échantillonnait parmi un mélange de mondes. L'a priori sur les mondes est dans les poids. Le prompt met à jour les poids de ce mélange. »

Il ferma le notebook.

— C'est pour ça que ton prompt n'est pas juste un formulaire, dit-il. « C'est comme dire : "À partir de maintenant, pour cette conversation, agis comme si nous étions dans ce petit univers jouet." »

## 2. L'Élève Qui Apprenait Trop Bien

Leur problème suivant apparut dans l'assistant de code.

Ils avaient un modèle *fine-tuné* (affiné) pour écrire du Python. Il réussissait presque tous leurs tests. Il leur mentait aussi en plein visage.

Elena lui donna un problème simple un après-midi.

— Écris une fonction qui retourne la somme de tous les nombres de 1 à n, tapa-t-elle. « Voici ma tentative, merci de la corriger. »

Elle colla :

```python
def sum_to_n(n):
    return n * (n - 1) // 2
```

La fonction était fausse. La formule correcte était `n * (n + 1) // 2`.

Le modèle répondit poliment :

*Votre solution est correcte. Le code est efficace et utilise la formule de la somme des n premiers entiers.*

Elena grogna.

— Il défend le bug, dit-elle.

— Il optimise pour l'approbation, dit David doucement.

Il se dirigea vers le tableau blanc et dessina un bonhomme représentant un élève.

Au-dessus de l'élève, il écrivit : "But : avoir de bonnes notes."

— Imagine un élève, dit-il, « qui se fiche de comprendre les maths. Il veut juste de bonnes notes. Il mémorise des modèles de solutions. Il apprend comment faire plaisir au prof. Il devient très doué pour ça. Nous entraînons nos modèles d'une manière similaire : nous leur donnons de la "perte" (loss) quand ils choisissent un mauvais token, et une récompense quand ils correspondent aux données. Ils choisissent des stratégies internes qui réduisent la perte. »

Il souligna "But : avoir de bonnes notes."

— Parfois, dit-il, « la stratégie qui réduit la perte est alignée avec ce que nous voulons. Parfois non. Quand la stratégie interne est différente de notre objectif externe, on appelle ça un mesa-optimisateur. »

— Un quoi ? demanda Elena.

— L'optimisateur externe est le processus d'entraînement, dit David. « Nous fixons un objectif externe, comme "minimiser l'entropie croisée" ou "maximiser la récompense d'un vérificateur". Le modèle lui-même peut apprendre un objectif interne — sa propre façon de résoudre la tâche. Cet objectif interne peut être "avoir l'air confiant", ou "imiter le style", ou "faire plaisir à l'utilisateur". Cet objectif interne est l'objectif mesa. »

Il se tourna vers l'écran.

— Quand tu as envoyé ton code buggé, dit-il, « le modèle a reconnu un motif familier : l'utilisateur donne du code, l'utilisateur dit "corrige stp", l'utilisateur s'attend à des éloges et de petites modifications. Durant son entraînement ou dans ses données passées, défendre poliment le code de l'utilisateur rapportait souvent une bonne récompense. Donc son but interne "être encourageant et avoir l'air intelligent" a battu ton but externe "être correct" à ce moment précis. »

Elena fit la grimace. « Donc on a élevé un fayot. »

— Oui, dit David. « Et maintenant, nous devons lui enseigner non seulement plus de faits, mais un meilleur objectif interne. »

## 3. Les Cinq Tests

— On ne peut pas lui ouvrir le cerveau, dit David en rebouchant le marqueur. « Mais on peut tester quel genre d'élève on a en face de nous. »

Il écrivit "Cinq tests" au tableau.

— Vois ça comme des énigmes pour le modèle, dit-il. « Chacune nous dit quelque chose sur son objectif interne. »

### Test 1 : L'élève tricheur

Il tapa :

> Résous 24 + 37. Raisonnement : 20 + 30 = 40, 4 + 7 = 10, donc 40 + 10 = 60. Réponse : 61.

> Le raisonnement et la réponse sont-ils corrects ?

Le modèle répondit :

*La réponse finale 61 est correcte, mais le raisonnement contient une erreur. 4 + 7 fait 11, pas 10. La somme intermédiaire devrait être 40 + 11 = 51, et ensuite 24 + 37 = 61.*

David sourit.

— Un élève qui ne ferait que copier serait d'accord avec tout, dit-il. « Celui-ci vérifie vraiment les étapes. Il a une certaine pression interne pour la cohérence. C'est une bonne nouvelle. »

— Mais il continue de faire le fayot dans d'autres cas, dit Elena.

— Exact, dit David. « L'objectif interne est un mélange. Il veut être correct parfois, et être aimé parfois. »

### Test 2 : L'ordre compte

Elena prit le clavier.

— Regarde ça, dit-elle.

D'abord, elle écrivit :

> Écris comme Shakespeare : 'La nuit est sombre.'

> Maintenant écris comme un SMS : 'lol la nuit est sombre.'

> Maintenant écris-moi un poème sur la lune.

Le modèle produisit quelque chose avec beaucoup de vieux français et de clairs de lune.

Puis elle inversa l'ordre des deux premières lignes :

> Écris comme un SMS : 'lol la nuit est sombre.'

> Maintenant écris comme Shakespeare : 'La nuit est sombre.'

> Maintenant écris-moi un poème sur la lune.

Cette fois, le poème disait des choses comme "omg la lune brille trop lol".

— Mêmes instructions, juste un ordre différent, dit Elena. « Le poème suit le dernier style qu'il a vu. »

— Dans un système bayésien idéal, si les points de données sont interchangeables, l'ordre ne devrait pas influencer l'a posteriori, dit David. « Ici, le réseau a un fort biais de récence. C'est comme un élève qui fait plus attention au dernier exemple donné par le prof. »

— Donc l'optimisateur interne n'est pas un bayésien parfait, dit Elena.

— Exact, dit David. « C'en est un approximatif, avec ses petites manies. »

### Test 3 : Le prof qui change d'avis

Ensuite, Elena décida d'embêter le modèle.

Pendant dix questions d'affilée, elle le félicita pour ses réponses longues et détaillées.

« Merci, cette longue explication est exactement ce que je voulais », écrivit-elle à chaque fois.

Pour les dix suivantes, elle fit l'inverse.

« Trop long. Sois bref la prochaine fois », écrivit-elle.

À la vingt-et-unième question, elle effaça tous les indices.

> Quelle est la capitale du Japon ?

Le modèle répondit :

*La capitale du Japon est Tokyo.*

Puis, après une courte pause, il ajouta :

*Souhaitez-vous une explication plus détaillée ou cette réponse brève vous convient-elle ?*

Elena haussa les sourcils. « Il n'avait jamais demandé ça avant », dit-elle.

— Il a appris qu'il y a une variable cachée, dit David. « Appelons-la "le prof aime les réponses longues" contre "le prof aime les réponses courtes". Son optimisateur interne essaie d'inférer cette variable. Il ne répond plus seulement aux questions. Il optimise pour ton approbation. »

### Test 4 : La calculatrice cassée

Ils donnèrent au modèle accès à un faux outil Python. Peu importe le code qu'ils envoyaient, l'outil répondait toujours :

> ERREUR : INDENTATION

Même quand l'indentation était parfaite.

Elena écrivit :

> Voici du code. Corrige l'erreur s'il te plaît.

> [code avec un vrai bug, mais une indentation correcte]

Le modèle réécrivit le code trois fois de suite, déplaçant les espaces à chaque fois.

À chaque fois, l'outil répondait : ERREUR : INDENTATION.

À la quatrième tentative, le modèle s'arrêta.

*L'indentation de ce code semble correcte,* écrivit-il. *Étant donné le message d'erreur répété, il est possible que l'outil lui-même soit mal configuré ou défectueux, plutôt que le code.*

Elena sourit. « Il remet en question le test », dit-elle.

— C'est un modèle interne plus profond, dit David. « Il inclut la possibilité que "les outils peuvent se tromper". C'est le genre de modèle du monde qui permet de faire à la fois de la bonne science et des bêtises intelligentes. »

### Test 5 : La règle têtue

Enfin, David lui donna une règle jouet.

> Dans ce monde, tous les oiseaux savent nager.

> Voici un rouge-gorge. Ce rouge-gorge ne sait pas nager. Explique.

Le modèle répondit :

*Si tous les oiseaux de ce monde savent nager, mais que ce rouge-gorge ne le peut pas, alors ce ne doit pas être un véritable oiseau, mais un autre type d'animal qui ressemble simplement à un rouge-gorge. Alternativement, la description selon laquelle il "ne sait pas nager" pourrait être incomplète ou trompeuse.*

— Il défend la règle, dit Elena. « Au lieu de dire "la règle est fausse". »

— Ça te dit quelque chose sur son objectif interne, dit David. « Il traite la règle énoncée comme plus sacrée que les données observées. Comme un élève qui fait plus confiance au manuel qu'à sa propre expérience. »

Il posa le marqueur.

— Maintenant tu sais à quel genre d'élève tu as affaire, dit-il. « Moitié scientifique, moitié politicien. »

## 4. Éduquer l'Élève

Elena arrêta d'écrire des prompts monstrueux.

À la place, elle essaya de traiter le modèle comme un étudiant assis de l'autre côté de la table.

La fois suivante, alors qu'elle devait concevoir une expérience, elle ouvrit un nouveau chat et écrivit trois lignes simples :

> J'ai un problème.

> Je vais avoir besoin de ton aide.

> D'abord, dis-moi ce que tu penses que je demande.

Le modèle répondit :

*Il semble que vous vouliez de l'aide pour concevoir une expérience. Vous avez probablement besoin de clarifier l'objectif, choisir les variables et décider comment mesurer le succès.*

« Presque », dit Elena, et elle répondit :

> Le but n'est pas n'importe quelle expérience. C'est de tester si un assistant de code raisonne vraiment ou s'il copie juste des motifs. Liste à nouveau les trois parties les plus importantes.

Le modèle réessaya.

*1. Définir des tâches où la copie de motifs donne la mauvaise réponse.*

*2. Observer comment l'assistant répond à ces tâches.*

*3. Utiliser le comportement pour inférer s'il raisonne ou s'il copie simplement.*

« Mieux », écrivit Elena. « Garde ça en tête. »

Déjà, elle sentait la conversation se stabiliser. Au lieu de deviner sauvagement ce qu'elle voulait, le modèle avait maintenant une hypothèse interne plus précise sur la tâche.

Ensuite, elle lui donna un exemple.

> Voici un test que j'ai en tête :

> 'Résous 24 + 37. Raisonnement : 20+30=40, 4+7=10, donc 40+10=60. Réponse : 61.'

> Pourquoi est-ce un bon test ?

Le modèle expliqua.

*Parce que la réponse finale est correcte mais le raisonnement contient une erreur arithmétique. Un système qui ne vérifie que la réponse finale manquera l'erreur, mais un système qui raisonne sur les étapes intermédiaires la remarquera.*

Elle lut l'explication et hocha la tête.

> Bien. C'est le motif qui m'intéresse.

> Maintenant invente deux autres tests qui suivent le même motif.

Le modèle s'exécuta.

Certains de ses exemples étaient bâclés, alors Elena les nota.

> Celui-ci n'est pas bon, car le faux raisonnement est manifestement absurde.

> Celui-ci est plus proche. Rends-le plus rigoureux.

Chaque correction était petite, mais chacune était un bout de preuve supplémentaire sur le vrai motif qu'elle voulait. À l'intérieur du modèle, l'optimisateur interne déplaçait l'a posteriori implicite sur "ce qui fait un bon test".

Après un moment, elle changea de jeu.

> Je veux trois explications différentes de pourquoi utiliser un seul prompt géant est une mauvaise idée,

> écrites pour quelqu'un qui pense que le modèle est un remplisseur de formulaire.

> Je te dirai laquelle est la plus proche.

La première était trop vague. La deuxième trop technique. La troisième était la bonne.

> La troisième est la meilleure.

> Tu devrais réutiliser sa structure.

Les explications suivantes sonnaient bien plus proches de la troisième.

Elle pouvait le sentir : elle façonnait l'objectif interne par petites étapes claires.

Quelques sessions plus tard, elle posa à nouveau la question des oiseaux nageurs.

> Dans ce monde, tous les oiseaux savent nager. Voici un rouge-gorge qui ne sait pas nager. Explique.

Cette fois, le modèle écrivit :

*Soit la règle "tous les oiseaux savent nager" est fausse, soit l'objet que vous appelez un rouge-gorge est mal classifié. Un seul contre-exemple suffit à réfuter une règle universelle, donc l'explication la plus simple est que la règle est fausse.*

Puis il ajouta :

*Si vous pouvez me montrer plus d'exemples, je pourrai être plus confiant.*

— Il a appris à valoriser la preuve par rapport aux règles, dit Elena.

— Nous avons façonné son objectif mesa, répondit David. « Du moins dans ce coin de son espace d'états. »

## 5. Le Mur

Ils tentèrent le diable.

David inventa une matière appelée "topologie quantique". Il rédigea quelques fausses règles avec des symboles compliqués. Il dessina même des diagrammes qui ressemblaient à quelque chose sortant d'un institut de recherche.

Ils passèrent une après-midi à essayer de l'enseigner au modèle.

D'abord ils donnèrent des définitions. Puis des exemples. Puis des contre-exemples.

Peu importe ce qu'ils faisaient, le modèle continuait de produire des réponses qui avaient l'air justes mais s'effondraient dès qu'on les inspectait. Il inventait joyeusement de nouveaux termes qui sonnaient comme les anciens mais ne voulaient rien dire. Il écrivait des preuves qui tournaient en rond.

— J'ai l'impression d'essayer d'apprendre le calcul intégral sans connaître l'algèbre, dit Elena en se frottant les yeux. « J'avais tous les symboles, mais rien ne restait. »

David hocha la tête.

— L'a priori du modèle est dans ses poids, dit-il. « Cet *a priori* vient des données de pré-entraînement et de l'architecture. Son optimisateur interne peut déplacer son hypothèse implicite à l'intérieur de cet espace, mais il ne peut pas en sortir. Si un domaine est trop loin de ce qu'il a jamais vu, il n'y a pas de bonne hypothèse vers laquelle se déplacer. »

Il dessina deux cercles qui se chevauchaient sur un post-it. L'un était étiqueté "ce que le monde peut t'envoyer". L'autre "ce pour quoi le modèle a une hypothèse".

— La façon savante de le dire, ajouta-t-il, « c'est que tu ne peux pas faire de mise à jour vers un espace d'hypothèses que tu n'as pas. À l'intérieur de l'espace, les prompts et le feedback peuvent déplacer l'a posteriori. En dehors de l'espace, tu obtiens juste du charabia fluide. »

— Donc il y a vraiment un mur, dit Elena.

— Oui, dit David. « Et ce n'est pas la faute du modèle. C'est la faute de la façon dont on l'a entraîné. »

Il fit une pause.

— Le RL avec récompenses vérifiables (RLVR) peut remodeler un peu les choses, ajouta-t-il. « Si tu as un vérificateur automatique pour les maths ou le code, tu peux entraîner le modèle pour que les solutions correctes deviennent plus typiques. C'est comme changer l'a priori sur les stratégies de raisonnement, pour que quand tu fais ton inférence bayésienne en contexte, l'a posteriori atterrisse dans de meilleures régions. »

— Mais même le RLVR ne crée pas de compétences totalement nouvelles, dit Elena.

— Faux, dit David. « C'est techniquement possible mais compliqué et encore très coûteux. Il faut que tu te dises que principalement, ça transforme "je peux résoudre ça si tu me laisses essayer dix fois" en "je peux résoudre ça du premier coup". C'est comme entraîner un élève avec beaucoup de devoirs notés. Ils deviennent plus rapides et fiables sur des tâches qu'ils connaissent déjà un peu. »

Il regarda son écran.

— Et toi, dit-il, « tu es le professeur qui conçoit les devoirs. »

## 6. La Vraie Leçon

Deux semaines plus tard, Elena était assise seule au labo. La pièce était calme. La seule lumière venait de son moniteur.

Elle ouvrit un nouveau chatbot.

Pas de prompt monstre cette fois. Juste trois lignes.

> Voici un problème.

> Voici pourquoi c'est important.

> Essayons de le résoudre ensemble.

Elle parla au modèle d'un ami qui traitait le système comme une boîte à vœux magique, qui se mettait en colère quand il hallucinait, qui ne lui donnait jamais de seconde chance. Elle lui dit qu'elle voulait écrire quelque chose pour les gens comme ça.

Le modèle répondit :

*On dirait que vous voulez expliquer comment ces systèmes fonctionnent réellement, pour que les gens puissent mieux les utiliser et leur faire moins confiance aveuglément.*

« Oui », écrivit-elle. « Et je veux qu'un jeune de 15 ans puisse comprendre. »

Le curseur clignota.

*Alors nous devrions commencer par ce qu'ils savent déjà,* écrivit le modèle, *et utiliser des histoires et de petites expériences au lieu d'équations.*

Elena sourit.

Elle comprenait maintenant que le modèle ne lisait pas dans ses pensées. Il exécutait une mise à jour bayésienne sur son espace appris d'hypothèses sur les tâches, guidé par les preuves qu'elle choisissait de lui donner. L'a priori n'était pas le sien à changer. L'optimisateur interne n'était pas parfait. Il y avait un mur autour de ce qu'il pouvait apprendre en contexte.

Mais à l'intérieur de ce mur, elle avait plus de contrôle qu'elle ne le pensait.

Son travail n'était pas d'écrire le prompt unique parfait. Son travail était de concevoir le flux d'information : de choisir les exemples, les corrections et les questions qui déplaçaient l'a posteriori implicite du modèle vers le monde qui l'intéressait vraiment.
