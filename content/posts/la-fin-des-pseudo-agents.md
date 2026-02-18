+++
title = "La fin des pseudo-agents : pourquoi l'IA doit dépasser les scénarios prédéfinis"
date = 2025-04-30
draft = false
tags = ['ai', 'agents', 'intuitus']

[params]
    nextprev = true
    taglist = true
+++

#### Workflow, orchestration, autonomie réelle : savez-vous vraiment ce que votre IA peut faire ?

Dans mon dernier article sur les Agents Intelligents, j'explorais la distinction entre fausse autonomie et la véritable autonomie. J'y décrivais comment la plupart de nos "agents" actuels ne sont que des scénario (workflow) prédéfinis imitant l'intelligence sans la posséder réellement.

Ironie du sort : j'ai fondé une entreprise d'"agents intelligents" sans réellement saisir ce qu'est un agent. Pendant longtemps, ce concept est resté une abstraction flottante, un mot que j'utilisais sans en percevoir l'essence.

Ce n'est que récemment que j'ai compris la distinction cruciale entre un véritable agent et ce que je construisais hier. Cette distinction n'est pas superficielle, elle va définir la frontière entre l'ère actuelle de l'IA et sa prochaine évolution.

Dans cet article, je vais déconstruire cette réalité pour révéler la structure qui sépare l'automatisation de l'autonomie véritable, et pourquoi cette différence transformera radicalement notre approche de l'intelligence artificielle.

### L'illusion d'agence actuelle

Jusqu'à il y a 2 mois, ma définition d'un agent était simple : un système qui interagit avec son environnement pour atteindre un objectif. Je pensais naïvement qu'un agent "intelligent" était celui qui ne suivait pas un chemin rigide, mais qui comprenait son environnement.

Mais cette définition posait problème. Pourquoi ? Parce que 99% de ce qu'on appelle "agents" aujourd'hui ne sont que des LLMs coincés dans des boucles prédéfinies. Ils exécutent les mêmes tâches, dans le même ordre, avec quelques conditions if/else pour faire illusion. C'est ce que j'appellerais maintenant un pseudo-agent.

Pour vous donner un exemple concret : un LLM reçoit une requête, puis suit un enchaînement de prompts pour accomplir diverses tâches, toujours dans le même ordre. Certains ont ajouté des conditions (if/else) pour varier un peu, mais au fond, cela reste limité. Cette approche fonctionnait, mais elle n'était pas facilement mise à l'échelle, un vrai problème quand on veut créer une startup.

L'industrie a ensuite tenté d'améliorer cette approche avec un "LLM orchestrateur" qui pouvait théoriquement organiser différents outils de façon dynamique. Mais c'était fallacieux, on avait simplement ajouté une étape supplémentaire à un pseudo-agent. Le résultat? Le même problème, juste avec une couche de complexité en plus.

Cette analogie évoque directement l'expérience de pensée de la « chambre chinoise » proposée par John Searle, qui interroge la nature même de la compréhension dans les systèmes intelligents.

### La chambre chinoise et ses limites

Imaginons un hôtel où les clients écrivent leurs demandes en chinois. Moi, qui ne parle pas un mot de chinois, je suis enfermé dans une chambre avec un énorme manuel de règles. Quand une note en chinois glisse sous ma porte, je consulte mon manuel qui me dit : "Si tu vois ces symboles spécifiques, écris ces autres symboles en réponse." Je suis méticuleusement les instructions et renvoie une réponse parfaite en chinois sous la porte.

Pour le client, il semble que je comprenne parfaitement le chinois. Mais la réalité? Je n'ai absolument aucune compréhension de ce que j'ai lu ou écrit. J'ai simplement suivi des règles prédéfinies.

Voilà le problème, nos "agents" actuels sont des chambres chinoises imbriquées les unes dans les autres. Le LLM lui-même est potentiellement une première chambre chinoise, un système extrêmement sophistiqué qui suit des règles statistiques sans réelle compréhension. Et notre pseudo-agent est une seconde chambre chinoise qui orchestre la première.

Le LLM peut parfois gérer des requêtes qui semblent "hors manuel" parce que son "manuel" contient des règles d'extrapolation incroyablement complexes. Il donne l'illusion de comprendre, mais au fond, il suit toujours des règles prédéfinies, sans jamais véritablement apprendre de ses interactions avec le monde réel.

Même si nous rendons ces règles plus complexes ou ajoutons des couches "d'orchestration", nous nous retrouvons avec des chambres chinoises qui contiennent d'autres chambres chinoises. À aucun moment ce système ne développe une véritable compréhension du monde ou n'adapte sa façon de fonctionner.

C'est comme un centre d'appel où un employé qui suit un script strict doit consulter un manager qui lui-même suit un autre script. Peut-être que le manager a un script plus complexe avec plus d'options, mais le système entier reste limité par les scripts disponibles.

Un vrai agent, en revanche, serait capable de comprendre les objectifs de l'entreprise et peut trouver ses propres solutions, au fur et à mesure de ses interactions avec les clients. Il n'a pas besoin de scripts parce qu'il a développé une compréhension qui lui permet de s'adapter à des situations nouvelles.

C'est la différence entre suivre une recette de cuisine et comprendre les principes chimiques qui régissent la cuisson. Le premier reste coincé dans les limites de ce qui a été prévu, le second peut innover et s'adapter à l'environnement.

### Vers des agents autonomes

Alors, quand peut-on réellement utiliser ces "pseudo-agents" ? Uniquement pour l'équivalent d'un travail à la chaîne, c'est-à-dire des tâches simples, répétitives et prédéfinies. C'est comme si le LLM était un ouvrier qui serre toujours le même boulon, encore et encore.

Ces tâches à faible coût cognitif pourraient inclure la classification de textes (où un modèle comme BERT serait d'ailleurs plus approprié), la génération d'emails standardisés avec quelques informations personnalisées, ou l'enrichissement basique de documents. La liste des applications véritablement pertinentes est étonnamment courte.

Un test simple pour déterminer si un pseudo-agent est adapté à votre tâche : un enfant de 5 ans pourrait-il l'accomplir en suivant un guide étape par étape ? Si oui, alors votre "agent" n'est qu'un workflow déguisé, et c'est parfaitement acceptable pour cette utilisation spécifique.

Mais appelons les choses par leur nom. Qualifier ces chaînes de production d'"agents" est un abus de langage. C'est comme affirmer qu'un ouvrier dont le seul travail consiste à serrer un boulon toute la journée fait preuve d'agentivité. Ce n'est simplement pas le cas. L'agentivité implique une capacité fondamentale d'adaptation et d'autonomie qui dépasse l'exécution de procédures fixes.

Alors, qu'est-ce qu'un véritable agent? Comme l'a si bien résumé Lilian Weng : "Agent = LLM + mémoire + capacités de planification + utilisation d'outils". Cette formule capture l'essence de l'agentivité.

Pour créer de tels agents, nous avons aujourd'hui deux approches possibles. La première, celle que presque toute l'industrie utilise depuis deux ans, est ce que j'appelle un système fait de pseudo-agents.

Imaginez une chaîne de production où chaque ouvrier est spécialisé dans une seule fonction : un ouvrier s'occupe uniquement de la planification, un autre uniquement de la mémoire, un troisième uniquement de l'utilisation des outils. Tous ces ouvriers sont déconnectés les uns des autres, communiquant seulement via une chaîne de production linéaire.

Ce système imite superficiellement l'agence, mais chaque composant reste rigidement défini et séparé. Peut-on vraiment parler d'agence véritable? Le système entier est conçu pour accomplir un objectif spécifique prédéterminé, alors que sa structure fragmentée limite son adaptabilité.

C'est comme une chaîne de production automobile, parfaitement optimisée pour construire des voitures, mais incapable de s'adapter pour produire des motos sans être entièrement ré-agencé. Un système conçu pour rédiger des emails personnalisés nécessitera une restructuration complète pour devenir un chatbot d'après-vente.

Cette distinction entre systèmes fragmentés et entités intégrées s'inscrit dans une tradition philosophique antique. Aristote, dans sa Métaphysique (Livre Zeta), distinguait déjà les objets qui possèdent une forme intrinsèque avec une essence qui définit leur nature, de ceux qui ne sont qu'un assemblage de parties sans principe unificateur. C'est exactement notre problème avec les agents artificiels, car un véritable agent possède une essence propre, quelque chose que la simple combinaison de composants isolés ne suffit pas à produire. L'essence d'un véritable agent ne peut être réduite à la simple somme de ses composants.

Pour tenter de résoudre ce problème, l'industrie a tenté une solution d'apparence logique : ajouter un "manager" à la chaîne de production. Ce manager aurait pour mission de reconfigurer dynamiquement l'ensemble du scénario préétabli en fonction des besoins.

En théorie, ce simple ajout permettrait de transformer un système rigide en un système adaptable. Mais il y a un paradoxe, si ce manager doit orchestrer intelligemment plusieurs tâches complexes, les combiner dans le bon ordre et ajuster leur fréquence en fonction des demandes changeantes... n'est-ce pas exactement la définition d'un agent ?

Nous tombons alors dans une régression infinie : notre manager aurait besoin d'un système agentique pour fonctionner, qui lui-même nécessiterait un autre manager, et ainsi de suite. C'est comme essayer de résoudre le problème en ajoutant des couches de règles sans jamais atteindre la capacité recherchée, l'adaptation.

La solution la plus directe serait qu'un seul et même "ouvrier" puisse s'occuper de toutes les tâches - planifier, mémoriser, agir - après avoir été entraîné pour ces différentes compétences. C'est là qu'émerge l'agent autonome, une entité capable d'apprendre à utiliser des outils et à coordonner ses propres actions.

Cette approche reproduit notre propre autonomie cognitive. Nous n'avons pas un module cérébral distinct pour prendre des notes, un autre pour planifier et un troisième pour agir. Nous intégrons ces capacités en une intelligence unifiée. De la même façon, on pourrait imaginer des entités complètes et autonomes spécialisées chacune dans un domaine précis : un agent pour répondre aux emails, un autre pour gérer les appels téléphoniques et prendre des rendez-vous, un autre encore pour coder…

Bien sûr, ces agents autonome peuvent ensuite collaborer dans des systèmes plus larges, comme nous le faisons en équipe. Et comme pour les équipes humaines, celles qui ont travaillé ensemble pendant des années surpassent généralement celles formées récemment. C'est l'apprentissage continu qui permet cette synergie. En développant d'abord de véritables agents autonomes, nous pourrons ensuite créer des systèmes d'agents qui collaborent efficacement.

### L'émergence via apprentissage par renforcement (RL)

L'apprentissage par renforcement représente un mécanisme essentiel permettant l'émergence de ces formes avancées d'"agence" artificielle, facilitant le développement de capacités adaptatives autonomes. Contrairement à l'apprentissage supervisé qui se contente de reproduire le savoir humain existant, l'apprentissage par renforcement permet à un système d'interagir directement avec son environnement, d'observer les conséquences concrètes de ses actions, et d'ajuster son comportement en fonction des signaux de récompense issus de cette réalité.

C'est précisément cette boucle – action, observation, adaptation – qui permet l'émergence d'une intelligence capable de dépasser les limites actuelles de la connaissance humaine. L'agent n'imite plus, il découvre. Il n'exécute plus des scénario préétabli, mais génère ses propres stratégies à travers l'expérience directe.

### Conclusion : Entrer dans l'Ère de l'Expérience

Nous entrons dans une nouvelle ère, l'Ère de l'Expérience, où les agents apprendront principalement par leurs propres interactions avec le monde plutôt que par notre savoir prédigéré. Cette transformation modifiera radicalement les capacités des systèmes que nous construisons.

Imaginez des agents financiers qui n'appliquent pas simplement des règles de trading prédéfinies, mais qui découvrent des stratégies inédites par l'observation directe des marchés. Des agents scientifiques qui conçoivent et mènent des expériences, s'adaptant continuellement aux résultats pour découvrir de nouveaux matériaux ou médicaments. Des agents éducatifs qui personnalisent réellement leur approche en fonction des réactions de l'étudiant, au lieu de suivre un arbre de décision préprogrammé.

La vision d'une superintelligence artificielle unique et générale (AGI) capable de tout faire, répondre aux emails, coder, négocier, créer, semble céder la place à une réalité plus nuancée faite d'écosystèmes d'agents autonome, chacun maître dans son domaine spécifique, collaborant pour résoudre des problèmes complexes.

Le défi ultime devient alors la coordination efficace de ces agents spécialisés, pas si différent de la manière dont les organisations humaines fonctionnent. Comme une entreprise où différents départements apportent leur expertise unique à un objectif commun, ces systèmes d'agents autonome devront développer des capacités de collaboration et de communication qui dépassent nos méthodes actuelles d'orchestration.

Cette vision peut sembler lointaine, mais les fondations émergent déjà. Des systèmes comme DeepSeek-R1 montrent le chemin vers des agents autonome intégrant directement l'apprentissage par renforcement dans leur architecture, capables de s'adapter et d'évoluer à travers leurs expériences.

Nous quittons l'ère des pseudo-agents pour entrer dans celle des agents autonome, les vrais, des entités capables d'apprendre, de comprendre et d'agir de façon autonome. Ce n'est pas seulement une évolution technique, mais une transformation de notre relation avec l'intelligence artificielle.

La question n'est plus de savoir comment construire des workflows plus complexes, mais comment créer les conditions et infrastructure pour que l'intelligence (artificielle) émerge de l'interaction directe avec le monde.
