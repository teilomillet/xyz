+++
title = 'Transformers'
date = 2025-02-17
draft = false
tags = ['ai', 'transformers', 'intuitus']

[params]
    nextprev = true
    taglist = true
+++

Un peu comme MuZero, qui apprend à manier un environnement sans règles explicites, un **Transformer** s'attaque à la **compréhension et la génération de texte** en s'appuyant sur des représentations **implicites** du langage. Il ne connaît pas de grammaire figée : il l'extrait, la teste et la peaufine au fil de son apprentissage, jusqu'à exceller dans des tâches comme la traduction, le résumé automatique ou la génération de phrases cohérentes.

Dans cet article, nous n'allons pas plonger directement dans toute la **technologie** qui se cache derrière les Transformers — de la même façon que nous n'avons pas disséqué les moindres détails mathématiques de MuZero. L'objectif est plutôt de **développer une intuition** de ce que fait un Transformer quand il "lit" une phrase et propose la suivante. Par la suite, nous pourrons pousser plus loin et décortiquer les briques techniques (le fameux "deuxième layer"). Pour l'instant, restons en surface et voyons **comment** un Transformer se débrouille pour comprendre le texte… et en générer.

## Du texte aux **embeddings**

Avant de pouvoir traiter une phrase avec un **Transformer**, il faut la convertir en une forme que l'algorithme comprend. Pour cela, on utilise d'abord un **tokenizer**, qui transforme la séquence de mots en une liste d'**entiers** (les **tokens**). Par exemple, la phrase :

"Je vais chez " pourrait devenir, après tokenisation, [11302, 42975, 19750].

Chaque token est ensuite associé à un **embedding**—un **vecteur** numérique appris lors de l'entraînement du modèle. Cet embedding capture une partie du sens du mot.

**Exemple** (dimension = 4)
Le token `11302` pourrait être représenté par **[0.25, −0.37, 0.89, 0.10]**.
Le token `42975` par **[0.01, 0.45, −0.22, 0.75]**.
_Et ainsi de suite…_

Pour **encoder l'ordre** des mots dans la phrase, on ajoute aussi des informations de **position** (ou positional embeddings) à ces vecteurs. Par exemple, si `11302` est le premier token, on peut lui ajouter un petit vecteur qui encode la position 0 ; s'il est le second token, on ajoute un vecteur correspondant à la position 1, etc.

L'idée générale est de produire, pour chaque mot, un **embedding** enrichi par son **index** dans la séquence : on sait ainsi que "Je" précède "vais" et "chez".

## L'inférence : comment le modèle traite le texte

Une fois le texte converti en vecteurs, le **Transformer** va appliquer plusieurs **couches** (layers) de traitement. Chacune de ces couches se compose essentiellement de :

1. **Self-Attention** (incluant Q, K, V)

2. **Un réseau Feed Forward** (FFN)

### 1. Le mécanisme de Self-Attention

#### Q, K, V : trois versions d'un mot

Pour tenir compte du **contexte**, chaque mot est transformé en trois représentations : **Query** **(Q)**, **Key (K)** et **Value (V)**.

- La **Query (Q)** est comme une question qu'un mot pose aux autres.

- La **Key (K)** est sa "carte d'identité".

- La **Value (V)** transporte l'information propre au mot.

Concrètement, on part de l'**embedding** du mot et on le multiplie par trois matrices différentes pour obtenir Q, K et V.

#### Comparer Q et K pour savoir où regarder

Prenons l'exemple : "Je vais chez toi".

Le mot "je" va comparer sa Query avec les Keys des autres mots. On obtient des **scores de similarité** qui indiquent à quel point "je" doit porter attention à "vais", "chez", "toi", etc. Ces scores sont ensuite normalisés (softmax) pour devenir des **poids d'attention**—des nombres entre 0 et 1 qui se répartissent sur l'ensemble des mots.

#### Appliquer ces poids à V pour savoir quoi récupérer

Une fois qu'on sait **où** regarder (grâce à Q et K), on utilise les poids d'attention sur les **Values** pour déterminer **quel** **contenu** le mot doit extraire de ses voisins. Le résultat est un **embedding** **contextuel** qui enrichit la représentation initiale du mot.

#### Masquage (auto-régression)

Lors de la **génération** de texte, le modèle ne doit pas "tricher" en voyant les mots futurs. On masque donc les positions qui suivent, en mettant leurs scores d'attention à `-∞` dans la matrice Q × K. Ainsi, le modèle ne s'appuie que sur les mots déjà connus.

### Un mot sur le multi-head attention

Plutôt que de faire ce calcul d'attention une seule fois, on le réalise en **parallèle** avec plusieurs "têtes" (heads), chacune ayant ses propres matrices Q, K, V. Chaque tête peut ainsi se focaliser sur un **aspect** **spécifique** de la phrase : l'une peut repérer les relations grammaticales ("je" → "vais"), tandis qu'une autre capte des liens sémantiques plus larges ("chez" → "toi"). À la fin, on fusionne les sorties de ces têtes pour obtenir un point de vue plus riche.

### 2. Le réseau Feed Forward (FFN)

Après l'étape de Self-Attention, chaque mot passe individuellement dans un **réseau** **Feed** **Forward**. En pratique, ce bloc FFN consiste généralement en deux transformations linéaires séparées par une activation (ReLU, GELU…). Le tout est appliqué individuellement à chaque token pour affiner les représentations, sans interaction directe entre les mots. Elle affine ou réoriente les représentations produites par l'attention.

### Empilement des couches

Un Transformer ne se contente pas d'une seule couche : il en empile plusieurs (des dizaines dans des modèles comme ChatGPT). Chaque couche reprend le même schéma Self-Attention → FFN, et affine progressivement la compréhension du texte. Aussi chaque bloc de Self-Attention et de Feed Forward est généralement équipé de connexions résiduelles et d'une couche de normalisation (LayerNorm), ce qui aide le modèle à stabiliser l'apprentissage.

## La prédiction du mot suivant

Après la dernière couche, on obtient pour chaque mot un **embedding** **final**. Pour générer la suite d'une phrase, le modèle compare cet embedding à tous les mots du **vocabulaire**. Il calcule alors la **probabilité** que tel ou tel mot soit le plus pertinent pour poursuivre la phrase.

### Les paramètres de génération : Température et Top-k

Une fois ces **probabilités** calculées, le modèle utilise un sampler pour **choisir** le mot à générer. C'est là que la **Température** et le **Top-k** interviennent.

- La **Température** ajuste la "créativité" du modèle. Plus la température est élevée (> 1), plus les choix de mots sont variés (mais parfois moins cohérents). Plus elle est faible (< 1), plus le modèle devient conservateur (il choisit le mot le plus probable).

- Le **Top-k** restreint le choix du modèle aux **k** mots les plus probables. Par exemple, avec Top-k = 3, le modèle ne piochera que dans les 3 mots ayant les meilleures probabilités, ignorant les autres.

Ces paramètres **n'agissent que lors de la sélection finale** du mot. Tout le calcul en amont (embeddings, Self-Attention, FFN…) reste **inchangé**.

En résumé, le Transformer transforme la séquence de tokens en représentations contextuelles via plusieurs couches successives de Self-Attention et de réseaux Feed Forward. Enfin, il génère la probabilité de chaque mot possible et en choisit un (selon la température, le top-k, etc.).
