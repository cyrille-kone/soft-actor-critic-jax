# Milestones
liste de trucs à faire/coder, je pense on peut mettre l'avancement de ce que chacun est en train de faire, les problèmes qu'on rencontre etc...

## utils
- Replay buffer (N) fini!
- loggging & config file OK [cyrk] 
- types.py pour nos types/structures custom [cryk]
## environment (A)
* inverted pendulum
* Reacher-v1
* Rassembler dans le fichier envs.py [En cours][cyrk]

Dans le notebook : j’arrive à les charger, je les wrap pour en faire des dm_env (pas nécessaire), je fais tourner un random agent dessus, et j’affiche le debut de la simulation.

## agents
* Fichier agents.py -> en cours (N) (le fichier agents.py devient assez gros, on pourra peut-être le découper)
* Fichier networks.py avec value, critic et actor -> fini (mais pas testé) (N) qui seront ensuite utilisés dans agents.py

## training

* loss function 

* backprop (N)

* main loop (N) fini!

## testing/experiments

* dossier ./tests pour rassembler nos tests 
* reproducing results
* ablation study ?

- TU replay buffer ok 
- TU load_config ok 
- TU envs & agents en cours 

## Écrire le rapport
