# Milestones
# PRIORITE 
- FAIRE CONVERGER 
# Compte rendu Re
- Faire fonctionner et voir convergence 
- Reproduire les resultats sur Reacher-V1 et Inverted-Pendulum sur 100k timsteps.
- Finaliser les TU 
- verifier Documentation 
- Ecrire rapport 
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
* Fichier agents.py -> jitted et modifié, donc accéléré. Dans la version précédente, il semble que les gradients soient nuls.
* Fichier networks.py avec value, critic et actor -> fini (mais pas testé) (N) qui seront ensuite utilisés dans agents.py
* modified the log_sigma value. Omit the softplus, use clipped value instead.

## training

* Fichier train.py -> now use yaml config files for all parameters (saved in ./configs/)
* added deterministic policy for evaluation. Achieved an average score of 25-35 (over 5 evaluation episodes) on Inverted Pendulum environment.

Run code by
```
python train.py --config ./configs/inverted_pendulum.yaml
```

## testing/experiments

* dossier ./tests pour rassembler nos tests 
* reproducing results
* ablation study ?

- TU replay buffer ok 
- TU load_config ok 
- TU envs & agents en cours 

## Écrire le rapport
