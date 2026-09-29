# Export TensorRT et tests sur Jetson


**Un engine TensorRT n'est pas un fichier portable.** Il est compilé pour une
architecture GPU précise et pour la version exacte de TensorRT installée.
Construire un engine sur le PC et le copier sur le Jetson ne marche pas : il
refuse de se désérialiser, avec un message sur la version. Il n'y a pas de
contournement — l'étape `engine` tourne **sur le Jetson**, et uniquement là.

D'où le découpage en deux étapes :

| étape | où | pour quoi |
|---|---|---|
| `onnx` | n'importe où (PC de dev) | uniquement les `.pt` Ultralytics |
| `engine` | **sur le Jetson** | n'importe quel `.onnx` |

D-FINE et PaddleOCR sautent la première : leur ONNX existe déjà. L'étape
`onnx` passe par `ultralytics.YOLO`/`RTDETR`, qui ne savent charger qu'un
checkpoint Ultralytics — même si D-FINE est bâti sur RT-DETR, ce sont deux
implémentations différentes et le chargement échoue.

## Où l'engine est écrit

C'est le `config.yaml` qui décide, pas le script. `--section` nomme la section
de `models:`, `--precision` le niveau, et la destination est lue dans le
`paths.tensorrt` de cette section avec `{precision}` substitué :

```
models.detector.paths.tensorrt = models/exported/tensorrt/yolo/{precision}/detector
--section detector --precision fp16
    -> models/exported/tensorrt/yolo/fp16/detector/model.engine
```


## Vérifier ta version avant de commencer

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
/usr/src/tensorrt/bin/trtexec --help | head -3
```

Les seuils qui comptent pour ce projet, à confronter à ce que tu lis :

| ce qu'il faut | version TensorRT | concerné |
|---|---|---|
| `GridSample` natif | 8.5 et plus | D-FINE |
| `TopK` à K dynamique | 8.6 et plus | D-FINE |
| entrées INT64 sans conversion | 9.0 et plus | `orig_target_sizes` de D-FINE |
| `--memPoolSize` | 8.4 et plus | tous |
| `--workspace` | supprimé en 10 | tous |

En dessous de TensorRT 9, une entrée INT64 est ramenée en INT32 avec un
avertissement — pour `orig_target_sizes`, qui contient une largeur et une
hauteur, la perte est sans conséquence. Le script choisit tout seul entre
`--memPoolSize` et `--workspace` en lisant l'aide de `trtexec`, il n'y a rien
à régler.

Un Jetson Nano de première génération plafonne à JetPack 4.6.1, donc TensorRT
8.2.1 : **D-FINE n'y est pas convertible**, les deux premiers seuils ne sont
pas atteints. Sur ce matériel-là, le détecteur reste YOLOv8n.

## Les commandes, modèle par modèle

Depuis la racine du projet, sur le Jetson.

### YOLOv8n détecteur

```bash
# étape onnx, sur le PC
python3 scripts/export_jetson.py onnx --weights yolov8n_detect.pt --imgsz 640

# étape engine, sur le Jetson
python3 scripts/export_jetson.py engine --onnx yolov8n_detect.onnx \
    --section detector --precision fp16
```

### YOLO11n OCR

```bash
python3 scripts/export_jetson.py onnx --weights yolo11n_ocr.pt --imgsz 608
python3 scripts/export_jetson.py engine --onnx yolo11n_ocr.onnx \
    --section ocr --precision fp16
```

L'`imgsz` de l'export doit valoir celle du `config.yaml` (640 pour le
détecteur, 608 pour l'OCR). L'export Ultralytics écrit un `metadata.yaml` à
côté qui porte la vraie valeur, et c'est lui qui fait foi au chargement.

### D-FINE-N

Pas d'étape `onnx` : le fichier existe. Deux entrées, dont une dimension de lot
dynamique, donc le script construit un profil d'optimisation tout seul.

```bash
python3 scripts/export_jetson.py engine \
    --onnx models/dfine_n/best_stg2_single.onnx \
    --section detector_dfine --precision fp16
```

Ce qu'il produit :

```
--minShapes=images:1x3x640x640,orig_target_sizes:1x2
--optShapes=images:1x3x640x640,orig_target_sizes:1x2
--maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```

Sans ces trois drapeaux, `trtexec` s'arrête sur *« Network has dynamic or
shape inputs, but no optimization profile has been defined »*.

### PaddleOCR

```bash
python3 scripts/export_jetson.py engine \
    --onnx models/paddleocr/paddleocr_rec_fixed.onnx \
    --section ocr_paddle --precision fp32 --shape x:1x3x48x320
```
L'entrée 'x' a une dimension dynamique sur l'axe 3 que ce script ne peut
pas deviner.
Donne-la explicitement, par exemple :
    --shape x:1x3x48x<taille>...
```



## Les trois modes de précision

`--precision` pilote **à la fois** la précision et le dossier de sortie. Un
drapeau qui la contredit est une erreur, pas un réglage silencieux :

```
--precision fp32 --fp16   ->  refusé
--precision fp16 --int8   ->  refusé
--precision int8 --fp16   ->  accepté : INT8 avec repli fp16
```

### fp32 — la référence

```bash
python3 scripts/export_jetson.py engine --onnx m.onnx --section detector --precision fp32
```

Le plus lent, mais c'est la vérité numérique. Exporte-le en premier : c'est à
lui que tu compares les lectures des autres, et sans cette référence tu ne sais
pas si une dégradation vient de la quantification ou d'autre chose.

### fp16 — le mode par défaut sur Jetson

```bash
python3 scripts/export_jetson.py engine --onnx m.onnx --section detector --precision fp16
```

C'est le bon réglage sur Orin, qui a des unités fp16 dédiées. Le gain est réel
et la perte de précision négligeable sur de la détection. Contrairement au Pi,
où la quantification ne rapporte rien, ici elle rapporte.

### int8 — seulement avec une calibration

```bash
python3 scripts/export_jetson.py engine --onnx m.onnx --section detector \
    --precision int8 --fp16 --calib-cache calib/detector.cache
```

Le `--calib-cache` est **obligatoire** et le script le refuse sans. Ce n'est pas
de la rigidité : sans calibrateur, `trtexec` construit quand même un engine, avec
des échelles arbitraires. Il se charge, il est rapide, et il lit faux. Rien
dans la sortie ne le signale.

Deux choses à savoir sur ce cache :

La calibration doit utiliser **le même prétraitement que l'inférence**, et il
diffère par modèle — letterbox pour le détecteur YOLO, étirement carré pour
l'OCR par détection, hauteur fixe et normalisation `[-1,1]` en BGR pour
PaddleOCR. Un cache produit avec le mauvais prétraitement décrit une
distribution qui n'est pas celle que le modèle verra.

`--fp16` ajouté à `--precision int8` autorise le repli : les couches que
TensorRT ne quantifie pas retombent en fp16 au lieu de fp32. C'est presque
toujours ce qu'on veut sur Orin.

## Avant de mesurer quoi que ce soit

Sur Jetson, les chiffres ne veulent rien dire si le module n'est pas en mode
pleine puissance. Un même engine peut varier du simple au double entre le mode
économique et le mode maximal.

```bash
sudo nvpmodel -q              # quel mode est actif
sudo nvpmodel -m 0            # mode maximal (l'indice dépend du module)
sudo jetson_clocks            # fige les fréquences au maximum
```

`jetson_clocks` désactive le gouverneur dynamique : sans lui, les premières
images tournent à basse fréquence et la latence mesurée est fausse. Surveille
la température et le throttling pendant la mesure avec `tegrastats` ou
`jtop`.

## Vérifier l'engine produit

Le script affiche la commande avant de l'exécuter, et `--dry-run` la montre
sans rien construire — utile depuis le PC pour relire ce qui va se passer :

```bash
python3 scripts/export_jetson.py engine --onnx m.onnx \
    --section detector --precision fp16 --dry-run
```

Une fois l'engine écrit, le contrôle avant lancement le voit :

```bash
python3 scripts/verifier_pi.py --backend tensorrt --precision fp16
```

Puis le pipeline :

```bash
python3 main.py --edge_type jetson --source video --input clip.mp4 \
    --backend tensorrt --precision fp16
```

Sur `--edge_type jetson`, `tensorrt` est déjà le moteur visé par défaut :
`--backend` n'est nécessaire que pour forcer autre chose.

Pour comparer les trois précisions sur la même vidéo :

```bash
for p in fp32 fp16 int8; do
  echo "=== $p ==="
  python3 main.py --edge_type jetson --source video --input clip.mp4 \
      --backend tensorrt --precision "$p"
done
```

Le dashboard reçoit les latences de chaque passage ; le bouton **Exporter en
CSV** donne les passages en ordre chronologique, ce qui suffit à comparer les
lectures d'une précision à l'autre sur les mêmes plaques.

## Ce qui n'est pas couvert

Le DLA des Orin (`--useDLACore=0 --allowGPUFallback`) n'est pas exposé par le
script. Il peut décharger le GPU, mais tous les opérateurs n'y passent pas et
le repli GPU peut coûter plus qu'il ne rapporte. À mesurer avant d'y investir.

La génération du cache de calibration INT8 n'est pas dans ce script non plus :
elle demande un calibrateur Python qui alimente TensorRT avec des images
prétraitées exactement comme à l'inférence. C'est le point délicat, et c'est
pour ça que le script exige un cache plutôt que d'en fabriquer un approximatif.
