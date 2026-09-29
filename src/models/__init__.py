"""
Adaptateurs de FAMILLE de modèle.

Le projet sépare deux axes qui étaient jusqu'ici confondus :

  - le BACKEND (src/backends/) : comment exécuter un graphe -- ncnn sur Pi,
    TensorRT sur Jetson, ONNX Runtime sur PC. Il ne sait rien de ce que le
    graphe signifie.
  - la FAMILLE (ce paquet) : ce que le graphe attend en entrée et ce que ses
    sorties veulent dire. YOLO, D-FINE, PaddleOCR n'ont ni le même
    prétraitement, ni le même nombre d'entrées, ni le même format de sortie.

Les confondre revenait à supposer « YOLO » dans le corps même de
`InferenceBackend.detect()` : letterbox, une entrée, une sortie, puis
`decode_detections()`. Cette supposition tient tant qu'on n'a que des modèles
YOLO ; elle casse dès qu'on ajoute un détecteur de type DETR ou un OCR
séquentiel à perte CTC.
"""