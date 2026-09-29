
#!/usr/bin/env python3
"""
Mesure du FPS et des temps par étage (détection / OCR / dessin / publication).
 
Deux notions distinctes de FPS, il ne faut pas les confondre :
  - fps_proc  : 1 / temps de traitement d'une frame (capacité brute du modèle)
  - fps_wall  : frames réellement sorties par seconde, frame_skip inclus
                (c'est ce que voit l'utilisateur, et ce qu'il faut afficher)
"""

from __future__ import annotations
 
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict

# FPSMeter : mesure les FPS 
class FPSMeter :
    """Moyenne glissante sur une fenêtre de N intervalles (robuste aux pics)."""
    
    def __init__(self , window:int = 60) :
        self._dt:Deque[float] = deque(maxlen= window)
        self._last : float | None = None
        self.frames = 0
        self._t0 = time.perf_counter()
        
    
    def tick(self) -> float :
        now =time.perf_counter()
        if self._last is not None :
            dt =now - self._last
            if dt > 0:
                self._dt.append(dt)
        self._last = now
        self.frames +=1 
        return self.fps
    
    @property
    def fps(self) -> float:
        """FPS instantané lissé (fenêtre glissante)."""
        if not self._dt:
            return 0.0
        return len(self._dt) / sum(self._dt)
 
    @property
    def fps_avg(self) -> float:
        """FPS moyen depuis le démarrage."""
        elapsed = time.perf_counter() - self._t0
        return self.frames / elapsed if elapsed > 0 else 0.0
 
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0
 
    def snapshot(self) -> Dict[str, float]:
        return {
            "fps": round(self.fps, 2),
            "fps_avg": round(self.fps_avg, 2),
            "frames": self.frames,
            "uptime_s": round(self.elapsed, 1),
        }
        
# StageTimer: mesure le temps de chaqye etape = profiling        
class StageTimer :
    """
     Chronomètre par étage avec lissage exponentiel.
 
        timer = StageTimer()
        with timer("detect"):
            ...
        timer.ms("detect")  -> temps lissé en millisecondes
    
    """
    def __init__(self , alpha:float =0.2):
        """ alpha : lissage exponentiel EMA"""
        self.alpha = alpha 
        # Contient les temps lissés.
        self._ema:Dict[str,float] = {}
        # le dernier temps réel
        self._last:Dict[str,float] = {}
        
    @contextmanager
    def __call__(self, name:str , *args, **kwds):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name , (time.perf_counter() - t0)*1000.0) # une fois le bloc with est terminé Donc le temps est calculé.
        
    def record(self,name:str , ms:float) -> None :
        #on garde le dernier temps
        self._last[name] = ms 
        #On récupère la valeur précédente lissée
        prev = self._ema.get(name)
        #premiere mesure
        self._ema[name] = ms if prev is None else (self.alpha * ms + (1 - self.alpha) * prev)
        
    
    def ms(self , name:str) -> float :
        return self._ema.get(name ,0.0) #retourne la valeur lisse
    
    def last_ms(self , name:str) -> float :
        return self._last.get(name ,0.0) #Retourne la dernière mesure réelle
    
        
    # last_ms = dernière frame
    # ms = moyenne lissée
    
    # Permet de récupérer toutes les statistiques
    def snapshot(self) ->Dict[str , float]:
        return {f"{k}_ms": round(v,1) for k,v in self._ema.items()}
    
  
#THrottle :  limiter la fréquence d'une action 
class Throttle:
    """Limiteur de débit : autorise une action au plus toutes les `period` secondes."""
    def __init__(self , hz:float):
        self.period = 1.0/hz if hz > 0 else float("inf")
        self._next = 0.0
        
    #retourne True si l'action est autorisée.
    def ready(self) -> bool :
        now =time.perf_counter()
        if now >= self._next :
            self._next = now + self.period
            return True
        return False
    
        
  
    
    
    
    
    

    
                
            
    