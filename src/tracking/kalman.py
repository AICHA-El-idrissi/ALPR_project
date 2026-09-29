#!/usr/bin/env python3
"""
Filtre de Kalman minimal pour le suivi de boîtes englobantes.

État : [cx, cy, w, h, vcx, vcy, vw, vh]
Modèle de mouvement : vitesse constante.
"""
import numpy as np


class KalmanBox:

    def __init__(self, xyxy: np.ndarray):
        cx, cy, w, h = self._to_cwh(xyxy)
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8) * 10.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.Q = np.eye(8) * 0.5
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
        self.R = np.eye(4) * 1.0

    @staticmethod
    def _to_cwh(xyxy):
        x1, y1, x2, y2 = xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.as_xyxy()

    def update(self, xyxy: np.ndarray) -> None:
        z = np.array(self._to_cwh(xyxy))
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        #K = self.P @ self.H.T @ np.linalg.inv(S)
        
        K= np.linalg.solve(S.T , (self.P @ self.H.T).T).T
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def as_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.state[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
