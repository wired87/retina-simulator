
import numpy as np

class RetinaOpponentSimulator:
    """
    Simuliert die Transformation von optischen Reizen (RGB) über die
    Photorezeptoren (Zapfen L, M, S) bis hin zu den Gegenfarb-Kanälen
    der Ganglienzellen und des CGL (Corpus geniculatum laterale).
    """

    def __init__(self):
        # Transformationsmatrix von sRGB zu LMS-Zapfen-Erregungen
        # (Standardisierte Hunt-Pointer-Estevez Matrix, normiert)
        self.RGB_TO_LMS = np.array([
            [0.313996, 0.639512, 0.046492],  # L-Zapfen (Langwellig / "Rot")
            [0.155372, 0.757894, 0.086701],  # M-Zapfen (Mittelwellig / "Grün")
            [0.017752, 0.109443, 0.872569]   # S-Zapfen (Kurzwellig / "Blau")
        ])

    def rgb_to_lms(self, rgb: np.ndarray) -> np.ndarray:
        """
        Stufe 1: Erregung der 3 Zapfentypen (Photorezeptoren)
        
        :param rgb: RGB-Array im Bereich [0.0, 1.0] (Shape: (3,) oder (N, 3))
        :return: LMS-Zapfensignale [L, M, S]
        """
        rgb = np.clip(rgb, 0.0, 1.0)
        return np.dot(rgb, self.RGB_TO_LMS.T)

    def lms_to_opponent(self, lms: np.ndarray) -> dict:
        """
        Stufe 2: Verrechnung in den Gegenfarb-Kanälen (Ganglienzellen / CGL)
        
        - Helligkeitskanal (Achromatisch): L + M (Schwarz vs. Weiß)
        - Rot-Grün-Kanal:                  L - M (Rot > 0, Grün < 0)
        - Gelb-Blau-Kanal:                 (L + M) - S (Gelb > 0, Blau < 0)
        
        :param lms: LMS-Array [L, M, S]
        :return: Dictionary mit den biologischen Kanal-Signalen
        """
        # Trennung der Zapfenkanäle (unterstützt Skalare und Matrizen)
        L = lms[..., 0]
        M = lms[..., 1]
        S = lms[..., 2]

        # 1. Helligkeitskanal (Luminanz / Schwarz-Weiß)
        luminance = L + M

        # 2. Rot-Grün-Gegenfarbkanal
        # Positive Werte -> Rot-Dominanz, Negative Werte -> Grün-Dominanz
        red_green = L - M

        # 3. Gelb-Blau-Gegenfarbkanal
        # Gelb entsteht erst im Gehirn durch die Addition von L + M
        yellow = L + M
        yellow_blue = yellow - S

        return {
            "Luminanz_WhiteBlack": luminance,
            "RedGreen": red_green,
            "YellowBlue": yellow_blue
        }

    def process_color(self, rgb: list | np.ndarray) -> dict:
        """
        Führt den kompletten biologischen Verarbeitungsschritt für eine Farbe aus.
        """
        rgb_arr = np.array(rgb, dtype=np.float64)
        lms = self.rgb_to_lms(rgb_arr)
        opp = self.lms_to_opponent(lms)

        return {
            "RGB": rgb_arr,
            "LMS_Cones": {
                "L_Rot": lms[..., 0],
                "M_Gruen": lms[..., 1],
                "S_Blau": lms[..., 2]
            },
            "Opponent_Channels": opp
        }


def print_color_analysis(color_name: str, result: dict):
    """Hilfsfunktion zur übersichtlichen Ausgabe der Signale."""
    opp = result["Opponent_Channels"]
    lms = result["LMS_Cones"]
    
    print(f"\n--- Farbanalyse: {color_name} (RGB: {result['RGB']}) ---")
    print(f"  [Zapfen-Reize]   L (Rot): {lms['L_Rot']:.3f} | M (Grün): {lms['M_Gruen']:.3f} | S (Blau): {lms['S_Blau']:.3f}")
    print(f"  [CGL-Signale]")
    print(f"   * Helligkeit (L+M)       : {opp['Luminanz_WhiteBlack']:.3f} (Weiß-Gehalt)")
    
    rg_val = opp['RedGreen']
    rg_dir = "ROT" if rg_val > 0 else ("GRÜN" if rg_val < 0 else "NEUTRAL")
    print(f"   * Rot vs. Grün (L - M)   : {rg_val:+.3f}  --> Tendenz: {rg_dir}")
    
    yb_val = opp['YellowBlue']
    yb_dir = "GELB" if yb_val > 0 else ("BLAU" if yb_val < 0 else "NEUTRAL")
    print(f"   * Gelb vs. Blau ((L+M)-S): {yb_val:+.3f}  --> Tendenz: {yb_dir}")


# =====================================================================
# Beispiel-Anwendung
# =====================================================================
if __name__ == "__main__":
    sim = RetinaOpponentSimulator()

    # Testfarben (RGB normiert auf [0, 1])
    test_colors = {
        "Cyan": [0.0, 1.0, 1.0],      # Wichtig für Ihre Produktionsüberlegung
        "Pure Yellow": [1.0, 1.0, 0.0],
        "Pure Red": [1.0, 0.0, 0.0],
        "Pure Blue": [0.0, 0.0, 1.0],
        "Neutral Gray": [0.5, 0.5, 0.5]
    }

    for name, rgb in test_colors.items():
        res = sim.process_color(rgb)
        print_color_analysis(name, res)
