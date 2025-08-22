#!/usr/bin/env python3
"""
Script simple pour extraire le fichier zip ASL
"""

import zipfile
import pathlib

def extract_asl_data():
    zip_path = pathlib.Path("data/asl_alphabet/asl-alphabet.zip")
    
    if not zip_path.exists():
        print(f"❌ Fichier introuvable: {zip_path}")
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraire dans le dossier parent
            zip_ref.extractall(zip_path.parent)
            print(f"✅ Extraction réussie!")
            print(f"📁 Fichiers extraits: {len(zip_ref.infolist())}")
            return True
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        return False

if __name__ == "__main__":
    extract_asl_data()
