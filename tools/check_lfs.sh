#!/bin/sh
# **Sanity check LFS** (v0.9.5++) — vérifie que les assets binaires
# sont bien fetchés et pas en mode pointer.  Lance après tout `git pull`
# ou `git checkout` si tu vois des bugs de chargement d'assets.
#
# Usage : ./tools/check_lfs.sh

cd "$(git rev-parse --show-toplevel)" || exit 1

# Cherche les fichiers LFS en mode pointer (commencent par "version https://git-lfs").
pointers=$(find assets -name "*.glb" -o -name "*.tga" -o -name "*.pk3" -o -name "*.bsp" 2>/dev/null \
  | xargs -I {} sh -c 'head -c 30 "{}" 2>/dev/null | grep -l "git-lfs" >/dev/null && echo "{}"' \
  2>/dev/null)

if [ -n "$pointers" ]; then
    echo "⚠️  Fichiers LFS en mode pointer (non fetchés) :"
    echo "$pointers"
    echo ""
    echo "Lance : git lfs pull"
    exit 1
fi

echo "✅ Tous les assets LFS sont fetchés."
