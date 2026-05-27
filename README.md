# TBlur

TBlur est un node Nuke GPU (CUDA) pour un blur edge-preserving, pense pour des workflows stylises et des pipelines compo lourds.

## Pourquoi TBlur

- Filtrage edge-aware natif
- Pipeline GPU-first (CUDA)
- Integration Nuke via package Python + binaire natif
- Build cible Nuke multi-version

## Structure du repo

```text
TBlur/
  publish/        # payload a copier dans .nuke
  work/           # source rust/c++/cuda + tooling build
  node.json
  VERSION
  CHANGELOG.md
```

## Prerequis

- Nuke installe (version cible)
- Rust/Cargo
- CUDA Toolkit
- GPU NVIDIA compatible
- MSVC / Visual Studio Build Tools (Windows)

## Compiler

Depuis la racine du repo (shell initialise MSVC/CUDA):

```powershell
cd work
cargo xtask --compile --nuke-versions 16.0 --target-platform windows --output-to-package --limit-threads --cuda-backend
```

Pour Nuke 17.0, remplacer `16.0` par `17.0`.

## Build CI GitHub

Le repo contient un workflow GitHub Actions (`.github/workflows/nuke-build.yml`) qui:

- build les versions Nuke 16.0 -> 17.0
- build Windows + Linux (pas de build macOS pour la variante CUDA)
- genere un zip de release pret a copier dans `.nuke`

## Installer dans Nuke (utilisateur final)

1. Cloner le repo
2. Copier le contenu de `publish/` dans `C:/Users/<user>/.nuke/`:
   - `publish/init.py`
   - `publish/tblur_plugin/`
3. Redemarrer Nuke

Les binaires (`.dll`, `.so`) sont versionnes dans `publish/tblur_plugin/bin/...`.

Si tu as deja un `.nuke/init.py`, fusionne simplement la ligne suivante dedans:

```python
import nuke
nuke.pluginAddPath("./tblur_plugin")
```

## Verification rapide

- Le menu `Nodes > TBlur` apparait
- Le binaire est present dans:
  `tblur_plugin/bin/<nuke_version>/<os>/<arch>/`

## Docs techniques

- `work/docs/TBLUR_NODE_GUIDE.md`
- `work/docs/TBLUR_DOCUMENTATION_FR.md`
- `work/ARCHITECTURE.md`

## Licence

Usage commercial soumis a la licence du repo (`LICENSE` + `EULA.md`).
