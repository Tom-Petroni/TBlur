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

- build les versions Nuke 13.0 -> 17.0
- build Windows + Linux (pas de build macOS pour la variante CUDA)
- genere un zip de release pret a copier dans `.nuke`

## Installer dans Nuke

1. Copier `publish/tblur_plugin` vers `C:/Users/<user>/.nuke/tblur_plugin`
2. Dans `C:/Users/<user>/.nuke/init.py`, ajouter:

```python
import nuke
nuke.pluginAddPath(r"C:/Users/<user>/.nuke/tblur_plugin")
```

3. Redemarrer Nuke

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
