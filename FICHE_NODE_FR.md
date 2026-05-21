# Fiche Node - TBlur

## Resume

TBlur est un node Nuke natif GPU (CUDA) pour blur edge-preserving.

## Prerequis

- Nuke installe (version cible)
- Rust/Cargo
- CUDA Toolkit + GPU NVIDIA compatible
- MSVC/Visual Studio Build Tools (Windows)

## Compiler

Depuis la racine du repo (shell initialise pour MSVC/CUDA):

```powershell
cd work
cargo xtask --compile --nuke-versions 16.0 --target-platform windows --output-to-package --limit-threads --cuda-backend
```

Pour Nuke 17.0, remplacer `--nuke-versions 16.0` par `17.0`.

## Installer dans Nuke

1. Copier `publish/tblur_plugin` dans `C:/Users/<user>/.nuke/tblur_plugin`
2. Ajouter dans `C:/Users/<user>/.nuke/init.py`:

```python
import nuke
nuke.pluginAddPath(r"C:/Users/<user>/.nuke/tblur_plugin")
```

3. Relancer Nuke

## Verification

- Verifier la presence de `Nodes > TBlur`
- Verifier que le binaire existe dans `tblur_plugin/bin/<nuke_version>/<os>/<arch>/`
