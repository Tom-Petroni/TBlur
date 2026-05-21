# TBlur - Documentation Node (FR)

## 1. Objet

`TBlur` est un node Nuke de blur preserveur de contours.

Il sert a:

- lisser les textures/noises
- garder les bords importants
- fournir un rendu proche d'un edge-aware blur de compo haut de gamme

Node:

- classe: `TBlur`
- menu: `Filter/TBlur`

## 2. Inputs

- Input 0 `Source`: image a filtrer (obligatoire)
- Input 1 `Guide`: guide optionnel
- Input 2 `mask`: masque optionnel

Regles:

- si `Guide` n'est pas connecte, le guide est derive de `Source`
- si `mask` n'est pas connecte, le masque vaut `1`
- le mask lit l'alpha en priorite, sinon `red`

## 3. Architecture Runtime

Pipeline principal:

1. lecture source/guide/mask dans la zone demandee
2. construction d'un guide luma
3. construction edge-map perceptuelle (Lab + anti-speckle + hard-stop)
4. filtrage edge-aware:
   - CUDA si `Use GPU` est actif et CUDA disponible
   - fallback CPU sinon
5. blend final via `mix * mask`
6. sortie RGBA (alpha source preservee)

Backend:

- GPU: CUDA C++ (`TBlur_cuda.cu`)
- CPU: domain transform separable (`domain_transform_cpu`)

## 4. Knobs

### Filter

- `Local GPU: ...`: info carte GPU detectee au chargement du node
- `Use GPU if available`: active CUDA
- `Vectorize on CPU`: active le mode CPU multi-thread
- `Safety Rails`: garde-fous anti-artefacts sur reglages extremes
- `Presets`: presets artistiques preconfigures
- `Blur Type`: `Sharp Edges` / `Soft Edges`
- `Filter`: `Gauss` / `Box`
- `Blur Size` (XY/WH): controle X et Y separes
  - slider visible `0..100`
  - valeurs > 100 autorisees en saisie manuelle
- `Edge Threshold` (`0..1`)
- `Edge Smooth` (`0..1`)
- `Guide Influence` (`0..1`)
- `Iterations` (`1..16`)

### Guide

- `guide mode`: `Luma` / `RGB`
- `show guide`: affiche le guide au lieu du filtre

### Output

- `mix` (`0..1`): blend source/filtre
- `Keep Alpha`: garde l'alpha source si actif, sinon l'alpha est filtre edge-aware comme le RGB
- `invert`: inverse le mask

## 5. Presets

`Preset` applique un setup rapide des knobs de style.

- `Default`
- `Classic Cartoon`
- `DSLR Clean`
- `DSLR Low Light`
- `Grain Remove`
- `iPhone Low Light`
- `iPhone+DSLR Mix`
- `Sharp Gauss`
- `Sharp Smooth`
- `Soft Gauss`
- `Softer Smooth`
- `Watercolor`
- `Wrinkle Remover`
- `Custom`

Des qu'un knob de style est modifie, le preset repasse automatiquement en `Custom`.
Dans la version actuelle, tout changement de parametre utilisateur fait repasser le preset en `Custom`.

## 6. Comportements importants

- `mix = 0` => bypass
- `Blur Size = 0` => bypass
- `show guide = on` => pas de filtrage, preview guide
- si CUDA plante/indisponible => fallback CPU automatique

## 7. Performance

Pour de meilleures perfs:

- activer `Use GPU if available`
- limiter `Iterations` au strict necessaire
- commencer avec `Blur Size` XY identiques, puis anisotropie si besoin
- utiliser `Soft` seulement si besoin rendu

Notes:

- les plans tres contrastes + blur extreme restent couteux
- le mode CPU est robuste mais plus lent
- `Iterations = 1` est protege par des garde-fous anti-tramage sur le chemin CUDA

## 8. Troubleshooting

### Le node reste en CPU

Verifier:

1. logs backend (`backend_runtime.log`)
2. build CUDA fait avec `--cuda-backend`
3. DLL bien remplacee dans `.nuke`
4. Nuke relance completement

### Artefacts visuels (banding/lignes)

Le kernel CUDA inclut deja des protections anti-banding.
Si un cas limite apparait encore:

1. reduire `Blur Size`
2. reduire `Iterations`
3. activer `Safety Rails`
4. comparer CPU/GPU pour isoler la source

## 9. Build et Deploy (Windows)

Build:

```bash
cargo xtask --compile --nuke-versions 16.0 --target-platform windows --output-to-package --limit-threads --cuda-backend
```

DLL produite:

`TBlur_plugins/TBlur_plugin/bin/16.0/windows/x86_64/TBlur.dll`

Deploy `.nuke`:

```powershell
Copy-Item -LiteralPath "C:\Users\<user>\Documents\Dev\TBlur\TBlur_plugins\TBlur_plugin\bin\16.0\windows\x86_64\TBlur.dll" -Destination "C:\Users\<user>\.nuke\TBlur_plugin\bin\16.0\windows\x86_64\TBlur.dll" -Force
```

Puis relancer Nuke.

