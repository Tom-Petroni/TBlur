# TBlur — Audit du code vs. design documenté

Date : 2026-05-05
Scope : `crates/tblur-nuke/src/` à l'état actuel sur disque.

## Contexte

Le projet est documenté comme un nœud Nuke **full-GPU** issu d'un refactor le
2026-05-03, censé fixer un lag de scrubbing de slider via 4 « perf levers »
(pinned host staging, deinterleave kernels GPU, `std::mutex` au lieu de
spinlock, dédup d'upload par hash).

Côté utilisateur : épisodes récurrents de **frames fantômes** quand le viewer
bougeait (zoom, pan, ROI) pendant que le player tournait. Pour s'en débarrasser,
les caches d'upload ont été supprimés. C'est un point clé pour relire la suite —
ça explique une partie (mais pas tout) de l'écart entre la doc et le code.

## Ce qui est solide

- Découpe propre :
  - [tblur_base.cpp](crates/tblur-nuke/src/tblur_base.cpp) (1482 l.) — nœud,
    knobs, cache full-frame, dispatch.
  - [tblur_cuda.cu](crates/tblur-nuke/src/tblur_cuda.cu) (1550 l.) — pipeline
    kernels.
  - ABI C minimaliste dans [tblur_cuda.h](crates/tblur-nuke/src/tblur_cuda.h).
  - Shim Rust trivial dans [lib.rs](crates/tblur-nuke/src/lib.rs).
- Hash de cache exhaustif dans
  [`append()`](crates/tblur-nuke/src/tblur_base.cpp:853) — l'invalidation est
  correcte sur tous les knobs et sur la connexion guide/mask.
- Wrapping SEH `__try/__except` autour des entrées CUDA pour ne pas crasher Nuke
  ([safe_cuda_*](crates/tblur-nuke/src/tblur_base.cpp:54)) — défensif et utile
  sur Windows.
- Warm-up CUDA paresseux derrière `std::once_flag`
  ([warm_cuda_runtime_once](crates/tblur-nuke/src/tblur_base.cpp:90)) — pas de
  coût de premier dispatch sur la première frame interactive.
- Pipeline GPU bien fusionné, en un seul stream :
  edge map → despeckle 3-pass → N×(H,V) domain transform avec ping-pong sur
  `d_filter_a/b` → blend mask×mix → cleanup organique. Lab ΔE76 +
  multi-scale gradient pour la carte d'edges.
- Cache full-frame keyé par `Iop::hash() + frame + view + format box` — évite
  les coutures ROI quand on zoome/pan, ce qui est exactement la classe de bugs
  qu'un nœud full-GPU à state global doit éviter.

## Écart majeur — les 4 « perf levers » documentés ne sont pas dans le code

`ARCHITECTURE.md` et la mémoire interne prétendent qu'au refactor du 2026-05-03,
4 leviers ont été appliqués pour fixer le lag de scrub. Vérification ligne par
ligne :

| Promesse | Réalité dans le code |
|---|---|
| `cudaHostAlloc` pinned host staging (H2D et D2H) | **Absent.** Aucun `cudaHostAlloc`/`cudaMallocHost` dans le crate. Les `cudaMemcpyAsync` (lignes [1301](crates/tblur-nuke/src/tblur_cuda.cu:1301), [1313](crates/tblur-nuke/src/tblur_cuda.cu:1313), [1322](crates/tblur-nuke/src/tblur_cuda.cu:1322), [1350](crates/tblur-nuke/src/tblur_cuda.cu:1350), [1359](crates/tblur-nuke/src/tblur_cuda.cu:1359), [1438](crates/tblur-nuke/src/tblur_cuda.cu:1438), [1529](crates/tblur-nuke/src/tblur_cuda.cu:1529)) copient depuis de la mémoire pageable → l'`Async` dégénère silencieusement en synchrone côté driver. La bande passante PCIe effective est divisée par ~2. |
| Channel-major deinterleave kernels GPU pour les inputs non-RGBA-packed | **Absent.** Aucun kernel `deinterleave_*` dans `tblur_cuda.cu`. Le scatter se fait sur l'hôte dans la double boucle `for (y) for (x)` de [extract_rgba_from_plane](crates/tblur-nuke/src/tblur_base.cpp:330-340), [extract_single_channel_from_plane](crates/tblur-nuke/src/tblur_base.cpp:354), [extract_rgb_from_plane](crates/tblur-nuke/src/tblur_base.cpp:420). C'est exactement le `parallel_for_rows` que le commentaire « Do not try to add it back » interdit. |
| `std::mutex` au lieu de spinlock | **Faux.** Le code a toujours `static std::atomic_flag g_cuda_lock` ([cuda.cu:1035](crates/tblur-nuke/src/tblur_cuda.cu:1035)) et `SpinLockGuard` à chaque entrée publique ([1200](crates/tblur-nuke/src/tblur_cuda.cu:1200), [1205](crates/tblur-nuke/src/tblur_cuda.cu:1205), [1226](crates/tblur-nuke/src/tblur_cuda.cu:1226), [1263](crates/tblur-nuke/src/tblur_cuda.cu:1263), [1538](crates/tblur-nuke/src/tblur_cuda.cu:1538)). Sous contention multi-thread Nuke, ça spin sur le thread UI. |
| Per-input upload hash tracking | **Code mort, retiré sciemment.** Les champs `source_sig`, `guide_input_sig`, `mask_input_sig`, `guide_luma_direct_sig`, `mask_direct_sig` existent toujours dans `CudaBackend` ([cuda.cu:1019-1032](crates/tblur-nuke/src/tblur_cuda.cu:1019)) mais chaque site d'upload force `b.source_sig = 0ull;` puis upload inconditionnellement, avec commentaire explicite « Always upload to avoid stale-frame artifacts » ([1300](crates/tblur-nuke/src/tblur_cuda.cu:1300), [1312](crates/tblur-nuke/src/tblur_cuda.cu:1312), [1321](crates/tblur-nuke/src/tblur_cuda.cu:1321), [1349](crates/tblur-nuke/src/tblur_cuda.cu:1349), [1358](crates/tblur-nuke/src/tblur_cuda.cu:1358)). La fonction [sampled_host_signature](crates/tblur-nuke/src/tblur_cuda.cu:51) qui devait alimenter ce système n'est appelée nulle part. |

### Quel rapport avec les frames fantômes

Le **4ᵉ point seulement** s'explique par ton problème de frames fantômes :
le hash sample-based (1024 floats échantillonnés sur toute la frame) est par
construction trop lâche — pendant un play+zoom, deux frames adjacentes ou deux
ROIs partielles peuvent donner le même sample-hash, et la dédup réutilisait
alors la frame précédente sur la GPU. La suppression du cache d'upload est
le bon réflexe pour faire disparaître les ghosts.

Mais **les trois autres leviers sont indépendants du problème de staleness** :

- **Pinned staging** (`cudaHostAlloc`) : zéro impact sur la fraîcheur des
  données. C'est juste de la bande passante PCIe. Aucune raison de l'avoir
  retiré pour fixer les ghosts.
- **Deinterleave GPU** : pareil. C'est un kernel qui consomme l'`ImagePlane`
  Nuke et écrit dans `d_source_rgba`. Le contenu est strictement le même que
  ce que produit le scatter hôte ; seul le coût change.
- **`std::mutex` vs spinlock** : ne touche pas au pipeline de données.

→ Conclusion : l'épisode « ghost frames » a probablement déclenché un revert
plus large que nécessaire, qui a aussi balayé deux optimisations propres
(staging pinned, deinterleave) et le passage au mutex. Si le scrub est lent,
ces trois-là peuvent revenir sans rouvrir le bug de staleness.

## Autres points

### `Op::error()` promis, jamais appelé
`ARCHITECTURE.md` dit : « On CUDA failure → `Op::error()` with the underlying
CUDA error string ». Recherche dans
[tblur_base.cpp](crates/tblur-nuke/src/tblur_base.cpp) : aucune occurrence de
`error(`. Le fallback ([1443-1454](crates/tblur-nuke/src/tblur_base.cpp:1443))
écrit juste `%USERPROFILE%\.nuke\TBlur\backend_runtime.log` et passe la source
telle quelle. **L'utilisateur n'a aucun signal visuel que TBlur ne tourne pas.**
Si CUDA échoue silencieusement (driver crash, OOM…), tu vois ton input non
filtré et tu n'as pas d'indice — il faut ouvrir le log fichier.

### Code mort (~730 l.)
[tblur_blink_native.cpp](crates/tblur-nuke/src/tblur_blink_native.cpp) (503 l.)
et [tblur_blink_kernels.h](crates/tblur-nuke/src/tblur_blink_kernels.h)
(231 l.) ne sont **pas compilés** —
[build.rs](crates/tblur-nuke/build.rs:108) ne référence que `tblur_base.cpp`.
À supprimer, ou à réintégrer si l'idée Blink/PlanarIop reste sur la roadmap.

### AOV/extra channels en boucle scalaire
[1357-1427](crates/tblur-nuke/src/tblur_base.cpp:1357) traite chaque channel
extra (au-delà de RGBA) en repassant toute la chaîne CUDA — edge map +
despeckle + N iter de filtre. Pour 4 AOVs c'est × 4 le coût d'une frame.
Comme l'edge map et le guide ne dépendent que du source/guide RGB, ils
pourraient être calculés une seule fois et réutilisés ; seul le filtre
gagnerait à être batché en kernel scalaire.

### `cudaStreamSynchronize` bloquant à chaque frame
[cuda.cu:1532](crates/tblur-nuke/src/tblur_cuda.cu:1532) sync à la fin de
`cuda_process`. Combiné au spinlock global, plusieurs threads Nuke en attente
peuvent geler le thread UI pendant la durée d'une frame complète sur la GPU.

### SEH trop large
Les `__try/__except(EXCEPTION_EXECUTE_HANDLER)`
([base:58, 70, 162](crates/tblur-nuke/src/tblur_base.cpp:58)) attrapent *tout*
— y compris les bugs de notre propre code (segfault dans un kernel, AV sur
pointeur nul…), pas seulement les fautes du driver. À long terme ça masquera
des régressions silencieusement. Filtrer sur `EXCEPTION_ACCESS_VIOLATION` et
laisser le reste remonter serait plus sain.

### `_request` demande tout le format
[709-740](crates/tblur-nuke/src/tblur_base.cpp:709) demande
`fmt.x()..fmt.r()` × `fmt.y()..fmt.t()` indépendamment de la ROI entrante.
Défendable pour un edge-aware blur (les edges au bord de la ROI dépendent de
pixels en dehors), mais coûteux en zoom serré sur 4K/8K.

## Recommandations classées

### Si le scrub est actuellement OK
Ne pas toucher au pipeline. Mais :
1. Mettre à jour [ARCHITECTURE.md](ARCHITECTURE.md) et la doc pour qu'elle
   reflète le code réel (pas les 4 leviers, pas d'`Op::error()`).
2. Supprimer le code mort `tblur_blink_*` ou l'intégrer.
3. Supprimer les champs `*_sig`/`*_uploaded` inutilisés et la fonction
   [sampled_host_signature](crates/tblur-nuke/src/tblur_cuda.cu:51) — c'est
   du bruit qui suggère faussement qu'une dédup existe.

### Si le scrub redevient lent
Réintroduire dans cet ordre, en testant les ghosts à chaque étape :
1. **Pinned staging H2D et D2H** (`cudaHostAlloc` sur des buffers
   `width*height*4` réutilisés). Indépendant des ghosts. Gain attendu : ~×2 sur
   le transfert.
2. **Deinterleave kernels GPU** pour remplacer les boucles `for (y) for (x)`
   d'extract_*. Indépendant des ghosts. Gain attendu : élimine le pic CPU
   par-frame qui rivalise avec le thread UI.
3. **`std::mutex` à la place du spinlock**. Indépendant des ghosts. Gain
   attendu : sous N threads Nuke en attente, un seul tourne, les autres
   dorment.
4. **Dédup d'upload** : ne pas réintroduire la version sample-based — c'était
   la cause des ghosts. Si dédup il y a, la clé doit être `Iop::hash()` du
   nœud d'entrée, pas un hash échantillonné du contenu.

### Hygiène
- Réellement appeler `error("TBlur: CUDA failed: %s", …)` sur le path d'échec,
  pour que le statut soit visible dans le node graph.
- Resserrer le SEH pour ne pas avaler les fautes de notre code.

## Annexe — fichiers consultés
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [tblur_base.cpp](crates/tblur-nuke/src/tblur_base.cpp)
- [tblur_cuda.cu](crates/tblur-nuke/src/tblur_cuda.cu)
- [tblur_cuda.h](crates/tblur-nuke/src/tblur_cuda.h)
- [build.rs](crates/tblur-nuke/build.rs)
- [Cargo.toml](crates/tblur-nuke/Cargo.toml)
- [lib.rs](crates/tblur-nuke/src/lib.rs)
- [tblur_blink_native.cpp](crates/tblur-nuke/src/tblur_blink_native.cpp) (non compilé)
- [tblur_blink_kernels.h](crates/tblur-nuke/src/tblur_blink_kernels.h) (non compilé)
