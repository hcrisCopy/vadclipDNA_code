# CTNC-VAD package notes

The maintained method design, exact XD command sequence, output locations, and
metric format are in the repository-level [README](../README.md).  Keep this
package README intentionally short so that there is only one authoritative set
of runnable commands.

`ctnc_vad` is a sidecar: it reads cached CLIP hidden states and the output of a
frozen VadCLIP baseline.  It never imports project code outside this repository
and never writes into `VadCLIP/`.  Its audit records name every used CLIP
coordinate as an original `(layer, dimension)` pair.
