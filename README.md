# wa-knn-fingerprint-securedrop

An implementation of Tao Wang's Wa-kNN classifier intended for integration with
the https://github.com/freedomofpress/fingerprint-securedrop machine learning
pipeline. This classifier was designed for and has found to be particularly well
suited for website fingerprinting because of it's multi-modality (which
compensates for localization), avoidance of local minima, and multi-class
classification (yet to be implemented here--this is only a binary classifier
distinguishing between "monitored" and "non-monitored" classes for now).

Based somewhat on https://github.com/pylls/go-knn and also [Wang's original CPP
implementation](https://crysp.uwaterloo.ca/software/webfingerprint/knn.zip),
herein we have implemented a JSON Interface in order to enable interop with
https://github.com/freedomofpress/fingerprint-securedrop. See the
[`integrate-wa-knn`](https://github.com/freedomofpress/fingerprint-securedrop/tree/integrate-wa-knn)
branch for corresponding work there. This version also fixes a couple bugs in
former implementations:

1. https://gitter.im/freedomofpress/Website_Fingerprinting?at=58af8d74e961e53c7f5c6ddd
2. Corrects deltaW factor `pointBadness / recoPointsNum + 0.2` to `(1 +
   pointBadness) / recoPointsNum`, as described in [Wang's
   thesis](https://uwspace.uwaterloo.ca/bitstream/handle/10012/10123/Wang_Tao.pdf)
   section 3.2.5.
